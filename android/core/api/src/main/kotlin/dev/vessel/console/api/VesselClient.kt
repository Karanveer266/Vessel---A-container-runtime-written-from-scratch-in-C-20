package dev.vessel.console.api

import dev.vessel.api.v1.VesselGrpcKt.VesselCoroutineStub
import dev.vessel.api.v1.containerRef
import dev.vessel.api.v1.createContainerRequest
import dev.vessel.api.v1.execRequest
import dev.vessel.api.v1.getHostInfoRequest
import dev.vessel.api.v1.listContainersRequest
import dev.vessel.api.v1.removeContainerRequest
import dev.vessel.api.v1.stopContainerRequest
import dev.vessel.api.v1.streamLogsRequest
import dev.vessel.api.v1.streamStatsRequest
import dev.vessel.console.model.Container
import dev.vessel.console.model.ContainerSpec
import dev.vessel.console.model.ConnectionSettings
import dev.vessel.console.model.ExecResult
import dev.vessel.console.model.GpuJobRequest
import dev.vessel.console.model.GpuJobResult
import dev.vessel.console.model.HostInfo
import dev.vessel.console.model.LogChunk
import dev.vessel.console.model.StatsSample
import dev.vessel.console.model.SyscallSummary
import dev.vessel.console.model.TraceEvent
import dev.vessel.console.model.TraceFilter
import dev.vessel.console.model.VesselException
import dev.vessel.console.model.VesselRepository
import dev.vessel.console.model.flow.BackoffPolicy
import dev.vessel.console.model.flow.StreamEvent
import dev.vessel.console.model.flow.retryWithBackoff
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/**
 * gRPC-backed [VesselRepository].
 *
 * Responsibilities, and only these:
 *  1. obtain the right channel for the *current* settings ([ChannelManager]);
 *  2. apply per-call deadlines;
 *  3. map proto <-> domain ([Mappers.kt]) and gRPC errors -> [VesselException];
 *  4. turn server streams into resilient cold [Flow]s.
 *
 * ### Retry policy
 *  * Idempotent reads (GetHostInfo, List/GetContainer, GetSyscallSummary) get
 *    [BackoffPolicy.UnaryRead]: a couple of fast retries on transient errors.
 *  * Mutations (create/start/stop/…, Exec, RunGpuJob) are **never** retried
 *    automatically — a timed-out `StartContainer` may well have succeeded, and
 *    re-running an Exec or a GPU job is not harmless. The user decides.
 *  * Streams retry indefinitely with [BackoffPolicy.Streaming], emitting
 *    [StreamEvent.Reconnecting] so the UI can show it.
 *
 * ### Reconnect on settings change
 * Streams are built with `settings.flatMapLatest { … }`: editing host/port/
 * token/TLS cancels the running RPC and resubscribes against the new channel,
 * with no restart of the screen.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class VesselClient(
    private val settings: Flow<ConnectionSettings>,
    private val channels: ChannelManager,
    private val unaryDeadlineMs: Long = 10_000,
    private val readPolicy: BackoffPolicy = BackoffPolicy.UnaryRead,
    private val streamPolicy: BackoffPolicy = BackoffPolicy.Streaming,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
) : VesselRepository {

    // ------------------------------------------------------------------ unary

    override suspend fun hostInfo(): HostInfo =
        read { it.getHostInfo(getHostInfoRequest {}).toDomain() }

    override suspend fun listContainers(includeStopped: Boolean): List<Container> =
        read { stub -> stub.listContainers(listContainersRequest { all = includeStopped }).containersList.map { it.toDomain() } }

    override suspend fun getContainer(ref: String): Container =
        read { it.getContainer(containerRef { id = ref }).toDomain() }

    override suspend fun createContainer(spec: ContainerSpec, start: Boolean): Container =
        mutate { stub ->
            stub.createContainer(
                createContainerRequest {
                    this.spec = spec.toProto()
                    this.start = start
                },
            ).toDomain()
        }

    override suspend fun startContainer(ref: String): Container =
        mutate { it.startContainer(containerRef { id = ref }).toDomain() }

    override suspend fun stopContainer(ref: String, timeoutMs: Int): Container =
        // The daemon may legitimately take the whole grace period before SIGKILL.
        mutate(deadlineMs = unaryDeadlineMs + timeoutMs) {
            it.stopContainer(
                stopContainerRequest {
                    id = ref
                    this.timeoutMs = timeoutMs
                },
            ).toDomain()
        }

    override suspend fun pauseContainer(ref: String): Container =
        mutate { it.pauseContainer(containerRef { id = ref }).toDomain() }

    override suspend fun resumeContainer(ref: String): Container =
        mutate { it.resumeContainer(containerRef { id = ref }).toDomain() }

    override suspend fun removeContainer(ref: String, force: Boolean) {
        mutate(deadlineMs = unaryDeadlineMs + VesselRepository.DEFAULT_STOP_TIMEOUT_MS) {
            it.removeContainer(
                removeContainerRequest {
                    id = ref
                    this.force = force
                },
            )
        }
    }

    override suspend fun exec(ref: String, args: List<String>, env: List<String>, timeoutMs: Int): ExecResult {
        val serverTimeout = if (timeoutMs > 0) timeoutMs.toLong() else EXEC_DEFAULT_TIMEOUT_MS
        return mutate(deadlineMs = serverTimeout + unaryDeadlineMs) { stub ->
            stub.exec(
                execRequest {
                    id = ref
                    this.args += args
                    this.env += env
                    this.timeoutMs = timeoutMs
                },
            ).toDomain()
        }
    }

    override suspend fun syscallSummary(ref: String): SyscallSummary =
        read { it.getSyscallSummary(containerRef { id = ref }).toDomain() }

    override suspend fun runGpuJob(request: GpuJobRequest): GpuJobResult =
        mutate(deadlineMs = GPU_JOB_DEADLINE_MS) { it.runGpuJob(request.toProto()).toDomain() }

    // ---------------------------------------------------------------- streams

    override fun streamStats(ref: String, intervalMs: Int): Flow<StreamEvent<StatsSample>> =
        resilientStream { stub ->
            stub.streamStats(
                streamStatsRequest {
                    id = ref
                    this.intervalMs = intervalMs
                },
            ).map { it.toDomain() }
        }

    override fun streamTrace(filter: TraceFilter): Flow<StreamEvent<TraceEvent>> = flow {
        val request = filter.toProto()
        resilientStream { stub -> stub.streamTrace(request) }.collect { event ->
            emit(
                when (event) {
                    is StreamEvent.Item -> StreamEvent.Item(event.value.toDomain(traceSeq.getAndIncrement()))
                    is StreamEvent.Reconnecting -> event
                },
            )
        }
    }

    /**
     * Client-wide (not per-subscription) so keys stay unique when a consumer
     * keeps events from a previous subscription in its buffer — e.g. after a
     * filter change or returning to the trace tab.
     */
    private val traceSeq = AtomicLong()

    override fun streamLogs(ref: String, follow: Boolean, tailBytes: Int): Flow<StreamEvent<LogChunk>> =
        resilientStream { stub ->
            stub.streamLogs(
                streamLogsRequest {
                    id = ref
                    this.follow = follow
                    this.tailBytes = tailBytes
                },
            ).map { it.toDomain() }
        }

    // -------------------------------------------------------------- plumbing

    private suspend fun stub(deadlineMs: Long? = null): VesselCoroutineStub {
        val channel = channels.channelFor(settings.first())
        val stub = VesselCoroutineStub(channel)
        return if (deadlineMs != null) stub.withDeadlineAfter(deadlineMs, TimeUnit.MILLISECONDS) else stub
    }

    /** Idempotent read: deadline + bounded retry on transient failures. */
    private suspend fun <T> read(block: suspend (VesselCoroutineStub) -> T): T = withContext(ioDispatcher) {
        var attempt = 0
        var result: Result<T>
        while (true) {
            result = runCatchingNonCancellation { block(stub(unaryDeadlineMs)) }
            val failure = result.exceptionOrNull() ?: break
            val retryable = (failure as? VesselException)?.isRetryable == true
            if (!retryable || attempt >= readPolicy.maxAttempts) break
            delay(readPolicy.delayFor(attempt++))
        }
        result.getOrThrow()
    }

    /** Like runCatching, but maps errors to the domain and never swallows cancellation. */
    private inline fun <T> runCatchingNonCancellation(block: () -> T): Result<T> = try {
        Result.success(block())
    } catch (c: CancellationException) {
        throw c
    } catch (t: Throwable) {
        Result.failure(t.toVesselException())
    }

    /** Non-idempotent call: deadline, error mapping, no automatic retry. */
    private suspend fun <T> mutate(
        deadlineMs: Long = unaryDeadlineMs,
        block: suspend (VesselCoroutineStub) -> T,
    ): T = withContext(ioDispatcher) {
        try {
            block(stub(deadlineMs))
        } catch (t: Throwable) {
            throw t.toVesselException()
        }
    }

    /**
     * Server stream with: settings-driven resubscription, gRPC->domain error
     * mapping, and exponential-backoff reconnect. No deadline — telemetry
     * streams are open-ended; liveness is policed by HTTP/2 keepalive.
     */
    private fun <T> resilientStream(open: (VesselCoroutineStub) -> Flow<T>): Flow<StreamEvent<T>> =
        settings
            .distinctUntilChanged()
            .flatMapLatest { s ->
                flow { emitAll(open(VesselCoroutineStub(channels.channelFor(s)))) }
                    .catch { throw it.toVesselException() }
                    .retryWithBackoff(streamPolicy) { it.isRetryable() }
            }
            .flowOn(ioDispatcher)

    companion object {
        /** Mirrors the daemon's documented default for `ExecRequest.timeout_ms = 0`. */
        const val EXEC_DEFAULT_TIMEOUT_MS = 30_000L

        /** GPU jobs include container setup + warmup + N iterations + CPU verification. */
        const val GPU_JOB_DEADLINE_MS = 10 * 60_000L
    }
}
