package dev.vessel.console.model

import dev.vessel.console.model.flow.StreamEvent
import kotlinx.coroutines.flow.Flow

/**
 * The domain-facing API of `vesseld`.
 *
 * Lives in the platform-free model module so that ViewModels depend on this
 * interface only; the gRPC implementation (`VesselClient` in `:core:api`) is
 * swapped for an in-memory fake in tests.
 *
 * Contract:
 *  * suspend functions throw [VesselException] on failure (never raw gRPC types);
 *  * idempotent reads may be retried internally, mutations never are;
 *  * stream functions return *cold* flows: the RPC starts on collection and is
 *    cancelled with the collector. They reconnect with backoff on transient
 *    failures and report it in-band as [StreamEvent.Reconnecting].
 */
interface VesselRepository {
    suspend fun hostInfo(): HostInfo

    suspend fun listContainers(includeStopped: Boolean = true): List<Container>
    suspend fun getContainer(ref: String): Container
    suspend fun createContainer(spec: ContainerSpec, start: Boolean): Container
    suspend fun startContainer(ref: String): Container
    suspend fun stopContainer(ref: String, timeoutMs: Int = DEFAULT_STOP_TIMEOUT_MS): Container
    suspend fun pauseContainer(ref: String): Container
    suspend fun resumeContainer(ref: String): Container
    suspend fun removeContainer(ref: String, force: Boolean)

    suspend fun exec(ref: String, args: List<String>, env: List<String> = emptyList(), timeoutMs: Int = 0): ExecResult

    /** [ref] empty = every running container (one multiplexed stream, used by the list screen). */
    fun streamStats(ref: String, intervalMs: Int = 1_000): Flow<StreamEvent<StatsSample>>
    /** Events carry a [TraceEvent.seq] unique across all subscriptions of this instance. */
    fun streamTrace(filter: TraceFilter): Flow<StreamEvent<TraceEvent>>
    fun streamLogs(ref: String, follow: Boolean = true, tailBytes: Int = DEFAULT_LOG_TAIL_BYTES): Flow<StreamEvent<LogChunk>>

    suspend fun syscallSummary(ref: String): SyscallSummary
    suspend fun runGpuJob(request: GpuJobRequest): GpuJobResult

    companion object {
        const val DEFAULT_STOP_TIMEOUT_MS = 10_000
        const val DEFAULT_LOG_TAIL_BYTES = 64 * 1024
    }
}

/**
 * Dispatches a lifecycle [action]. Returns the updated container, or null for
 * [ContainerAction.Remove]. Running/paused containers are force-removed (the
 * UI asks for confirmation first).
 */
suspend fun VesselRepository.perform(container: Container, action: ContainerAction): Container? = when (action) {
    ContainerAction.Start -> startContainer(container.id)
    ContainerAction.Stop -> stopContainer(container.id)
    ContainerAction.Pause -> pauseContainer(container.id)
    ContainerAction.Resume -> resumeContainer(container.id)
    ContainerAction.Remove -> {
        val force = container.state == ContainerState.Running || container.state == ContainerState.Paused
        removeContainer(container.id, force)
        null
    }
}
