package dev.vessel.console.testing

import dev.vessel.console.model.CgroupVersion
import dev.vessel.console.model.Container
import dev.vessel.console.model.ContainerSpec
import dev.vessel.console.model.ContainerState
import dev.vessel.console.model.ExecResult
import dev.vessel.console.model.GpuBackend
import dev.vessel.console.model.GpuJobRequest
import dev.vessel.console.model.GpuJobResult
import dev.vessel.console.model.HostInfo
import dev.vessel.console.model.LogChunk
import dev.vessel.console.model.LogStream
import dev.vessel.console.model.Psi
import dev.vessel.console.model.StatsSample
import dev.vessel.console.model.SyscallSummary
import dev.vessel.console.model.TraceBackend
import dev.vessel.console.model.TraceEvent
import dev.vessel.console.model.TraceEventKind
import dev.vessel.console.model.TraceFilter
import dev.vessel.console.model.VesselException
import dev.vessel.console.model.VesselRepository
import dev.vessel.console.model.flow.StreamEvent
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.onStart
import kotlinx.coroutines.flow.takeWhile

/**
 * Scriptable in-memory [VesselRepository].
 *
 * Streams are hot [MutableSharedFlow]s the test pushes into; subscriber
 * counts and requested filters are recorded so tests can assert lifecycle
 * behaviour (e.g. "the RPC was cancelled when the UI unsubscribed").
 */
class FakeVesselRepository : VesselRepository {
    var host: HostInfo = dev.vessel.console.testing.hostInfo()
    var hostError: Throwable? = null
    var hostCalls = 0

    val containers = mutableListOf<Container>()
    var listError: Throwable? = null
    var listCalls = 0
    val actions = mutableListOf<Pair<String, String>>()
    var actionError: Throwable? = null

    var created: Pair<ContainerSpec, Boolean>? = null
    var execResult = ExecResult(0, "ok\n", "", false)
    var execArgs: List<String>? = null
    var gpuResult: GpuJobResult = gpuResult()
    var gpuRequest: GpuJobRequest? = null
    var summary = SyscallSummary("c1", emptyList(), 0, 0, TraceBackend.Ebpf)
    var summaryCalls = 0

    // ---- streams ------------------------------------------------------------

    /** Marker to complete a stream from the test. */
    private object End

    val stats = MutableSharedFlow<Any>(extraBufferCapacity = 1_024)
    val trace = MutableSharedFlow<Any>(extraBufferCapacity = 100_000)
    val logs = MutableSharedFlow<Any>(extraBufferCapacity = 1_024)

    var statsSubscribers = 0; private set
    var traceSubscribers = 0; private set
    val traceFilters = mutableListOf<TraceFilter>()
    val statsRefs = mutableListOf<String>()

    suspend fun endLogs() = logs.emit(End)

    @Suppress("UNCHECKED_CAST")
    private fun <T> stream(source: MutableSharedFlow<Any>, onSub: () -> Unit, onUnsub: () -> Unit): Flow<StreamEvent<T>> =
        source
            .onStart { onSub() }
            .onCompletion { onUnsub() }
            .takeWhile { it !== End }
            .let { flow -> kotlinx.coroutines.flow.flow { flow.collect { emit(it as StreamEvent<T>) } } }

    override fun streamStats(ref: String, intervalMs: Int): Flow<StreamEvent<StatsSample>> {
        statsRefs += ref
        return stream(stats, { statsSubscribers++ }, { statsSubscribers-- })
    }

    override fun streamTrace(filter: TraceFilter): Flow<StreamEvent<TraceEvent>> {
        traceFilters += filter
        return stream(trace, { traceSubscribers++ }, { traceSubscribers-- })
    }

    override fun streamLogs(ref: String, follow: Boolean, tailBytes: Int): Flow<StreamEvent<LogChunk>> =
        stream(logs, {}, {})

    // ---- unary ----------------------------------------------------------------

    override suspend fun hostInfo(): HostInfo {
        hostCalls++
        hostError?.let { throw it }
        return host
    }

    override suspend fun listContainers(includeStopped: Boolean): List<Container> {
        listCalls++
        listError?.let { throw it }
        return containers.filter { includeStopped || it.state == ContainerState.Running }.toList()
    }

    override suspend fun getContainer(ref: String): Container =
        containers.firstOrNull { it.id == ref } ?: throw VesselException.NotFound(ref)

    override suspend fun createContainer(spec: ContainerSpec, start: Boolean): Container {
        created = spec to start
        return container("new123", spec.name, if (start) ContainerState.Running else ContainerState.Created).also { containers += it }
    }

    private fun act(name: String, ref: String, newState: ContainerState): Container {
        actions += name to ref
        actionError?.let { throw it }
        val i = containers.indexOfFirst { it.id == ref }
        val updated = containers[i].copy(state = newState)
        containers[i] = updated
        return updated
    }

    override suspend fun startContainer(ref: String) = act("start", ref, ContainerState.Running)
    override suspend fun stopContainer(ref: String, timeoutMs: Int) = act("stop", ref, ContainerState.Stopped)
    override suspend fun pauseContainer(ref: String) = act("pause", ref, ContainerState.Paused)
    override suspend fun resumeContainer(ref: String) = act("resume", ref, ContainerState.Running)
    override suspend fun removeContainer(ref: String, force: Boolean) {
        actions += "remove(force=$force)" to ref
        actionError?.let { throw it }
        containers.removeAll { it.id == ref }
    }

    override suspend fun exec(ref: String, args: List<String>, env: List<String>, timeoutMs: Int): ExecResult {
        execArgs = args
        return execResult
    }

    override suspend fun syscallSummary(ref: String): SyscallSummary {
        summaryCalls++
        return summary
    }

    override suspend fun runGpuJob(request: GpuJobRequest): GpuJobResult {
        gpuRequest = request
        return gpuResult
    }
}

// ---- builders -------------------------------------------------------------------

fun hostInfo() = HostInfo(
    kernelRelease = "6.6.0", machine = "aarch64", isAndroid = true, cgroupVersion = CgroupVersion.V2,
    cgroupControllers = listOf("cpu", "memory"), ebpfAvailable = true, rootless = false, psiAvailable = true,
    gpus = emptyList(), daemonVersion = "0.1", onlineCpus = 8, totalMemoryBytes = 8L shl 30,
    defaultTraceBackend = TraceBackend.Ebpf,
)

fun container(id: String, name: String = id, state: ContainerState = ContainerState.Running, createdAt: Long = 0) = Container(
    id = id, spec = ContainerSpec(name, "/rootfs", listOf("/bin/sh")), state = state, hostPid = 100, exitCode = 0,
    oomKilled = false, createdAtUnixNs = createdAt, startedAtUnixNs = 0, finishedAtUnixNs = 0, cgroupPath = "",
    activeTraceBackend = TraceBackend.Ebpf, error = "",
)

fun stats(id: String = "c1", cpu: Int = 0, mem: Long = 0, oomKill: Long = 0) = StatsSample(
    containerId = id, timestampUnixNs = 0, cpuUsageUsec = 0, cpuNrThrottled = 0, cpuThrottledUsec = 0,
    cpuMillicoresUsed = cpu, memoryCurrentBytes = mem, memoryMaxBytes = 0, memoryPeakBytes = 0,
    memoryEventsHigh = 0, memoryEventsMax = 0, memoryEventsOom = 0, memoryEventsOomKill = oomKill,
    pidsCurrent = 1, pidsMax = 0, cpuPressure = Psi.Zero, memoryPressure = Psi.Zero, ioPressure = Psi.Zero,
    netRxBytes = 0, netTxBytes = 0, droppedSamples = 0,
)

fun traceEvent(seq: Long, name: String = "read", ret: Long = 0, latencyNs: Long = 1_000, dropped: Long = 0) = TraceEvent(
    seq = seq, containerId = "c1", kind = TraceEventKind.Syscall, timestampUnixNs = seq, hostPid = 1, hostTid = 1,
    comm = "app", syscallNr = 0, syscallName = name, ret = ret, latencyNs = latencyNs, bytes = 0, egress = false,
    detail = "", droppedEvents = dropped, backend = TraceBackend.Ebpf,
)

fun logChunk(text: String, stream: LogStream = LogStream.Stdout) = LogChunk(stream, text.toByteArray(), 0)

fun gpuResult() = GpuJobResult(
    ok = true, error = "", backend = GpuBackend.Simulated, timeMs = 1.5, gflops = 900.0, busBandwidthGbps = 0.0,
    maxAbsError = 1e-6, containerId = "gpu1", deviceNames = "sim", metrics = mapOf("k" to "v"),
)

fun <T> item(value: T) = StreamEvent.Item(value)
