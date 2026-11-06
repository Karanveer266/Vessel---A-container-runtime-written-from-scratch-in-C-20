@file:Suppress("TooManyFunctions")

package dev.vessel.console.api

import dev.vessel.api.v1.containerSpec
import dev.vessel.api.v1.gpuJobRequest
import dev.vessel.api.v1.mount
import dev.vessel.api.v1.resourceLimits
import dev.vessel.api.v1.streamTraceRequest
import dev.vessel.console.model.CgroupVersion
import dev.vessel.console.model.Container
import dev.vessel.console.model.ContainerSpec
import dev.vessel.console.model.ContainerState
import dev.vessel.console.model.ExecResult
import dev.vessel.console.model.GpuBackend
import dev.vessel.console.model.GpuDevice
import dev.vessel.console.model.GpuJobRequest
import dev.vessel.console.model.GpuJobResult
import dev.vessel.console.model.GpuOp
import dev.vessel.console.model.HostInfo
import dev.vessel.console.model.LogChunk
import dev.vessel.console.model.LogStream
import dev.vessel.console.model.Mount
import dev.vessel.console.model.NetworkMode
import dev.vessel.console.model.Psi
import dev.vessel.console.model.ResourceLimits
import dev.vessel.console.model.SeccompProfile
import dev.vessel.console.model.StatsSample
import dev.vessel.console.model.SyscallCount
import dev.vessel.console.model.SyscallSummary
import dev.vessel.console.model.TraceBackend
import dev.vessel.console.model.TraceEvent
import dev.vessel.console.model.TraceEventKind
import dev.vessel.console.model.TraceFilter
import dev.vessel.api.v1.CgroupVersion as PbCgroupVersion
import dev.vessel.api.v1.Container as PbContainer
import dev.vessel.api.v1.ContainerSpec as PbContainerSpec
import dev.vessel.api.v1.ContainerState as PbContainerState
import dev.vessel.api.v1.ExecResponse as PbExecResponse
import dev.vessel.api.v1.GpuBackend as PbGpuBackend
import dev.vessel.api.v1.GpuDevice as PbGpuDevice
import dev.vessel.api.v1.GpuJobRequest as PbGpuJobRequest
import dev.vessel.api.v1.GpuJobResult as PbGpuJobResult
import dev.vessel.api.v1.GpuOp as PbGpuOp
import dev.vessel.api.v1.HostInfo as PbHostInfo
import dev.vessel.api.v1.LogChunk as PbLogChunk
import dev.vessel.api.v1.NetworkMode as PbNetworkMode
import dev.vessel.api.v1.Psi as PbPsi
import dev.vessel.api.v1.ResourceLimits as PbResourceLimits
import dev.vessel.api.v1.SeccompProfile as PbSeccompProfile
import dev.vessel.api.v1.StatsSample as PbStatsSample
import dev.vessel.api.v1.StreamTraceRequest as PbStreamTraceRequest
import dev.vessel.api.v1.SyscallSummary as PbSyscallSummary
import dev.vessel.api.v1.TraceBackend as PbTraceBackend
import dev.vessel.api.v1.TraceEvent as PbTraceEvent
import dev.vessel.api.v1.TraceEventKind as PbTraceEventKind

/*
 * Proto <-> domain mapping.
 *
 * Kept as pure top-level functions (no client state) so they are trivially
 * unit-testable and the generated classes never leak past this module.
 *
 * Enum policy: every `when` over a proto enum is exhaustive *including*
 * UNRECOGNIZED, which (like *_UNSPECIFIED) maps to the domain `Unspecified`.
 * Adding a value to the proto therefore breaks this file's compilation
 * after regeneration — a deliberate tripwire — but never crashes an
 * already-shipped app talking to a newer daemon.
 *
 * Unsigned proto fields (uint32/uint64) arrive as signed Java ints/longs. All
 * realistic values (bytes, ns, counts) fit in the signed range, so they're
 * used as-is rather than paying for kotlin.ULong throughout the UI.
 */

// ---------------------------------------------------------------- enums: proto -> domain

internal fun PbCgroupVersion.toDomain(): CgroupVersion = when (this) {
    PbCgroupVersion.CGROUP_VERSION_V1 -> CgroupVersion.V1
    PbCgroupVersion.CGROUP_VERSION_V2 -> CgroupVersion.V2
    PbCgroupVersion.CGROUP_VERSION_HYBRID -> CgroupVersion.Hybrid
    PbCgroupVersion.CGROUP_VERSION_UNSPECIFIED, PbCgroupVersion.UNRECOGNIZED -> CgroupVersion.Unspecified
}

internal fun PbTraceBackend.toDomain(): TraceBackend = when (this) {
    PbTraceBackend.TRACE_BACKEND_NONE -> TraceBackend.None
    PbTraceBackend.TRACE_BACKEND_EBPF -> TraceBackend.Ebpf
    PbTraceBackend.TRACE_BACKEND_PTRACE -> TraceBackend.Ptrace
    PbTraceBackend.TRACE_BACKEND_AUTO -> TraceBackend.Auto
    PbTraceBackend.TRACE_BACKEND_UNSPECIFIED, PbTraceBackend.UNRECOGNIZED -> TraceBackend.Unspecified
}

internal fun PbContainerState.toDomain(): ContainerState = when (this) {
    PbContainerState.CONTAINER_STATE_CREATED -> ContainerState.Created
    PbContainerState.CONTAINER_STATE_RUNNING -> ContainerState.Running
    PbContainerState.CONTAINER_STATE_PAUSED -> ContainerState.Paused
    PbContainerState.CONTAINER_STATE_STOPPED -> ContainerState.Stopped
    PbContainerState.CONTAINER_STATE_FAILED -> ContainerState.Failed
    PbContainerState.CONTAINER_STATE_UNSPECIFIED, PbContainerState.UNRECOGNIZED -> ContainerState.Unspecified
}

internal fun PbNetworkMode.toDomain(): NetworkMode = when (this) {
    PbNetworkMode.NETWORK_MODE_NONE -> NetworkMode.None
    PbNetworkMode.NETWORK_MODE_LOOPBACK -> NetworkMode.Loopback
    PbNetworkMode.NETWORK_MODE_HOST -> NetworkMode.Host
    PbNetworkMode.NETWORK_MODE_UNSPECIFIED, PbNetworkMode.UNRECOGNIZED -> NetworkMode.Unspecified
}

internal fun PbSeccompProfile.toDomain(): SeccompProfile = when (this) {
    PbSeccompProfile.SECCOMP_PROFILE_DEFAULT -> SeccompProfile.Default
    PbSeccompProfile.SECCOMP_PROFILE_STRICT -> SeccompProfile.Strict
    PbSeccompProfile.SECCOMP_PROFILE_UNCONFINED -> SeccompProfile.Unconfined
    PbSeccompProfile.SECCOMP_PROFILE_UNSPECIFIED, PbSeccompProfile.UNRECOGNIZED -> SeccompProfile.Unspecified
}

internal fun PbTraceEventKind.toDomain(): TraceEventKind = when (this) {
    PbTraceEventKind.TRACE_EVENT_KIND_SYSCALL -> TraceEventKind.Syscall
    PbTraceEventKind.TRACE_EVENT_KIND_MEMORY -> TraceEventKind.Memory
    PbTraceEventKind.TRACE_EVENT_KIND_NET -> TraceEventKind.Net
    PbTraceEventKind.TRACE_EVENT_KIND_PROCESS_EXEC -> TraceEventKind.ProcessExec
    PbTraceEventKind.TRACE_EVENT_KIND_PROCESS_EXIT -> TraceEventKind.ProcessExit
    PbTraceEventKind.TRACE_EVENT_KIND_OOM_KILL -> TraceEventKind.OomKill
    PbTraceEventKind.TRACE_EVENT_KIND_MEMORY_PRESSURE -> TraceEventKind.MemoryPressure
    PbTraceEventKind.TRACE_EVENT_KIND_SECCOMP_DENIED -> TraceEventKind.SeccompDenied
    PbTraceEventKind.TRACE_EVENT_KIND_LIFECYCLE -> TraceEventKind.Lifecycle
    PbTraceEventKind.TRACE_EVENT_KIND_UNSPECIFIED, PbTraceEventKind.UNRECOGNIZED -> TraceEventKind.Unspecified
}

internal fun PbGpuBackend.toDomain(): GpuBackend = when (this) {
    PbGpuBackend.GPU_BACKEND_CUDA -> GpuBackend.Cuda
    PbGpuBackend.GPU_BACKEND_SIMULATED -> GpuBackend.Simulated
    PbGpuBackend.GPU_BACKEND_AUTO -> GpuBackend.Auto
    PbGpuBackend.GPU_BACKEND_UNSPECIFIED, PbGpuBackend.UNRECOGNIZED -> GpuBackend.Unspecified
}

internal fun PbLogChunk.Stream.toDomain(): LogStream = when (this) {
    PbLogChunk.Stream.STREAM_STDOUT -> LogStream.Stdout
    PbLogChunk.Stream.STREAM_STDERR -> LogStream.Stderr
    PbLogChunk.Stream.STREAM_UNSPECIFIED, PbLogChunk.Stream.UNRECOGNIZED -> LogStream.Unspecified
}

// ---------------------------------------------------------------- enums: domain -> proto

internal fun TraceBackend.toProto(): PbTraceBackend = when (this) {
    TraceBackend.Unspecified -> PbTraceBackend.TRACE_BACKEND_UNSPECIFIED
    TraceBackend.None -> PbTraceBackend.TRACE_BACKEND_NONE
    TraceBackend.Ebpf -> PbTraceBackend.TRACE_BACKEND_EBPF
    TraceBackend.Ptrace -> PbTraceBackend.TRACE_BACKEND_PTRACE
    TraceBackend.Auto -> PbTraceBackend.TRACE_BACKEND_AUTO
}

internal fun NetworkMode.toProto(): PbNetworkMode = when (this) {
    NetworkMode.Unspecified -> PbNetworkMode.NETWORK_MODE_UNSPECIFIED
    NetworkMode.None -> PbNetworkMode.NETWORK_MODE_NONE
    NetworkMode.Loopback -> PbNetworkMode.NETWORK_MODE_LOOPBACK
    NetworkMode.Host -> PbNetworkMode.NETWORK_MODE_HOST
}

internal fun SeccompProfile.toProto(): PbSeccompProfile = when (this) {
    SeccompProfile.Unspecified -> PbSeccompProfile.SECCOMP_PROFILE_UNSPECIFIED
    SeccompProfile.Default -> PbSeccompProfile.SECCOMP_PROFILE_DEFAULT
    SeccompProfile.Strict -> PbSeccompProfile.SECCOMP_PROFILE_STRICT
    SeccompProfile.Unconfined -> PbSeccompProfile.SECCOMP_PROFILE_UNCONFINED
}

internal fun TraceEventKind.toProto(): PbTraceEventKind = when (this) {
    TraceEventKind.Unspecified -> PbTraceEventKind.TRACE_EVENT_KIND_UNSPECIFIED
    TraceEventKind.Syscall -> PbTraceEventKind.TRACE_EVENT_KIND_SYSCALL
    TraceEventKind.Memory -> PbTraceEventKind.TRACE_EVENT_KIND_MEMORY
    TraceEventKind.Net -> PbTraceEventKind.TRACE_EVENT_KIND_NET
    TraceEventKind.ProcessExec -> PbTraceEventKind.TRACE_EVENT_KIND_PROCESS_EXEC
    TraceEventKind.ProcessExit -> PbTraceEventKind.TRACE_EVENT_KIND_PROCESS_EXIT
    TraceEventKind.OomKill -> PbTraceEventKind.TRACE_EVENT_KIND_OOM_KILL
    TraceEventKind.MemoryPressure -> PbTraceEventKind.TRACE_EVENT_KIND_MEMORY_PRESSURE
    TraceEventKind.SeccompDenied -> PbTraceEventKind.TRACE_EVENT_KIND_SECCOMP_DENIED
    TraceEventKind.Lifecycle -> PbTraceEventKind.TRACE_EVENT_KIND_LIFECYCLE
}

internal fun GpuOp.toProto(): PbGpuOp = when (this) {
    GpuOp.Unspecified -> PbGpuOp.GPU_OP_UNSPECIFIED
    GpuOp.Gemm -> PbGpuOp.GPU_OP_GEMM
    GpuOp.GemmSplitKAllReduce -> PbGpuOp.GPU_OP_GEMM_SPLITK_ALLREDUCE
    GpuOp.FlashAttention -> PbGpuOp.GPU_OP_FLASH_ATTENTION
    GpuOp.AllReduceBench -> PbGpuOp.GPU_OP_ALLREDUCE_BENCH
}

internal fun GpuBackend.toProto(): PbGpuBackend = when (this) {
    GpuBackend.Unspecified -> PbGpuBackend.GPU_BACKEND_UNSPECIFIED
    GpuBackend.Cuda -> PbGpuBackend.GPU_BACKEND_CUDA
    GpuBackend.Simulated -> PbGpuBackend.GPU_BACKEND_SIMULATED
    GpuBackend.Auto -> PbGpuBackend.GPU_BACKEND_AUTO
}

// ---------------------------------------------------------------- messages: proto -> domain

internal fun PbHostInfo.toDomain() = HostInfo(
    kernelRelease = kernelRelease,
    machine = machine,
    isAndroid = isAndroid,
    cgroupVersion = cgroupVersion.toDomain(),
    cgroupControllers = cgroupControllersList.toList(),
    ebpfAvailable = ebpfAvailable,
    rootless = rootless,
    psiAvailable = psiAvailable,
    gpus = gpusList.map { it.toDomain() },
    daemonVersion = daemonVersion,
    onlineCpus = onlineCpus,
    totalMemoryBytes = totalMemoryBytes,
    defaultTraceBackend = defaultTraceBackend.toDomain(),
)

internal fun PbGpuDevice.toDomain() = GpuDevice(
    index = index,
    name = name,
    totalMemoryBytes = totalMemoryBytes,
    smMajor = smMajor,
    smMinor = smMinor,
)

internal fun PbResourceLimits.toDomain() = ResourceLimits(
    cpuMillicores = cpuMillicores,
    cpuWeight = cpuWeight,
    memoryMaxBytes = memoryMaxBytes,
    memoryHighBytes = memoryHighBytes,
    pidsMax = pidsMax,
    cpusetCpus = cpusetCpus,
    disableSwap = disableSwap,
)

internal fun PbContainerSpec.toDomain() = ContainerSpec(
    name = name,
    rootfs = rootfs,
    args = argsList.toList(),
    env = envList.toList(),
    hostname = hostname,
    workingDir = workingDir,
    limits = limits.toDomain(),
    network = network.toDomain(),
    seccomp = seccomp.toDomain(),
    trace = trace.toDomain(),
    mounts = mountsList.map { Mount(it.source, it.target, it.readOnly) },
    readOnlyRootfs = readOnlyRootfs,
    gpuDevices = gpuDevicesList.toList(),
    autoRemove = autoRemove,
)

internal fun PbContainer.toDomain() = Container(
    id = id,
    spec = spec.toDomain(),
    state = state.toDomain(),
    hostPid = hostPid,
    exitCode = exitCode,
    oomKilled = oomKilled,
    createdAtUnixNs = createdAtUnixNs,
    startedAtUnixNs = startedAtUnixNs,
    finishedAtUnixNs = finishedAtUnixNs,
    cgroupPath = cgroupPath,
    activeTraceBackend = activeTraceBackend.toDomain(),
    error = error,
)

internal fun PbExecResponse.toDomain() = ExecResult(
    exitCode = exitCode,
    // Process output is not guaranteed UTF-8; String(bytes) replaces invalid
    // sequences with U+FFFD instead of throwing.
    stdout = stdout.toStringUtf8(),
    stderr = stderr.toStringUtf8(),
    timedOut = timedOut,
)

internal fun PbPsi.toDomain() = Psi(
    someAvg10 = someAvg10,
    someAvg60 = someAvg60,
    fullAvg10 = fullAvg10,
    someTotalUsec = someTotalUsec,
)

internal fun PbStatsSample.toDomain() = StatsSample(
    containerId = containerId,
    timestampUnixNs = timestampUnixNs,
    cpuUsageUsec = cpuUsageUsec,
    cpuNrThrottled = cpuNrThrottled,
    cpuThrottledUsec = cpuThrottledUsec,
    cpuMillicoresUsed = cpuMillicoresUsed,
    memoryCurrentBytes = memoryCurrentBytes,
    memoryMaxBytes = memoryMaxBytes,
    memoryPeakBytes = memoryPeakBytes,
    memoryEventsHigh = memoryEventsHigh,
    memoryEventsMax = memoryEventsMax,
    memoryEventsOom = memoryEventsOom,
    memoryEventsOomKill = memoryEventsOomKill,
    pidsCurrent = pidsCurrent,
    pidsMax = pidsMax,
    // Absent sub-messages (no PSI on this kernel) read as default instances = zeros.
    cpuPressure = if (hasCpuPressure()) cpuPressure.toDomain() else Psi.Zero,
    memoryPressure = if (hasMemoryPressure()) memoryPressure.toDomain() else Psi.Zero,
    ioPressure = if (hasIoPressure()) ioPressure.toDomain() else Psi.Zero,
    netRxBytes = netRxBytes,
    netTxBytes = netTxBytes,
    droppedSamples = droppedSamples,
)

/** [seq] is assigned by the client per subscription; see [TraceEvent.seq]. */
internal fun PbTraceEvent.toDomain(seq: Long) = TraceEvent(
    seq = seq,
    containerId = containerId,
    kind = kind.toDomain(),
    timestampUnixNs = timestampUnixNs,
    hostPid = hostPid,
    hostTid = hostTid,
    comm = comm,
    syscallNr = syscallNr,
    syscallName = syscallName,
    ret = ret,
    latencyNs = latencyNs,
    bytes = bytes,
    egress = egress,
    detail = detail,
    droppedEvents = droppedEvents,
    backend = backend.toDomain(),
)

internal fun PbSyscallSummary.toDomain() = SyscallSummary(
    containerId = containerId,
    syscalls = syscallsList
        .map { SyscallCount(it.name, it.nr, it.count, it.errors, it.totalLatencyNs, it.maxLatencyNs) }
        .sortedByDescending { it.count },
    totalEvents = totalEvents,
    droppedEvents = droppedEvents,
    backend = backend.toDomain(),
)

internal fun PbLogChunk.toDomain() = LogChunk(
    stream = stream.toDomain(),
    data = data.toByteArray(),
    timestampUnixNs = timestampUnixNs,
)

internal fun PbGpuJobResult.toDomain() = GpuJobResult(
    ok = ok,
    error = error,
    backend = backend.toDomain(),
    timeMs = timeMs,
    gflops = gflops,
    busBandwidthGbps = busBandwidthGbps,
    maxAbsError = maxAbsError,
    containerId = containerId,
    deviceNames = deviceNames,
    // Sorted for stable rendering; proto maps have no defined order.
    metrics = metricsMap.toSortedMap(),
)

// ---------------------------------------------------------------- messages: domain -> proto

internal fun ResourceLimits.toProto(): PbResourceLimits = resourceLimits {
    cpuMillicores = this@toProto.cpuMillicores
    cpuWeight = this@toProto.cpuWeight
    memoryMaxBytes = this@toProto.memoryMaxBytes
    memoryHighBytes = this@toProto.memoryHighBytes
    pidsMax = this@toProto.pidsMax
    cpusetCpus = this@toProto.cpusetCpus
    disableSwap = this@toProto.disableSwap
}

internal fun ContainerSpec.toProto(): PbContainerSpec {
    val src = this
    return containerSpec {
        name = src.name
        rootfs = src.rootfs
        args += src.args
        env += src.env
        hostname = src.hostname
        workingDir = src.workingDir
        limits = src.limits.toProto()
        network = src.network.toProto()
        seccomp = src.seccomp.toProto()
        trace = src.trace.toProto()
        mounts += src.mounts.map { m ->
            mount {
                source = m.source
                target = m.target
                readOnly = m.readOnly
            }
        }
        readOnlyRootfs = src.readOnlyRootfs
        gpuDevices += src.gpuDevices
        autoRemove = src.autoRemove
    }
}

internal fun TraceFilter.toProto(): PbStreamTraceRequest {
    val src = this
    return streamTraceRequest {
        id = src.containerId
        kinds += src.kinds.map { it.toProto() }
        syscalls += src.syscalls
        minLatencyNs = src.minLatencyNs
        errorsOnly = src.errorsOnly
    }
}

internal fun GpuJobRequest.toProto(): PbGpuJobRequest {
    val src = this
    return gpuJobRequest {
        op = src.op.toProto()
        backend = src.backend.toProto()
        m = src.m
        n = src.n
        k = src.k
        ranks = src.ranks
        batchHeads = src.batchHeads
        seqLen = src.seqLen
        headDim = src.headDim
        causal = src.causal
        iterations = src.iterations
        verify = src.verify
        limits = src.limits.toProto()
        rootfs = src.rootfs
    }
}
