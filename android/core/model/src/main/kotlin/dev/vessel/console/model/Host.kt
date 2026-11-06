package dev.vessel.console.model

/**
 * Domain mirror of `vessel.v1.HostInfo`.
 *
 * Every enum carries an explicit `Unspecified` member: the proto contract
 * guarantees `*_UNSPECIFIED = 0`, and unknown future values (protobuf's
 * `UNRECOGNIZED`) are folded into it so an old console degrades to "unknown"
 * instead of crashing on a newer daemon.
 */
data class HostInfo(
    val kernelRelease: String,
    val machine: String,
    val isAndroid: Boolean,
    val cgroupVersion: CgroupVersion,
    val cgroupControllers: List<String>,
    val ebpfAvailable: Boolean,
    val rootless: Boolean,
    val psiAvailable: Boolean,
    val gpus: List<GpuDevice>,
    val daemonVersion: String,
    val onlineCpus: Int,
    val totalMemoryBytes: Long,
    val defaultTraceBackend: TraceBackend,
)

enum class CgroupVersion(val label: String) {
    Unspecified("unknown"),
    V1("v1 (legacy)"),
    V2("v2 (unified)"),
    Hybrid("hybrid"),
}

enum class TraceBackend(val label: String) {
    Unspecified("default"),
    None("none"),
    Ebpf("eBPF"),
    Ptrace("ptrace"),
    Auto("auto"),
}

data class GpuDevice(
    val index: Int,
    val name: String,
    val totalMemoryBytes: Long,
    val smMajor: Int,
    val smMinor: Int,
) {
    val computeCapability: String get() = "sm_$smMajor$smMinor"
}
