package dev.vessel.console.model

enum class ContainerState(val label: String) {
    Unspecified("unknown"),
    Created("created"),
    Running("running"),
    Paused("paused"),
    Stopped("stopped"),
    Failed("failed");

    /** Which lifecycle actions make sense from this state (drives the UI menus). */
    val allowedActions: Set<ContainerAction>
        get() = when (this) {
            Created -> setOf(ContainerAction.Start, ContainerAction.Remove)
            Running -> setOf(ContainerAction.Stop, ContainerAction.Pause, ContainerAction.Remove)
            Paused -> setOf(ContainerAction.Resume, ContainerAction.Stop, ContainerAction.Remove)
            Stopped, Failed -> setOf(ContainerAction.Start, ContainerAction.Remove)
            Unspecified -> setOf(ContainerAction.Remove)
        }
}

enum class ContainerAction(val label: String) {
    Start("Start"),
    Stop("Stop"),
    Pause("Pause"),
    Resume("Resume"),
    Remove("Remove"),
}

enum class NetworkMode(val label: String) {
    Unspecified("daemon default"),
    None("none"),
    Loopback("loopback"),
    Host("host"),
}

enum class SeccompProfile(val label: String) {
    Unspecified("daemon default"),
    Default("default"),
    Strict("strict"),
    Unconfined("unconfined"),
}

/** All numeric limits use 0 as "unlimited / daemon default", exactly like the proto. */
data class ResourceLimits(
    val cpuMillicores: Int = 0,
    val cpuWeight: Int = 0,
    val memoryMaxBytes: Long = 0,
    val memoryHighBytes: Long = 0,
    val pidsMax: Int = 0,
    val cpusetCpus: String = "",
    val disableSwap: Boolean = false,
)

/** Bind mount of a host path into the container. */
data class Mount(
    val source: String,
    val target: String,
    val readOnly: Boolean = false,
)

data class ContainerSpec(
    val name: String,
    val rootfs: String,
    val args: List<String>,
    val env: List<String> = emptyList(),
    val hostname: String = "",
    val workingDir: String = "",
    val limits: ResourceLimits = ResourceLimits(),
    val network: NetworkMode = NetworkMode.Unspecified,
    val seccomp: SeccompProfile = SeccompProfile.Unspecified,
    val trace: TraceBackend = TraceBackend.Unspecified,
    val mounts: List<Mount> = emptyList(),
    val readOnlyRootfs: Boolean = false,
    val gpuDevices: List<Int> = emptyList(),
    val autoRemove: Boolean = false,
)

data class Container(
    val id: String,
    val spec: ContainerSpec,
    val state: ContainerState,
    val hostPid: Int,
    val exitCode: Int,
    val oomKilled: Boolean,
    val createdAtUnixNs: Long,
    val startedAtUnixNs: Long,
    val finishedAtUnixNs: Long,
    val cgroupPath: String,
    val activeTraceBackend: TraceBackend,
    val error: String,
) {
    /** Human name if set, otherwise the short id (the daemon accepts either as a ref). */
    val displayName: String get() = spec.name.ifBlank { id }
}

data class ExecResult(
    val exitCode: Int,
    val stdout: String,
    val stderr: String,
    val timedOut: Boolean,
)
