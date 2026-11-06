package dev.vessel.console.ui.create

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.vessel.console.model.Container
import dev.vessel.console.model.ContainerSpec
import dev.vessel.console.model.Mount
import dev.vessel.console.model.NetworkMode
import dev.vessel.console.model.ResourceLimits
import dev.vessel.console.model.SeccompProfile
import dev.vessel.console.model.TraceBackend
import dev.vessel.console.model.VesselRepository
import dev.vessel.console.model.format.CommandLine
import dev.vessel.console.model.format.Quantities
import dev.vessel.console.model.userMessage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** Raw form input. Everything is a String until validated, so the user can type freely. */
data class CreateForm(
    val name: String = "",
    val rootfs: String = "",
    val command: String = "/bin/sh -c 'while true; do date; sleep 1; done'",
    /** One KEY=VALUE per line. */
    val env: String = "",
    val hostname: String = "",
    val network: NetworkMode = NetworkMode.Loopback,
    val seccomp: SeccompProfile = SeccompProfile.Default,
    val trace: TraceBackend = TraceBackend.Auto,
    val cpu: String = "",
    val memoryMax: String = "",
    val memoryHigh: String = "",
    val pidsMax: String = "",
    val cpuset: String = "",
    val autoRemove: Boolean = false,
    val startImmediately: Boolean = true,
    // ---- advanced ----
    val workingDir: String = "",
    val readOnlyRootfs: Boolean = false,
    /** Relative CPU weight 1..10000; empty = daemon default. */
    val cpuWeight: String = "",
    val disableSwap: Boolean = false,
    val mounts: List<MountRow> = emptyList(),
)

/**
 * One editable bind-mount row. [id] is assigned by the ViewModel and never
 * reused, so Compose keys and per-row errors stay attached to the right row
 * when rows above it are removed.
 */
data class MountRow(
    val id: Long,
    val source: String = "",
    val target: String = "",
    val readOnly: Boolean = false,
) {
    val isBlank: Boolean get() = source.isBlank() && target.isBlank()
}

/** Per-row errors; null = field is fine. */
data class MountErrors(val source: String? = null, val target: String? = null)

enum class Field { Name, Rootfs, Command, Env, Hostname, Cpu, MemoryMax, MemoryHigh, PidsMax, Cpuset, WorkingDir, CpuWeight }

sealed interface Validation {
    data class Valid(val spec: ContainerSpec) : Validation
    data class Invalid(
        val errors: Map<Field, String>,
        /** Keyed by [MountRow.id]. */
        val mountErrors: Map<Long, MountErrors> = emptyMap(),
    ) : Validation
}

/**
 * Pure form -> [ContainerSpec] validation, separate from the ViewModel so it
 * is exhaustively unit-testable. Mirrors the daemon's rules where they are
 * cheap to check client-side; the daemon remains the authority (its
 * INVALID_ARGUMENT messages are shown verbatim).
 */
object CreateFormValidator {
    private val NAME = Regex("^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}$")
    private val HOSTNAME = Regex("^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$")
    private val ENV = Regex("^[A-Za-z_][A-Za-z0-9_]*=.*$")

    fun validate(f: CreateForm): Validation {
        val errors = mutableMapOf<Field, String>()

        if (f.name.isNotBlank() && !NAME.matches(f.name.trim())) {
            errors[Field.Name] = "Letters, digits, '_', '.', '-' (max 63)"
        }
        if (f.rootfs.isBlank()) errors[Field.Rootfs] = "Required"
        else if (!f.rootfs.trim().startsWith("/")) errors[Field.Rootfs] = "Must be an absolute host path"

        val args = try {
            CommandLine.split(f.command)
        } catch (e: CommandLine.ParseException) {
            errors[Field.Command] = e.message ?: "Invalid command"
            emptyList()
        }
        if (args.isEmpty() && Field.Command !in errors) errors[Field.Command] = "Required"

        val env = f.env.lines().map { it.trim() }.filter { it.isNotEmpty() }
        env.firstOrNull { !ENV.matches(it) }?.let { errors[Field.Env] = "Invalid entry: \"$it\" (expected KEY=VALUE)" }

        if (f.hostname.isNotBlank() && !HOSTNAME.matches(f.hostname.trim())) errors[Field.Hostname] = "Invalid hostname"

        val cpu = Quantities.parseMillicores(f.cpu).also { if (it == null) errors[Field.Cpu] = "e.g. 500m or 1.5" }
        val memMax = Quantities.parseBytes(f.memoryMax).also { if (it == null) errors[Field.MemoryMax] = "e.g. 256Mi or 1Gi" }
        val memHigh = Quantities.parseBytes(f.memoryHigh).also { if (it == null) errors[Field.MemoryHigh] = "e.g. 200Mi" }
        if (memMax != null && memHigh != null && memMax > 0 && memHigh > memMax) {
            errors[Field.MemoryHigh] = "Must not exceed memory max"
        }
        val pids = Quantities.parseCount(f.pidsMax).also { if (it == null) errors[Field.PidsMax] = "Non-negative integer" }
        if (!Quantities.isValidCpuset(f.cpuset)) errors[Field.Cpuset] = "e.g. 0-1 or 0,2"

        val workingDir = f.workingDir.trim()
        if (workingDir.isNotEmpty() && !workingDir.startsWith("/")) errors[Field.WorkingDir] = "Must be an absolute path inside the container"

        val weight = Quantities.parseCount(f.cpuWeight)
        if (weight == null || (f.cpuWeight.isNotBlank() && weight !in CPU_WEIGHT_RANGE)) {
            errors[Field.CpuWeight] = "${CPU_WEIGHT_RANGE.first}..${CPU_WEIGHT_RANGE.last}"
        }

        val (mounts, mountErrors) = validateMounts(f.mounts)

        if (errors.isNotEmpty() || mountErrors.isNotEmpty()) return Validation.Invalid(errors, mountErrors)
        return Validation.Valid(
            ContainerSpec(
                name = f.name.trim(),
                rootfs = f.rootfs.trim(),
                args = args,
                env = env,
                hostname = f.hostname.trim(),
                limits = ResourceLimits(
                    cpuMillicores = cpu!!,
                    memoryMaxBytes = memMax!!,
                    memoryHighBytes = memHigh!!,
                    pidsMax = pids!!,
                    cpusetCpus = f.cpuset.trim(),
                    cpuWeight = weight!!,
                    disableSwap = f.disableSwap,
                ),
                network = f.network,
                seccomp = f.seccomp,
                trace = f.trace,
                workingDir = workingDir,
                mounts = mounts,
                readOnlyRootfs = f.readOnlyRootfs,
                autoRemove = f.autoRemove,
            ),
        )
    }

    /**
     * Rules: completely empty rows are ignored (an "add" tapped by mistake
     * shouldn't block submit); source and target are required absolute paths;
     * target may not be "/" (would shadow the rootfs) or contain ".."; two
     * mounts may not share a target (after normalizing trailing slashes) —
     * the second one is flagged.
     */
    private fun validateMounts(rows: List<MountRow>): Pair<List<Mount>, Map<Long, MountErrors>> {
        val errors = mutableMapOf<Long, MountErrors>()
        val mounts = mutableListOf<Mount>()
        val seenTargets = mutableSetOf<String>()
        for (row in rows) {
            if (row.isBlank) continue
            val source = row.source.trim()
            val target = normalizeTarget(row.target.trim())
            val sourceError = when {
                source.isEmpty() -> "Required"
                !source.startsWith("/") -> "Must be an absolute host path"
                else -> null
            }
            val targetError = when {
                target.isEmpty() -> "Required"
                !target.startsWith("/") -> "Must start with '/'"
                target == "/" -> "Cannot mount over the root"
                target.split('/').any { it == ".." } -> "Must not contain '..'"
                !seenTargets.add(target) -> "Duplicate target"
                else -> null
            }
            if (sourceError != null || targetError != null) {
                errors[row.id] = MountErrors(sourceError, targetError)
            } else {
                mounts += Mount(source, target, row.readOnly)
            }
        }
        return mounts to errors
    }

    private fun normalizeTarget(t: String): String =
        if (t.length > 1) t.trimEnd('/').ifEmpty { "/" } else t

    val CPU_WEIGHT_RANGE = 1..10_000
}

data class CreateUiState(
    val form: CreateForm = CreateForm(),
    /** Errors are only shown after the first submit attempt, then live. */
    val errors: Map<Field, String> = emptyMap(),
    val mountErrors: Map<Long, MountErrors> = emptyMap(),
    /** Advanced section expanded. Auto-expands when it contains an error. */
    val advancedExpanded: Boolean = false,
    val submitting: Boolean = false,
    val submitError: String? = null,
    /** Set once; the screen navigates to the detail page and calls [CreateContainerViewModel.createdHandled]. */
    val created: Container? = null,
)

class CreateContainerViewModel(private val repository: VesselRepository) : ViewModel() {
    private val _state = MutableStateFlow(CreateUiState())
    val state: StateFlow<CreateUiState> = _state.asStateFlow()
    private var attempted = false

    private var nextMountId = 0L

    fun update(transform: (CreateForm) -> CreateForm) {
        _state.update { s ->
            val form = transform(s.form)
            if (!attempted) return@update s.copy(form = form)
            val invalid = CreateFormValidator.validate(form) as? Validation.Invalid
            s.copy(form = form, errors = invalid?.errors.orEmpty(), mountErrors = invalid?.mountErrors.orEmpty())
        }
    }

    fun toggleAdvanced() = _state.update { it.copy(advancedExpanded = !it.advancedExpanded) }

    fun addMount() = update { it.copy(mounts = it.mounts + MountRow(id = nextMountId++)) }

    fun removeMount(id: Long) = update { f -> f.copy(mounts = f.mounts.filterNot { it.id == id }) }

    fun updateMount(id: Long, transform: (MountRow) -> MountRow) =
        update { f -> f.copy(mounts = f.mounts.map { if (it.id == id) transform(it) else it }) }

    fun submit() {
        if (_state.value.submitting) return
        attempted = true
        when (val v = CreateFormValidator.validate(_state.value.form)) {
            is Validation.Invalid -> _state.update {
                // Never hide an error inside a collapsed section.
                val advancedError = v.mountErrors.isNotEmpty() || v.errors.keys.any { f -> f in ADVANCED_FIELDS }
                it.copy(errors = v.errors, mountErrors = v.mountErrors, advancedExpanded = it.advancedExpanded || advancedError)
            }
            is Validation.Valid -> viewModelScope.launch {
                _state.update { it.copy(errors = emptyMap(), mountErrors = emptyMap(), submitting = true, submitError = null) }
                try {
                    val created = repository.createContainer(v.spec, start = _state.value.form.startImmediately)
                    _state.update { it.copy(submitting = false, created = created) }
                } catch (c: CancellationException) {
                    throw c
                } catch (t: Throwable) {
                    _state.update { it.copy(submitting = false, submitError = t.userMessage()) }
                }
            }
        }
    }

    fun createdHandled() = _state.update { it.copy(created = null) }

    companion object {
        val ADVANCED_FIELDS = setOf(Field.WorkingDir, Field.CpuWeight)
    }
}
