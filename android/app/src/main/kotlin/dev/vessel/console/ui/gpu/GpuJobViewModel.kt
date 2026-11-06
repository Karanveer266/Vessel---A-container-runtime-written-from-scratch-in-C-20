package dev.vessel.console.ui.gpu

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.vessel.console.model.GpuBackend
import dev.vessel.console.model.GpuJobRequest
import dev.vessel.console.model.GpuJobResult
import dev.vessel.console.model.GpuOp
import dev.vessel.console.model.ResourceLimits
import dev.vessel.console.model.VesselRepository
import dev.vessel.console.model.format.Quantities
import dev.vessel.console.model.userMessage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class GpuForm(
    val op: GpuOp = GpuOp.Gemm,
    val backend: GpuBackend = GpuBackend.Auto,
    val m: String = "1024",
    val n: String = "1024",
    val k: String = "1024",
    val ranks: String = "2",
    val batchHeads: String = "16",
    val seqLen: String = "1024",
    val headDim: String = "64",
    val causal: Boolean = true,
    val iterations: String = "10",
    val verify: Boolean = true,
    val memoryMax: String = "",
) {
    /** Returns the request or a field->message error map. Only fields relevant to [op] are validated. */
    fun toRequest(): Pair<GpuJobRequest?, Map<String, String>> {
        val errors = mutableMapOf<String, String>()
        fun dim(name: String, raw: String, max: Int = 65_536): Int {
            val v = raw.trim().toIntOrNull()
            if (v == null || v !in 1..max) errors[name] = "1..$max"
            return v ?: 0
        }
        val usesMatrix = !op.usesAttentionShape
        // For AllReduceBench `m` is the element count (often millions), not a matrix dimension.
        val m = when {
            op == GpuOp.AllReduceBench -> dim("m", m, max = MAX_ALLREDUCE_ELEMENTS)
            usesMatrix -> dim("m", m)
            else -> 0
        }
        val n = if (usesMatrix && op != GpuOp.AllReduceBench) dim("n", n) else 0
        val k = if (usesMatrix && op != GpuOp.AllReduceBench) dim("k", k) else 0
        val ranks = if (op.usesRanks) dim("ranks", ranks, max = 64) else 1
        val bh = if (op.usesAttentionShape) dim("batchHeads", batchHeads, 4_096) else 0
        val sl = if (op.usesAttentionShape) dim("seqLen", seqLen) else 0
        val hd = if (op.usesAttentionShape) dim("headDim", headDim, 256) else 0
        val it = dim("iterations", iterations, 10_000)
        val mem = Quantities.parseBytes(memoryMax).also { if (it == null) errors["memoryMax"] = "e.g. 4Gi" } ?: 0
        if (op == GpuOp.Unspecified) errors["op"] = "Choose an operation"
        if (errors.isNotEmpty()) return null to errors
        return GpuJobRequest(
            op = op, backend = backend, m = m, n = n, k = k, ranks = ranks,
            batchHeads = bh, seqLen = sl, headDim = hd, causal = causal && op.usesAttentionShape,
            iterations = it, verify = verify, limits = ResourceLimits(memoryMaxBytes = mem),
        ) to emptyMap()
    }
}

private const val MAX_ALLREDUCE_ELEMENTS = 1 shl 30

/** One completed run, kept so the user can compare configurations. */
data class GpuRun(val id: Int, val request: GpuJobRequest, val result: GpuJobResult)

data class GpuUiState(
    val form: GpuForm = GpuForm(),
    val errors: Map<String, String> = emptyMap(),
    val running: Boolean = false,
    val error: String? = null,
    /** Newest first, bounded. */
    val history: List<GpuRun> = emptyList(),
)

class GpuJobViewModel(private val repository: VesselRepository) : ViewModel() {
    private val _state = MutableStateFlow(GpuUiState())
    val state: StateFlow<GpuUiState> = _state.asStateFlow()
    private var job: Job? = null
    private var nextId = 0

    fun update(transform: (GpuForm) -> GpuForm) = _state.update { it.copy(form = transform(it.form), errors = emptyMap()) }

    fun run() {
        if (_state.value.running) return
        val (request, errors) = _state.value.form.toRequest()
        if (request == null) {
            _state.update { it.copy(errors = errors) }
            return
        }
        _state.update { it.copy(running = true, error = null) }
        job = viewModelScope.launch {
            try {
                val result = repository.runGpuJob(request)
                _state.update {
                    it.copy(running = false, history = (listOf(GpuRun(nextId++, request, result)) + it.history).take(MAX_HISTORY))
                }
            } catch (c: CancellationException) {
                _state.update { it.copy(running = false, error = "Cancelled") }
                throw c
            } catch (t: Throwable) {
                _state.update { it.copy(running = false, error = t.userMessage()) }
            }
        }
    }

    /** Cancels the RPC client-side (the gRPC call is cancelled; the daemon tears down the job container). */
    fun cancel() {
        job?.cancel()
    }

    companion object {
        const val MAX_HISTORY = 10
    }
}
