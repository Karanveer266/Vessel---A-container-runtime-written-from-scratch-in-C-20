package dev.vessel.console.ui.gpu

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.vessel.console.model.GpuBackend
import dev.vessel.console.model.GpuOp
import dev.vessel.console.model.format.Formatters
import dev.vessel.console.ui.components.Banner
import dev.vessel.console.ui.components.EnumDropdown
import dev.vessel.console.ui.components.FormField
import dev.vessel.console.ui.components.KeyValue
import dev.vessel.console.ui.components.LabeledSwitch
import dev.vessel.console.ui.components.SectionCard
import dev.vessel.console.ui.theme.VesselTheme

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GpuJobScreen(viewModel: GpuJobViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val f = state.form
    val e = state.errors
    Scaffold(topBar = { TopAppBar(title = { Text("GPU jobs") }) }) { padding ->
        LazyColumn(
            Modifier.fillMaxSize().padding(padding),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item(key = "form") {
                SectionCard("Workload") {
                    EnumDropdown("Operation", f.op, GpuOp.entries.filter { it != GpuOp.Unspecified }, { it.label }, { v -> viewModel.update { it.copy(op = v) } })
                    EnumDropdown("Backend", f.backend, GpuBackend.entries.filter { it != GpuBackend.Unspecified }, { it.label }, { v -> viewModel.update { it.copy(backend = v) } })
                    if (f.op.usesAttentionShape) {
                        Row {
                            Num("B·H", f.batchHeads, e["batchHeads"], Modifier.weight(1f)) { v -> viewModel.update { it.copy(batchHeads = v) } }
                            Spacer(Modifier.width(8.dp))
                            Num("seq len", f.seqLen, e["seqLen"], Modifier.weight(1f)) { v -> viewModel.update { it.copy(seqLen = v) } }
                            Spacer(Modifier.width(8.dp))
                            Num("head dim", f.headDim, e["headDim"], Modifier.weight(1f)) { v -> viewModel.update { it.copy(headDim = v) } }
                        }
                        LabeledSwitch("Causal mask", f.causal, { v -> viewModel.update { it.copy(causal = v) } })
                    } else {
                        Row {
                            Num(if (f.op == GpuOp.AllReduceBench) "elements" else "M", f.m, e["m"], Modifier.weight(1f)) { v -> viewModel.update { it.copy(m = v) } }
                            if (f.op != GpuOp.AllReduceBench) {
                                Spacer(Modifier.width(8.dp))
                                Num("N", f.n, e["n"], Modifier.weight(1f)) { v -> viewModel.update { it.copy(n = v) } }
                                Spacer(Modifier.width(8.dp))
                                Num("K", f.k, e["k"], Modifier.weight(1f)) { v -> viewModel.update { it.copy(k = v) } }
                            }
                        }
                    }
                    Row {
                        if (f.op.usesRanks) {
                            Num("Ranks", f.ranks, e["ranks"], Modifier.weight(1f)) { v -> viewModel.update { it.copy(ranks = v) } }
                            Spacer(Modifier.width(8.dp))
                        }
                        Num("Iterations", f.iterations, e["iterations"], Modifier.weight(1f)) { v -> viewModel.update { it.copy(iterations = v) } }
                    }
                    FormField("Sandbox memory max", f.memoryMax, { v -> viewModel.update { it.copy(memoryMax = v) } }, error = e["memoryMax"], hint = "optional, e.g. 4Gi")
                    LabeledSwitch("Verify against CPU reference", f.verify, { v -> viewModel.update { it.copy(verify = v) } })
                    if (state.running) {
                        LinearProgressIndicator(Modifier.fillMaxWidth())
                        OutlinedButton(onClick = viewModel::cancel, modifier = Modifier.fillMaxWidth()) { Text("Cancel") }
                    } else {
                        Button(onClick = viewModel::run, modifier = Modifier.fillMaxWidth()) { Text("Run") }
                    }
                }
            }
            state.error?.let { msg -> item(key = "error") { Banner(msg) } }
            items(state.history, key = { it.id }) { run -> ResultCard(run) }
        }
    }
}

@Composable
private fun Num(label: String, value: String, error: String?, modifier: Modifier, onChange: (String) -> Unit) =
    FormField(label, value, { onChange(it.filter(Char::isDigit)) }, modifier, error = error, keyboardType = KeyboardType.Number)

@Composable
private fun ResultCard(run: GpuRun) {
    val r = run.result
    val req = run.request
    val colors = VesselTheme.chartColors
    val shape = if (req.op.usesAttentionShape) "${req.batchHeads}×${req.seqLen}×${req.headDim}" else "${req.m}×${req.n}×${req.k}"
    SectionCard("${req.op.label} · $shape · ${r.backend.label}") {
        if (!r.ok) {
            Text("Failed: ${r.error}", color = colors.critical, style = MaterialTheme.typography.bodyMedium)
        }
        KeyValue("Time / iteration", Formatters.durationMs(r.timeMs))
        if (r.gflops > 0) KeyValue("Throughput", Formatters.gflops(r.gflops))
        if (r.busBandwidthGbps > 0) KeyValue("Bus bandwidth", Formatters.bandwidth(r.busBandwidthGbps))
        if (req.verify) KeyValue("Max abs error", "%.3g".format(java.util.Locale.ROOT, r.maxAbsError))
        if (r.deviceNames.isNotEmpty()) KeyValue("Devices", r.deviceNames)
        if (r.containerId.isNotEmpty()) KeyValue("Container", r.containerId, monospace = true)
        if (r.metrics.isNotEmpty()) {
            Text("Metrics", style = MaterialTheme.typography.labelLarge)
            r.metrics.forEach { (k, v) -> KeyValue(k, v, monospace = true) }
        }
    }
}
