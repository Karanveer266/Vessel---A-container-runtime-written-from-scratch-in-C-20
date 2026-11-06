package dev.vessel.console.ui.detail

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.vessel.console.model.LogStream
import dev.vessel.console.model.SyscallSummary
import dev.vessel.console.model.format.Formatters
import dev.vessel.console.ui.common.Load
import dev.vessel.console.ui.components.Banner
import dev.vessel.console.ui.components.CenteredProgress
import dev.vessel.console.ui.components.ChartMath
import dev.vessel.console.ui.components.EmptyState
import dev.vessel.console.ui.components.FormField
import dev.vessel.console.ui.components.HBar
import dev.vessel.console.ui.components.KeyValue
import dev.vessel.console.ui.components.SectionCard
import dev.vessel.console.ui.theme.VesselTheme

// ============================================================================
// Summary: top syscalls bar chart + latency
// ============================================================================

private enum class SummarySort(val label: String) { Count("count"), TotalTime("total time"), Errors("errors") }

@Composable
fun SummaryTab(viewModel: SyscallSummaryViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    when (val s = state.summary) {
        Load.Idle, Load.Loading -> CenteredProgress()
        is Load.Failed -> Column(Modifier.padding(16.dp)) { Banner(s.message, actionLabel = "Retry", onAction = viewModel::refresh) }
        is Load.Ready -> SummaryContent(s.value, viewModel::refresh)
    }
}

@Composable
private fun SummaryContent(summary: SyscallSummary, onRefresh: () -> Unit) {
    var sort by rememberSaveable { mutableStateOf(SummarySort.Count) }
    val colors = VesselTheme.chartColors
    val rows = when (sort) {
        SummarySort.Count -> summary.syscalls
        SummarySort.TotalTime -> summary.syscalls.sortedByDescending { it.totalLatencyNs }
        SummarySort.Errors -> summary.syscalls.sortedByDescending { it.errors }
    }.take(TOP_N)
    val max = rows.maxOfOrNull {
        when (sort) {
            SummarySort.Count -> it.count
            SummarySort.TotalTime -> it.totalLatencyNs
            SummarySort.Errors -> it.errors
        }
    } ?: 0L

    LazyColumn(contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        item(key = "header") {
            SectionCard("Totals") {
                KeyValue("Events", Formatters.count(summary.totalEvents))
                KeyValue("Dropped", Formatters.count(summary.droppedEvents))
                KeyValue("Backend", summary.backend.label)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    SummarySort.entries.forEach { s ->
                        FilterChip(selected = s == sort, onClick = { sort = s }, label = { Text(s.label) }, modifier = Modifier.padding(end = 6.dp))
                    }
                    Spacer(Modifier.weight(1f))
                    TextButton(onClick = onRefresh) { Text("Refresh") }
                }
            }
        }
        if (rows.isEmpty()) item(key = "empty") { EmptyState("No syscalls recorded yet.") }
        items(rows, key = { it.name.ifEmpty { "#${it.nr}" } }) { c ->
            val value = when (sort) {
                SummarySort.Count -> c.count
                SummarySort.TotalTime -> c.totalLatencyNs
                SummarySort.Errors -> c.errors
            }
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        c.name.ifEmpty { "#${c.nr}" },
                        style = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
                        modifier = Modifier.width(110.dp),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Box(Modifier.weight(1f).height(14.dp)) {
                        HBar(
                            ChartMath.barFraction(value, max),
                            if (sort == SummarySort.Errors) colors.critical else colors.series1,
                            Modifier.fillMaxSize(),
                        )
                    }
                    Text(
                        when (sort) {
                            SummarySort.TotalTime -> Formatters.durationNs(value)
                            else -> Formatters.count(value)
                        },
                        style = MaterialTheme.typography.labelMedium,
                        modifier = Modifier.width(64.dp).padding(start = 8.dp),
                    )
                }
                Text(
                    "avg ${Formatters.durationNs(c.avgLatencyNs)} · max ${Formatters.durationNs(c.maxLatencyNs)}" +
                        if (c.errors > 0) " · ${Formatters.count(c.errors)} errors" else "",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(start = 110.dp),
                )
            }
        }
    }
}

private const val TOP_N = 25

// ============================================================================
// Logs
// ============================================================================

@Composable
fun LogsTab(viewModel: LogsViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val listState = rememberLazyListState()
    var follow by rememberSaveable { mutableStateOf(true) }
    val colors = VesselTheme.chartColors

    LaunchedEffect(state.lines.lastOrNull()?.seq, follow) {
        if (follow && state.lines.isNotEmpty()) listState.scrollToItem(state.lines.lastIndex)
    }

    Column(Modifier.fillMaxSize()) {
        Row(Modifier.padding(horizontal = 16.dp, vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            FilterChip(selected = follow, onClick = { follow = !follow }, label = { Text("Follow") })
            Spacer(Modifier.weight(1f))
            Text(
                if (state.ended) "stream ended · ${state.lines.size} lines" else "${state.lines.size} lines",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Box(Modifier.padding(horizontal = 16.dp)) { StatusBanner(state.status) }
        if (state.lines.isEmpty()) {
            EmptyState("No output yet.")
        } else {
            SelectionContainer {
                LazyColumn(state = listState, modifier = Modifier.fillMaxSize().padding(horizontal = 8.dp)) {
                    items(state.lines, key = { it.seq }) { line ->
                        Text(
                            line.text,
                            style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                            color = if (line.stream == LogStream.Stderr) colors.critical else MaterialTheme.colorScheme.onSurface,
                        )
                    }
                }
            }
        }
    }
}

// ============================================================================
// Exec
// ============================================================================

@Composable
fun ExecTab(viewModel: ExecViewModel, running: Boolean) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val colors = VesselTheme.chartColors
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (!running) Banner("Exec requires a running container.")
        FormField("Command", state.command, viewModel::onCommandChange, error = state.error, monospace = true, hint = "argv; use sh -c '…' for shell features")
        Button(onClick = viewModel::run, enabled = running && !state.running, modifier = Modifier.fillMaxWidth()) {
            if (state.running) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp) else Text("Run")
        }
        state.result?.let { r ->
            SectionCard("Result") {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        if (r.timedOut) "timed out" else "exit ${r.exitCode}",
                        color = if (r.exitCode == 0 && !r.timedOut) colors.good else colors.critical,
                        style = MaterialTheme.typography.titleSmall,
                    )
                    Spacer(Modifier.weight(1f))
                    Text("${state.durationMs} ms", style = MaterialTheme.typography.labelSmall)
                }
                OutputBlock("stdout", r.stdout)
                if (r.stderr.isNotEmpty()) OutputBlock("stderr", r.stderr)
            }
        }
    }
}

@Composable
private fun OutputBlock(label: String, text: String) {
    Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    SelectionContainer {
        Text(
            text.ifEmpty { "(empty)" }.take(MAX_OUTPUT_CHARS),
            style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
            modifier = Modifier.horizontalScroll(rememberScrollState()),
        )
    }
}

/** Exec output is buffered daemon-side; cap what we lay out to keep text measurement cheap. */
private const val MAX_OUTPUT_CHARS = 64 * 1024
