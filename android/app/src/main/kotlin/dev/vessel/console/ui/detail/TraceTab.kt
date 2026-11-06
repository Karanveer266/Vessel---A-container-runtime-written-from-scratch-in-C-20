package dev.vessel.console.ui.detail

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.vessel.console.model.TraceEvent
import dev.vessel.console.model.TraceEventKind
import dev.vessel.console.model.format.Formatters
import dev.vessel.console.ui.components.EmptyState
import dev.vessel.console.ui.components.FormField
import dev.vessel.console.ui.theme.VesselTheme

@Composable
fun TraceTab(viewModel: TraceViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val listState = rememberLazyListState()

    // Newest-first list: stay pinned to the top while the user is at the top;
    // if they scrolled down to inspect something, don't yank the list.
    val firstKey = state.events.firstOrNull()?.seq
    LaunchedEffect(firstKey) {
        if (listState.firstVisibleItemIndex <= 1 && !state.paused) listState.scrollToItem(0)
    }

    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            StatusBanner(state.status)
            Row(verticalAlignment = Alignment.CenterVertically) {
                FilterChip(
                    selected = state.input.errorsOnly,
                    onClick = { viewModel.onFilterChange { it.copy(errorsOnly = !it.errorsOnly) } },
                    label = { Text("Errors only") },
                )
                Spacer(Modifier.width(8.dp))
                FilterChip(selected = state.paused, onClick = viewModel::togglePause, label = { Text(if (state.paused) "Paused" else "Pause") })
                Spacer(Modifier.weight(1f))
                TextButton(onClick = viewModel::clear) { Text("Clear") }
            }
            Row {
                FormField(
                    "Syscalls", state.input.syscalls, { v -> viewModel.onFilterChange { it.copy(syscalls = v) } },
                    Modifier.weight(2f), hint = "openat, read", monospace = true,
                )
                Spacer(Modifier.width(8.dp))
                FormField(
                    "Min µs", state.input.minLatencyUs, { v -> viewModel.onFilterChange { it.copy(minLatencyUs = v.filter(Char::isDigit)) } },
                    Modifier.weight(1f), keyboardType = KeyboardType.Number,
                )
            }
            Text(
                buildString {
                    append("${state.events.size}/${state.capacity} shown · ${Formatters.count(state.received)} received")
                    if (state.droppedUpstream > 0) append(" · ${Formatters.count(state.droppedUpstream)} dropped by daemon")
                    if (state.evictedLocally > 0) append(" · ${Formatters.count(state.evictedLocally)} rotated out")
                },
                style = MaterialTheme.typography.labelSmall,
                color = if (state.droppedUpstream > 0) VesselTheme.chartColors.serious else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        HorizontalDivider()
        if (state.events.isEmpty()) {
            EmptyState("No events yet. Tracing must be enabled for this container (trace backend ≠ none).")
        } else {
            LazyColumn(state = listState, modifier = Modifier.fillMaxSize()) {
                // seq is client-assigned and unique per subscription: stable keys
                // let LazyColumn reuse row compositions as the window slides.
                items(state.events, key = { it.seq }, contentType = { it.kind }) { e -> TraceRow(e) }
            }
        }
    }
}

@Composable
private fun TraceRow(e: TraceEvent) {
    val colors = VesselTheme.chartColors
    val mono = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace)
    val rowBg = if (e.isError) colors.critical.copy(alpha = 0.08f) else androidx.compose.ui.graphics.Color.Transparent
    Column(Modifier.fillMaxWidth().background(rowBg).padding(horizontal = 16.dp, vertical = 4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                when (e.kind) {
                    TraceEventKind.Syscall -> e.syscallName.ifEmpty { "#${e.syscallNr}" }
                    else -> e.kind.label
                },
                style = mono.copy(color = if (e.isError) colors.critical else MaterialTheme.colorScheme.onSurface),
                modifier = Modifier.weight(1f),
                maxLines = 1,
            )
            if (e.kind == TraceEventKind.Syscall) {
                Text("= ${Formatters.syscallReturn(e.ret)}", style = mono)
                Spacer(Modifier.width(8.dp))
                Text(Formatters.durationNs(e.latencyNs), style = mono, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Text(
            buildString {
                append("${e.comm}[${e.hostPid}")
                if (e.hostTid != e.hostPid) append("/${e.hostTid}")
                append("]")
                if (e.bytes > 0) append(" ${Formatters.bytes(e.bytes)}${if (e.kind == TraceEventKind.Net) if (e.egress) " ↑" else " ↓" else ""}")
                if (e.detail.isNotEmpty()) append(" ${e.detail}")
                if (e.droppedEvents > 0) append("  ⚠ ${e.droppedEvents} dropped before")
            },
            style = mono,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}
