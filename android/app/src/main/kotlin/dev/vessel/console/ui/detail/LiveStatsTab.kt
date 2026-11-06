package dev.vessel.console.ui.detail

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.vessel.console.model.Container
import dev.vessel.console.model.ContainerState
import dev.vessel.console.model.StatsSample
import dev.vessel.console.model.format.Formatters
import dev.vessel.console.ui.common.StreamStatus
import dev.vessel.console.ui.components.Banner
import dev.vessel.console.ui.components.ChartSeries
import dev.vessel.console.ui.components.EmptyState
import dev.vessel.console.ui.components.KeyValue
import dev.vessel.console.ui.components.LineChart
import dev.vessel.console.ui.components.ReconnectingBanner
import dev.vessel.console.ui.components.SectionCard
import dev.vessel.console.ui.theme.VesselTheme

@Composable
fun LiveStatsTab(viewModel: LiveStatsViewModel, container: Container) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val stats = state.stats
    val colors = VesselTheme.chartColors
    val latest = stats.latest

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        StatusBanner(state.status)
        if (latest == null) {
            EmptyState(
                if (container.state == ContainerState.Running) "Waiting for samples…"
                else "Container is ${container.state.label}; stats are only sampled while running.",
            )
            return@Column
        }
        if (latest.hasOomEvents || container.oomKilled) OomBanner(latest)

        SectionCard("CPU · ${Formatters.millicores(latest.cpuMillicoresUsed)} cores") {
            val limit = container.spec.limits.cpuMillicores
            LineChart(
                series = remember(stats.cpuMillicores, colors) { listOf(ChartSeries("used", stats.cpuMillicores, colors.series1)) },
                capacity = state.capacity,
                formatY = { Formatters.millicores(it.toInt()) },
                minYMax = if (limit > 0) limit.toFloat() else 1_000f,
                summary = "CPU usage chart, currently ${Formatters.millicores(latest.cpuMillicoresUsed)} cores",
            )
            KeyValue("Limit", Formatters.millicores(limit, zeroLabel = "unlimited"))
            ThrottleRow(latest)
        }

        SectionCard("Memory · ${Formatters.bytes(latest.memoryCurrentBytes)}") {
            val hasLimit = latest.memoryMaxBytes > 0
            val series = remember(stats.memoryCurrentBytes, stats.memoryMaxBytes, colors, hasLimit) {
                buildList {
                    add(ChartSeries("current", stats.memoryCurrentBytes, colors.series1))
                    // Limit drawn dashed: distinguishable without relying on colour.
                    if (hasLimit) add(ChartSeries("max", stats.memoryMaxBytes, colors.limit, dashed = true))
                }
            }
            LineChart(
                series = series,
                capacity = state.capacity,
                formatY = { Formatters.bytes(it.toLong()) },
                minYMax = 16f * (1 shl 20),
                summary = "Memory chart, ${Formatters.bytes(latest.memoryCurrentBytes)} of ${Formatters.bytesOrUnlimited(latest.memoryMaxBytes)}",
            )
            KeyValue("Peak", Formatters.bytes(latest.memoryPeakBytes))
            KeyValue("Max", Formatters.bytesOrUnlimited(latest.memoryMaxBytes))
            KeyValue("memory.events high / max", "${latest.memoryEventsHigh} / ${latest.memoryEventsMax}")
        }

        SectionCard("PIDs · ${latest.pidsCurrent}${if (latest.pidsMax > 0) " / ${latest.pidsMax}" else ""}") {
            LineChart(
                series = remember(stats.pids, colors) { listOf(ChartSeries("tasks", stats.pids, colors.series3)) },
                capacity = state.capacity,
                formatY = { it.toInt().toString() },
                minYMax = 8f,
                height = 100.dp,
                summary = "Task count chart, currently ${latest.pidsCurrent}",
            )
        }

        SectionCard("Pressure (PSI some avg10)") {
            LineChart(
                series = remember(stats.psiCpuSome10, stats.psiMemorySome10, stats.psiIoSome10, colors) {
                    listOf(
                        ChartSeries("cpu", stats.psiCpuSome10, colors.series1),
                        ChartSeries("memory", stats.psiMemorySome10, colors.series2),
                        ChartSeries("io", stats.psiIoSome10, colors.series3),
                    )
                },
                capacity = state.capacity,
                formatY = { Formatters.psi(it) },
                minYMax = 10f,
                summary = "Pressure stall chart",
            )
            KeyValue("cpu / mem / io", listOf(latest.cpuPressure, latest.memoryPressure, latest.ioPressure).joinToString(" / ") { Formatters.psi(it.someAvg10) })
        }

        SectionCard("Network") {
            KeyValue("rx", Formatters.bytes(latest.netRxBytes))
            KeyValue("tx", Formatters.bytes(latest.netTxBytes))
            if (latest.droppedSamples > 0) KeyValue("Dropped samples", latest.droppedSamples.toString())
        }
    }
}

@Composable
internal fun StatusBanner(status: StreamStatus) {
    when (status) {
        is StreamStatus.Reconnecting -> ReconnectingBanner(status.attempt)
        is StreamStatus.Failed -> Banner(status.message)
        StreamStatus.Connecting, StreamStatus.Live -> Unit
    }
}

@Composable
private fun ThrottleRow(s: StatsSample) {
    val colors = VesselTheme.chartColors
    val throttled = s.cpuNrThrottled > 0
    Row(verticalAlignment = Alignment.CenterVertically) {
        if (throttled) Icon(Icons.Filled.Warning, contentDescription = "Throttled", tint = colors.warning, modifier = Modifier.padding(end = 4.dp))
        KeyValue("Throttled", "${s.cpuNrThrottled}× · ${Formatters.durationNs(s.cpuThrottledUsec * 1_000)}")
    }
}

@Composable
private fun OomBanner(s: StatsSample) {
    val critical = VesselTheme.chartColors.critical
    Surface(color = critical, contentColor = Color.White, shape = MaterialTheme.shapes.medium, modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.Warning, contentDescription = null, modifier = Modifier.padding(end = 8.dp))
            Text(
                "OOM: ${s.memoryEventsOom} events, ${s.memoryEventsOomKill} kills",
                style = MaterialTheme.typography.titleSmall,
            )
        }
    }
}
