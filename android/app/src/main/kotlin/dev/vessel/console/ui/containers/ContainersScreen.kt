package dev.vessel.console.ui.containers

import androidx.compose.foundation.clickable
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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.vessel.console.model.Container
import dev.vessel.console.model.ContainerAction
import dev.vessel.console.model.format.Formatters
import dev.vessel.console.ui.common.Load
import dev.vessel.console.ui.common.StreamStatus
import dev.vessel.console.ui.components.Banner
import dev.vessel.console.ui.components.CenteredProgress
import dev.vessel.console.ui.components.EmptyState
import dev.vessel.console.ui.components.ReconnectingBanner
import dev.vessel.console.ui.components.Sparkline
import dev.vessel.console.ui.components.StateChip
import dev.vessel.console.ui.theme.VesselTheme

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ContainersScreen(viewModel: ContainersViewModel, onOpen: (String) -> Unit, onCreate: () -> Unit) {
    // Lifecycle-aware: below STARTED this stops collecting, the StateFlow loses
    // its subscriber and (after the grace period) polling + StatsStream stop.
    val state by viewModel.state.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }
    var confirmRemove by remember { mutableStateOf<Container?>(null) }

    LaunchedEffect(state.message) {
        state.message?.let {
            snackbar.showSnackbar(it)
            viewModel.messageShown()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Containers") },
                actions = {
                    FilterChip(
                        selected = state.showStopped,
                        onClick = { viewModel.setShowStopped(!state.showStopped) },
                        label = { Text("Show stopped") },
                        modifier = Modifier.padding(end = 8.dp),
                    )
                },
            )
        },
        floatingActionButton = {
            FloatingActionButton(onClick = onCreate) { Icon(Icons.Filled.Add, contentDescription = "Create container") }
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        PullToRefreshBox(
            isRefreshing = state.refreshing,
            onRefresh = viewModel::refresh,
            modifier = Modifier.fillMaxSize().padding(padding),
        ) {
            when (val rows = state.rows) {
                Load.Idle, Load.Loading -> CenteredProgress()
                is Load.Failed -> Column(Modifier.padding(16.dp)) {
                    Banner(rows.message, actionLabel = "Retry", onAction = viewModel::refresh)
                }
                is Load.Ready -> LazyColumn(
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxSize(),
                ) {
                    (state.statsStatus as? StreamStatus.Reconnecting)?.let { r ->
                        item(key = "reconnecting") { ReconnectingBanner(r.attempt) }
                    }
                    if (rows.value.isEmpty()) item(key = "empty") { EmptyState("No containers. Tap + to create one.") }
                    // Stable keys: rows keep identity (and their remembered state) as the list re-sorts.
                    items(rows.value, key = { it.container.id }, contentType = { "container" }) { row ->
                        ContainerRowCard(
                            row = row,
                            onClick = { onOpen(row.container.id) },
                            onAction = { action ->
                                if (action == ContainerAction.Remove) confirmRemove = row.container
                                else viewModel.perform(row.container, action)
                            },
                            modifier = Modifier.animateItem(),
                        )
                    }
                }
            }
        }
    }

    confirmRemove?.let { c ->
        AlertDialog(
            onDismissRequest = { confirmRemove = null },
            title = { Text("Remove ${c.displayName}?") },
            text = { Text("Running containers are stopped first. This cannot be undone.") },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.perform(c, ContainerAction.Remove)
                    confirmRemove = null
                }) { Text("Remove") }
            },
            dismissButton = { TextButton(onClick = { confirmRemove = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun ContainerRowCard(
    row: ContainerRow,
    onClick: () -> Unit,
    onAction: (ContainerAction) -> Unit,
    modifier: Modifier = Modifier,
) {
    val c = row.container
    val colors = VesselTheme.chartColors
    Card(modifier.fillMaxWidth().clickable(onClick = onClick)) {
        Column(Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(c.displayName, style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(
                        "${c.id} · ${c.spec.args.joinToString(" ")}",
                        style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                StateChip(c.state)
                ActionsMenu(row, onAction)
            }
            if (c.oomKilled) {
                Text("OOM-killed", color = colors.critical, style = MaterialTheme.typography.labelMedium)
            }
            val latest = row.sparks.latest
            if (latest != null && !row.sparks.cpuMillicores.isEmpty()) {
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    MiniMetric(
                        label = "CPU ${Formatters.millicores(latest.cpuMillicoresUsed)}",
                        modifier = Modifier.weight(1f),
                    ) {
                        Sparkline(row.sparks.cpuMillicores, ContainersViewModel.SPARK_CAPACITY, colors.series1, Modifier.fillMaxWidth().height(28.dp), minYMax = 100f)
                    }
                    MiniMetric(
                        label = "MEM ${Formatters.bytes(latest.memoryCurrentBytes)}",
                        modifier = Modifier.weight(1f),
                    ) {
                        Sparkline(row.sparks.memoryBytes, ContainersViewModel.SPARK_CAPACITY, colors.series2, Modifier.fillMaxWidth().height(28.dp), minYMax = 1f * (1 shl 20))
                    }
                }
            } else if (c.state != dev.vessel.console.model.ContainerState.Running) {
                val detail = when {
                    c.error.isNotEmpty() -> c.error
                    c.state == dev.vessel.console.model.ContainerState.Stopped -> "exit code ${c.exitCode}"
                    else -> null
                }
                detail?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            }
        }
    }
}

@Composable
private fun MiniMetric(label: String, modifier: Modifier = Modifier, chart: @Composable () -> Unit) {
    Column(modifier) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        chart()
    }
}

@Composable
private fun ActionsMenu(row: ContainerRow, onAction: (ContainerAction) -> Unit) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    Box {
        if (row.pending != null) {
            CircularProgressIndicator(Modifier.padding(12.dp).size(24.dp), strokeWidth = 2.dp)
        } else {
            IconButton(onClick = { expanded = true }) { Icon(Icons.Filled.MoreVert, contentDescription = "Actions for ${row.container.displayName}") }
        }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            ContainerAction.entries.filter { it in row.container.state.allowedActions }.forEach { action ->
                DropdownMenuItem(text = { Text(action.label) }, onClick = {
                    expanded = false
                    onAction(action)
                })
            }
        }
    }
    Spacer(Modifier.width(0.dp))
}
