package dev.vessel.console.ui.detail

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.PrimaryScrollableTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.vessel.console.di.AppContainer
import dev.vessel.console.model.Container
import dev.vessel.console.ui.common.Load
import dev.vessel.console.ui.components.Banner
import dev.vessel.console.ui.components.CenteredProgress
import dev.vessel.console.ui.components.StateChip
import dev.vessel.console.ui.containerViewModel

enum class DetailTab(val title: String) { Stats("Live stats"), Trace("Syscalls"), Summary("Summary"), Logs("Logs"), Exec("Exec") }

/**
 * Container detail with tabs.
 *
 * Each tab has its own ViewModel scoped to this back-stack entry, and only the
 * selected tab is composed. Its StateFlow therefore has a subscriber only
 * while visible: switching from "Syscalls" to "Logs" stops the trace stream
 * (after the 5 s grace period) and starts the log stream. With all five
 * streams open at once a phone on Wi-Fi would be decoding traffic it never
 * shows. ViewModels survive tab switches, so returning is instant for
 * non-stream state (exec output, filters).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ContainerDetailScreen(app: AppContainer, containerId: String, onBack: () -> Unit) {
    val header = containerViewModel(app, key = "header-$containerId") { ContainerHeaderViewModel(containerId, repository) }
    val state by header.state.collectAsStateWithLifecycle()
    var tab by rememberSaveable { mutableStateOf(DetailTab.Stats) }
    val snackbar = remember { SnackbarHostState() }

    LaunchedEffect(state.removed) { if (state.removed) onBack() }
    LaunchedEffect(state.message) {
        state.message?.let {
            snackbar.showSnackbar(it)
            header.messageShown()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(state.container.valueOrNull?.displayName ?: containerId, maxLines = 1, overflow = TextOverflow.Ellipsis)
                },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") }
                },
                actions = {
                    state.container.valueOrNull?.let { c ->
                        StateChip(c.state)
                        HeaderActions(c, state.pending != null, header::perform)
                    }
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            when (val c = state.container) {
                is Load.Failed -> Banner(c.message, actionLabel = "Retry", onAction = header::refresh)
                Load.Loading, Load.Idle -> CenteredProgress()
                is Load.Ready -> {
                    PrimaryScrollableTabRow(selectedTabIndex = tab.ordinal) {
                        DetailTab.entries.forEach { t ->
                            Tab(selected = t == tab, onClick = { tab = t }, text = { Text(t.title) })
                        }
                    }
                    when (tab) {
                        DetailTab.Stats -> LiveStatsTab(
                            containerViewModel(app, key = "stats-$containerId") { LiveStatsViewModel(containerId, repository) },
                            c.value,
                        )
                        DetailTab.Trace -> TraceTab(
                            containerViewModel(app, key = "trace-$containerId") { TraceViewModel(containerId, repository) },
                        )
                        DetailTab.Summary -> SummaryTab(
                            containerViewModel(app, key = "summary-$containerId") { SyscallSummaryViewModel(containerId, repository) },
                        )
                        DetailTab.Logs -> LogsTab(
                            containerViewModel(app, key = "logs-$containerId") { LogsViewModel(containerId, repository) },
                        )
                        DetailTab.Exec -> ExecTab(
                            containerViewModel(app, key = "exec-$containerId") { ExecViewModel(containerId, repository) },
                            running = c.value.state == dev.vessel.console.model.ContainerState.Running,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun HeaderActions(c: Container, busy: Boolean, onAction: (dev.vessel.console.model.ContainerAction) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    IconButton(onClick = { expanded = true }, enabled = !busy) { Icon(Icons.Filled.MoreVert, contentDescription = "Container actions") }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
        dev.vessel.console.model.ContainerAction.entries.filter { it in c.state.allowedActions }.forEach { a ->
            DropdownMenuItem(
                text = { Text(a.label, color = if (a == dev.vessel.console.model.ContainerAction.Remove) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface) },
                onClick = {
                    expanded = false
                    onAction(a)
                },
            )
        }
    }
}
