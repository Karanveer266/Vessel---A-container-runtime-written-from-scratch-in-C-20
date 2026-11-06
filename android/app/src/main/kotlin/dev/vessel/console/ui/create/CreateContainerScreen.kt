package dev.vessel.console.ui.create

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.vessel.console.model.NetworkMode
import dev.vessel.console.model.SeccompProfile
import dev.vessel.console.model.TraceBackend
import dev.vessel.console.ui.components.Banner
import dev.vessel.console.ui.components.EnumDropdown
import dev.vessel.console.ui.components.FormField
import dev.vessel.console.ui.components.LabeledSwitch
import dev.vessel.console.ui.components.SectionCard

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreateContainerScreen(viewModel: CreateContainerViewModel, onBack: () -> Unit, onCreated: (String) -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val f = state.form
    val e = state.errors

    // One-shot navigation event modelled as state + acknowledgement.
    LaunchedEffect(state.created) {
        state.created?.let {
            viewModel.createdHandled()
            onCreated(it.id)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("New container") },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .imePadding()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SectionCard("Workload") {
                FormField("Name", f.name, { v -> viewModel.update { it.copy(name = v) } }, error = e[Field.Name], hint = "Optional; daemon generates an id")
                FormField("Root filesystem", f.rootfs, { v -> viewModel.update { it.copy(rootfs = v) } }, error = e[Field.Rootfs], hint = "Host path, e.g. /var/lib/vessel/rootfs/alpine", monospace = true)
                FormField("Command", f.command, { v -> viewModel.update { it.copy(command = v) } }, error = e[Field.Command], hint = "argv, shell-style quoting", monospace = true, singleLine = false)
                FormField("Environment", f.env, { v -> viewModel.update { it.copy(env = v) } }, error = e[Field.Env], hint = "One KEY=VALUE per line", monospace = true, singleLine = false)
                FormField("Hostname", f.hostname, { v -> viewModel.update { it.copy(hostname = v) } }, error = e[Field.Hostname])
            }
            SectionCard("Isolation") {
                EnumDropdown("Network", f.network, NetworkMode.entries, { it.label }, { v -> viewModel.update { it.copy(network = v) } })
                EnumDropdown("Seccomp profile", f.seccomp, SeccompProfile.entries, { it.label }, { v -> viewModel.update { it.copy(seccomp = v) } })
                EnumDropdown("Syscall tracing", f.trace, TraceBackend.entries, { it.label }, { v -> viewModel.update { it.copy(trace = v) } })
            }
            SectionCard("Resources (empty = unlimited)") {
                Row {
                    FormField("CPU", f.cpu, { v -> viewModel.update { it.copy(cpu = v) } }, Modifier.weight(1f), error = e[Field.Cpu], hint = "500m, 1.5")
                    Spacer(Modifier.width(8.dp))
                    FormField("PIDs max", f.pidsMax, { v -> viewModel.update { it.copy(pidsMax = v) } }, Modifier.weight(1f), error = e[Field.PidsMax], keyboardType = KeyboardType.Number)
                }
                Row {
                    FormField("Memory max", f.memoryMax, { v -> viewModel.update { it.copy(memoryMax = v) } }, Modifier.weight(1f), error = e[Field.MemoryMax], hint = "256Mi")
                    Spacer(Modifier.width(8.dp))
                    FormField("Memory high", f.memoryHigh, { v -> viewModel.update { it.copy(memoryHigh = v) } }, Modifier.weight(1f), error = e[Field.MemoryHigh], hint = "soft limit")
                }
                FormField("cpuset", f.cpuset, { v -> viewModel.update { it.copy(cpuset = v) } }, error = e[Field.Cpuset], hint = "0-1 or 0,2")
            }
            SectionCard("Lifecycle") {
                LabeledSwitch("Start immediately", f.startImmediately, { v -> viewModel.update { it.copy(startImmediately = v) } })
                LabeledSwitch("Remove on exit", f.autoRemove, { v -> viewModel.update { it.copy(autoRemove = v) } })
            }
            AdvancedSection(state, viewModel)
            state.submitError?.let { Banner(it) }
            Button(onClick = viewModel::submit, enabled = !state.submitting, modifier = Modifier.fillMaxWidth()) {
                if (state.submitting) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                else Text(if (f.startImmediately) "Create & start" else "Create")
            }
        }
    }
}

/**
 * Less common ContainerSpec fields, collapsed by default so the main form
 * stays short. The ViewModel auto-expands it when validation fails inside.
 */
@Composable
private fun AdvancedSection(state: CreateUiState, viewModel: CreateContainerViewModel) {
    val f = state.form
    val e = state.errors
    val summary = buildList {
        if (f.workingDir.isNotBlank()) add("workdir")
        if (f.readOnlyRootfs) add("ro rootfs")
        if (f.cpuWeight.isNotBlank()) add("weight")
        if (f.disableSwap) add("no swap")
        val mounts = f.mounts.count { !it.isBlank }
        if (mounts > 0) add("$mounts mount${if (mounts > 1) "s" else ""}")
    }.joinToString(" · ")

    Card(Modifier.fillMaxWidth()) {
        Row(
            Modifier
                .fillMaxWidth()
                .clickable(onClickLabel = if (state.advancedExpanded) "Collapse" else "Expand", onClick = viewModel::toggleAdvanced)
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text("Advanced", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
                if (summary.isNotEmpty() && !state.advancedExpanded) {
                    Text(summary, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Icon(
                if (state.advancedExpanded) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                contentDescription = null,
            )
        }
        AnimatedVisibility(visible = state.advancedExpanded) {
            Column(
                Modifier.padding(start = 16.dp, end = 16.dp, bottom = 16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FormField("Working directory", f.workingDir, { v -> viewModel.update { it.copy(workingDir = v) } }, error = e[Field.WorkingDir], hint = "Inside the container, e.g. /srv", monospace = true)
                FormField(
                    "CPU weight", f.cpuWeight, { v -> viewModel.update { it.copy(cpuWeight = v.filter(Char::isDigit)) } },
                    error = e[Field.CpuWeight], hint = "Relative share 1..10000 (cpu.weight); empty = default", keyboardType = KeyboardType.Number,
                )
                LabeledSwitch("Read-only root filesystem", f.readOnlyRootfs, { v -> viewModel.update { it.copy(readOnlyRootfs = v) } })
                LabeledSwitch("Disable swap", f.disableSwap, { v -> viewModel.update { it.copy(disableSwap = v) } })

                HorizontalDivider(Modifier.padding(vertical = 4.dp))
                Text("Bind mounts", style = MaterialTheme.typography.labelLarge)
                f.mounts.forEach { row ->
                    // key() keeps each row's text-field state attached to its id when rows above are removed.
                    key(row.id) { MountRowEditor(row, state.mountErrors[row.id], viewModel) }
                }
                OutlinedButton(onClick = viewModel::addMount, modifier = Modifier.fillMaxWidth()) {
                    Icon(Icons.Filled.Add, contentDescription = null)
                    Spacer(Modifier.width(4.dp))
                    Text("Add mount")
                }
            }
        }
    }
}

@Composable
private fun MountRowEditor(row: MountRow, errors: MountErrors?, viewModel: CreateContainerViewModel) {
    OutlinedCard(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            FormField("Host source", row.source, { v -> viewModel.updateMount(row.id) { it.copy(source = v) } }, error = errors?.source, monospace = true)
            FormField("Container target", row.target, { v -> viewModel.updateMount(row.id) { it.copy(target = v) } }, error = errors?.target, hint = "Absolute path, e.g. /data", monospace = true)
            Row(verticalAlignment = Alignment.CenterVertically) {
                LabeledSwitch("Read-only", row.readOnly, { v -> viewModel.updateMount(row.id) { it.copy(readOnly = v) } }, Modifier.weight(1f))
                IconButton(onClick = { viewModel.removeMount(row.id) }) {
                    Icon(Icons.Filled.Delete, contentDescription = "Remove mount ${row.target.ifBlank { row.source }}")
                }
            }
        }
    }
}
