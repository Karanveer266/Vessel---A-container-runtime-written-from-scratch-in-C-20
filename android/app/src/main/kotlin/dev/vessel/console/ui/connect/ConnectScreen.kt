package dev.vessel.console.ui.connect

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.vessel.console.model.HostInfo
import dev.vessel.console.model.format.Formatters
import dev.vessel.console.ui.common.Load
import dev.vessel.console.ui.components.Banner
import dev.vessel.console.ui.components.CenteredProgress
import dev.vessel.console.ui.components.FormField
import dev.vessel.console.ui.components.KeyValue
import dev.vessel.console.ui.components.LabeledSwitch
import dev.vessel.console.ui.components.SectionCard
import androidx.compose.material3.TextButton

@Composable
fun ConnectScreen(viewModel: ConnectViewModel, onOpenContainers: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    ConnectContent(state, viewModel::onFormChange, viewModel::connect, onOpenContainers)
}

/** Stateless content: previewable and trivially driven by any [ConnectUiState]. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConnectContent(
    state: ConnectUiState,
    onFormChange: (ConnectForm) -> Unit,
    onConnect: () -> Unit,
    onOpenContainers: () -> Unit,
) {
    Scaffold(topBar = { TopAppBar(title = { Text("Vessel Console") }) }) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SettingsCard(state, onFormChange, onConnect)
            when (val host = state.host) {
                Load.Idle -> Unit
                Load.Loading -> CenteredProgress()
                is Load.Failed -> Banner(host.message, actionLabel = "Retry", onAction = onConnect)
                is Load.Ready -> {
                    HostInfoCards(host.value)
                    Button(onClick = onOpenContainers, modifier = Modifier.fillMaxWidth()) { Text("Open containers") }
                }
            }
        }
    }
}

@Composable
private fun SettingsCard(state: ConnectUiState, onFormChange: (ConnectForm) -> Unit, onConnect: () -> Unit) {
    val form = state.form
    var showToken by rememberSaveable { mutableStateOf(false) }
    SectionCard("Daemon") {
        Row {
            FormField("Host", form.host, { onFormChange(form.copy(host = it)) }, Modifier.weight(2f), keyboardType = KeyboardType.Uri)
            androidx.compose.foundation.layout.Spacer(Modifier.width(8.dp))
            FormField("Port", form.port, { onFormChange(form.copy(port = it.filter(Char::isDigit).take(5))) }, Modifier.weight(1f), keyboardType = KeyboardType.Number)
        }
        OutlinedTextField(
            value = form.token,
            onValueChange = { onFormChange(form.copy(token = it)) },
            label = { Text("Bearer token") },
            singleLine = true,
            visualTransformation = if (showToken) VisualTransformation.None else PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
            trailingIcon = { TextButton(onClick = { showToken = !showToken }) { Text(if (showToken) "Hide" else "Show") } },
            modifier = Modifier.fillMaxWidth(),
        )
        LabeledSwitch("TLS", form.useTls, { onFormChange(form.copy(useTls = it)) })
        state.form.errors.forEach { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
        if (state.insecureWarning) {
            Banner("Token will be sent unencrypted to a non-loopback host. Enable TLS on untrusted networks.")
        }
        Text(
            "USB: adb reverse tcp:50051 tcp:50051 → 127.0.0.1 · Emulator: 10.0.2.2 · Same device: 127.0.0.1",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        OutlinedButton(onClick = onConnect, enabled = state.canConnect, modifier = Modifier.fillMaxWidth()) {
            Text(if (state.dirty) "Save & connect" else "Reconnect")
        }
    }
}

@Composable
private fun HostInfoCards(h: HostInfo) {
    SectionCard("Host") {
        KeyValue("Kernel", h.kernelRelease, monospace = true)
        KeyValue("Machine", h.machine + if (h.isAndroid) " (Android)" else "")
        KeyValue("Daemon", h.daemonVersion.ifEmpty { "?" })
        KeyValue("CPUs", h.onlineCpus.toString())
        KeyValue("Memory", Formatters.bytes(h.totalMemoryBytes))
    }
    SectionCard("Isolation") {
        KeyValue("cgroup", h.cgroupVersion.label)
        KeyValue("Controllers", h.cgroupControllers.joinToString(" ").ifEmpty { "—" }, monospace = true)
        KeyValue("Rootless", yesNo(h.rootless))
        KeyValue("PSI", yesNo(h.psiAvailable))
    }
    SectionCard("Tracing") {
        KeyValue("eBPF available", yesNo(h.ebpfAvailable))
        KeyValue("Default backend", h.defaultTraceBackend.label)
    }
    SectionCard("GPUs") {
        if (h.gpus.isEmpty()) {
            Text("No GPUs — GPU jobs can use the simulated backend.", style = MaterialTheme.typography.bodyMedium)
        }
        h.gpus.forEach { g ->
            KeyValue("#${g.index} ${g.name}", "${Formatters.bytes(g.totalMemoryBytes)} · ${g.computeCapability}")
        }
    }
}

private fun yesNo(b: Boolean) = if (b) "yes" else "no"
