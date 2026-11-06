package dev.vessel.console.ui.connect

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.vessel.console.data.SettingsStore
import dev.vessel.console.model.ConnectionSettings
import dev.vessel.console.model.HostInfo
import dev.vessel.console.model.VesselRepository
import dev.vessel.console.model.userMessage
import dev.vessel.console.ui.common.Load
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** Editable form fields are kept as raw strings so partially typed input (e.g. empty port) is representable. */
data class ConnectForm(
    val host: String = ConnectionSettings.DEFAULT_HOST,
    val port: String = ConnectionSettings.DEFAULT_PORT.toString(),
    val token: String = "",
    val useTls: Boolean = false,
) {
    fun toSettings(): ConnectionSettings? {
        val p = port.trim().toIntOrNull() ?: return null
        return ConnectionSettings(host.trim(), p, token.trim(), useTls).takeIf { it.validate().isEmpty() }
    }

    val errors: List<String>
        get() = port.trim().toIntOrNull()?.let { ConnectionSettings(host, it, token, useTls).validate() }
            ?: listOf("Port must be a number")

    companion object {
        fun from(s: ConnectionSettings) = ConnectForm(s.host, s.port.toString(), s.token, s.useTls)
    }
}

data class ConnectUiState(
    val form: ConnectForm = ConnectForm(),
    val loaded: Boolean = false,
    /** True when the form differs from what is persisted. */
    val dirty: Boolean = false,
    val host: Load<HostInfo> = Load.Idle,
) {
    val canConnect: Boolean get() = loaded && form.errors.isEmpty() && host !is Load.Loading
    val insecureWarning: Boolean get() = form.toSettings()?.isInsecureRemote == true
}

/**
 * Connect/Host screen. "Connect" = persist settings, then fetch HostInfo
 * through the repository — which reads the *persisted* settings, so a
 * successful probe proves the exact configuration every other screen will use.
 */
class ConnectViewModel(
    private val settingsStore: SettingsStore,
    private val repository: VesselRepository,
) : ViewModel() {

    private val _state = MutableStateFlow(ConnectUiState())
    val state: StateFlow<ConnectUiState> = _state.asStateFlow()

    private var saved: ConnectionSettings? = null
    private var probe: Job? = null

    init {
        viewModelScope.launch {
            val current = settingsStore.settings.first()
            saved = current
            _state.update { it.copy(form = ConnectForm.from(current), loaded = true) }
            connect() // auto-probe the saved daemon on launch
        }
    }

    fun onFormChange(form: ConnectForm) {
        _state.update { it.copy(form = form, dirty = form.toSettings() != saved) }
    }

    fun connect() {
        val settings = _state.value.form.toSettings() ?: return
        probe?.cancel()
        probe = viewModelScope.launch {
            _state.update { it.copy(host = Load.Loading) }
            try {
                if (settings != saved) {
                    settingsStore.save(settings)
                    saved = settings
                }
                val info = repository.hostInfo()
                _state.update { it.copy(host = Load.Ready(info), dirty = false) }
            } catch (c: CancellationException) {
                throw c
            } catch (t: Throwable) {
                _state.update { it.copy(host = Load.Failed(t.userMessage()), dirty = false) }
            }
        }
    }
}
