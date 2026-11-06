package dev.vessel.console.testing

import dev.vessel.console.data.SettingsStore
import dev.vessel.console.model.ConnectionSettings
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class FakeSettingsStore(initial: ConnectionSettings = ConnectionSettings()) : SettingsStore {
    private val state = MutableStateFlow(initial)
    override val settings: StateFlow<ConnectionSettings> = state
    var saves = 0

    override suspend fun save(settings: ConnectionSettings) {
        saves++
        state.value = settings
    }
}
