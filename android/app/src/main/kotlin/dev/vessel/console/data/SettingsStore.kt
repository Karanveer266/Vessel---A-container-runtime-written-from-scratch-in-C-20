package dev.vessel.console.data

import dev.vessel.console.model.ConnectionSettings
import kotlinx.coroutines.flow.Flow

/** Persistence port for connection settings (DataStore in prod, in-memory in tests). */
interface SettingsStore {
    /** Emits the current settings immediately, then on every change. */
    val settings: Flow<ConnectionSettings>

    suspend fun save(settings: ConnectionSettings)
}
