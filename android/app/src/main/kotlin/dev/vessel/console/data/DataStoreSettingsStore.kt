package dev.vessel.console.data

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import dev.vessel.console.model.ConnectionSettings
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import java.io.IOException

/**
 * [SettingsStore] on Jetpack DataStore Preferences.
 *
 * Why Preferences and not Proto DataStore: four scalar fields don't justify a
 * second protobuf schema and codegen step; Preferences gives the same
 * transactional, coroutine-native, main-safe I/O.
 *
 * The bearer token is encrypted with an Android Keystore key ([TokenCipher])
 * before it touches disk, so a backup or a rooted-device file dump does not
 * leak daemon credentials.
 */
class DataStoreSettingsStore(
    private val dataStore: DataStore<Preferences>,
    private val cipher: TokenCipher,
) : SettingsStore {

    override val settings: Flow<ConnectionSettings> = dataStore.data
        .catch { e -> if (e is IOException) emit(emptyPreferences()) else throw e }
        .map { prefs ->
            ConnectionSettings(
                host = prefs[HOST] ?: ConnectionSettings.DEFAULT_HOST,
                port = prefs[PORT] ?: ConnectionSettings.DEFAULT_PORT,
                token = prefs[TOKEN]?.let(cipher::decrypt).orEmpty(),
                useTls = prefs[TLS] ?: false,
            )
        }
        .distinctUntilChanged()

    override suspend fun save(settings: ConnectionSettings) {
        dataStore.edit { prefs ->
            prefs[HOST] = settings.host.trim()
            prefs[PORT] = settings.port
            prefs[TLS] = settings.useTls
            if (settings.token.isEmpty()) prefs.remove(TOKEN) else prefs[TOKEN] = cipher.encrypt(settings.token)
        }
    }

    private companion object {
        val HOST = stringPreferencesKey("host")
        val PORT = intPreferencesKey("port")
        val TOKEN = stringPreferencesKey("token_enc")
        val TLS = booleanPreferencesKey("tls")
    }
}
