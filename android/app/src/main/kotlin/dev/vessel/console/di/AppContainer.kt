package dev.vessel.console.di

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.preferencesDataStore
import dev.vessel.console.api.ChannelManager
import dev.vessel.console.api.VesselClient
import dev.vessel.console.data.AndroidChannelFactory
import dev.vessel.console.data.DataStoreSettingsStore
import dev.vessel.console.data.KeystoreTokenCipher
import dev.vessel.console.data.SettingsStore
import dev.vessel.console.model.VesselRepository

private val Context.settingsDataStore: DataStore<Preferences> by preferencesDataStore(name = "connection")

/**
 * Manual dependency injection: the whole object graph in ~20 lines.
 *
 * The app has exactly three singletons (settings store, channel manager,
 * repository) and five ViewModels. Hilt would add kapt/KSP, an annotation
 * processor pass on every build, generated code to debug, and Gradle plugin
 * coupling to AGP versions — for a graph this small, constructor injection
 * through one container is clearer and keeps the build fast. ViewModel
 * factories (see `ui/ViewModelFactories.kt`) pull from here. If the graph grew
 * (multiple daemons, background sync workers), migrating is mechanical
 * because every class already takes its dependencies via the constructor.
 */
class AppContainer(context: Context) {
    val settingsStore: SettingsStore = DataStoreSettingsStore(context.settingsDataStore, KeystoreTokenCipher())

    val channelManager = ChannelManager(AndroidChannelFactory(context))

    val repository: VesselRepository = VesselClient(settingsStore.settings, channelManager)
}
