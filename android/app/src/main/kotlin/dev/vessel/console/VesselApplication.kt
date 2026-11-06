package dev.vessel.console

import android.app.Application
import dev.vessel.console.di.AppContainer

class VesselApplication : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }
}
