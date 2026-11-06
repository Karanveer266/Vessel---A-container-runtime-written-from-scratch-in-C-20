package dev.vessel.console

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import dev.vessel.console.ui.VesselApp
import dev.vessel.console.ui.theme.VesselTheme

/** Single activity; everything else is Compose destinations. */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        val container = (application as VesselApplication).container
        setContent {
            VesselTheme {
                VesselApp(container)
            }
        }
    }
}
