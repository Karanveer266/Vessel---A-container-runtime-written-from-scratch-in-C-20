package dev.vessel.console.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color

/**
 * Data-visualization palette, separate from the Material scheme.
 *
 * Categorical slots are assigned in a fixed order (never cycled) and each mode
 * has its own validated steps — dark mode is not an automatic inversion.
 * Status colours are reserved for state (OOM = critical, throttling = warning)
 * and always ship with an icon + label, never colour alone.
 */
@Immutable
data class ChartColors(
    val series1: Color,
    val series2: Color,
    val series3: Color,
    val grid: Color,
    val axisText: Color,
    val limit: Color,
    val good: Color = Color(0xFF0CA30C),
    val warning: Color = Color(0xFFFAB219),
    val serious: Color = Color(0xFFEC835A),
    val critical: Color = Color(0xFFD03B3B),
)

private val LightChart = ChartColors(
    series1 = Color(0xFF2A78D6),
    series2 = Color(0xFFEB6834),
    series3 = Color(0xFF1BAF7A),
    grid = Color(0xFFE4E3DF),
    axisText = Color(0xFF52514E),
    limit = Color(0xFF8A8984),
)

private val DarkChart = ChartColors(
    series1 = Color(0xFF3987E5),
    series2 = Color(0xFFD95926),
    series3 = Color(0xFF199E70),
    grid = Color(0xFF383835),
    axisText = Color(0xFFC3C2B7),
    limit = Color(0xFF8A8984),
)

val LocalChartColors = staticCompositionLocalOf { LightChart }

private val LightScheme = lightColorScheme(
    primary = Color(0xFF2A78D6),
    secondary = Color(0xFF52514E),
    background = Color(0xFFF9F9F7),
    surface = Color(0xFFFCFCFB),
    error = Color(0xFFD03B3B),
)

private val DarkScheme = darkColorScheme(
    primary = Color(0xFF7FB3F0),
    secondary = Color(0xFFC3C2B7),
    background = Color(0xFF0D0D0D),
    surface = Color(0xFF1A1A19),
    surfaceContainer = Color(0xFF222221),
    error = Color(0xFFE66767),
)

@Composable
fun VesselTheme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    CompositionLocalProvider(LocalChartColors provides if (darkTheme) DarkChart else LightChart) {
        MaterialTheme(colorScheme = if (darkTheme) DarkScheme else LightScheme, content = content)
    }
}

object VesselTheme {
    val chartColors: ChartColors
        @Composable get() = LocalChartColors.current
}
