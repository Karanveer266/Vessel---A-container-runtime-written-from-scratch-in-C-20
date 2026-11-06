package dev.vessel.console.ui.components

import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.roundToInt

/**
 * Pure geometry for the Canvas charts — no Compose types, so it is unit
 * tested on the JVM.
 */
object ChartMath {

    /**
     * Rounds [value] up to a "nice" axis maximum (1, 2, 2.5, 5 × 10^n) so grid
     * lines land on human-friendly numbers and the axis does not jitter with
     * every sample. Returns [floor] when the data is all zero.
     */
    fun niceCeiling(value: Float, floor: Float = 1f): Float {
        if (value.isNaN() || value <= 0f) return floor
        val exp = floor(log10(value.toDouble()))
        val base = 10.0.pow(exp)
        val fraction = value / base
        val nice = when {
            fraction <= 1.0 -> 1.0
            fraction <= 2.0 -> 2.0
            fraction <= 2.5 -> 2.5
            fraction <= 5.0 -> 5.0
            else -> 10.0
        }
        return maxOf((nice * base).toFloat(), floor)
    }

    /**
     * Maps sample i of n to an x coordinate. The window is anchored on the
     * right edge (newest sample) with a fixed slot width of
     * `width / (capacity - 1)`, so a partially filled buffer grows from the
     * right like a strip chart instead of stretching across the whole width.
     */
    fun xFor(i: Int, n: Int, capacity: Int, width: Float): Float {
        if (capacity <= 1) return width
        val slot = width / (capacity - 1)
        return width - (n - 1 - i) * slot
    }

    fun yFor(value: Float, yMax: Float, height: Float): Float {
        if (yMax <= 0f) return height
        return height - (value.coerceIn(0f, yMax) / yMax) * height
    }

    /** Inverse of [xFor]: which sample index is under a touch at [x] (clamped). */
    fun indexAt(x: Float, n: Int, capacity: Int, width: Float): Int {
        if (n == 0) return -1
        if (capacity <= 1) return n - 1
        val slot = width / (capacity - 1)
        val fromRight = ((width - x) / slot).roundToInt()
        return (n - 1 - fromRight).coerceIn(0, n - 1)
    }

    /** Evenly spaced gridline values from 0 to [yMax] inclusive. */
    fun gridValues(yMax: Float, lines: Int = 4): List<Float> =
        (0..lines).map { yMax * it / lines }

    /** Bars for a ranked list: fraction of the largest value, min visible width for non-zero. */
    fun barFraction(value: Long, max: Long, minVisible: Float = 0.01f): Float = when {
        max <= 0L || value <= 0L -> 0f
        else -> maxOf(value.toFloat() / max, minVisible)
    }
}
