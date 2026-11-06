package dev.vessel.console.ui

import dev.vessel.console.ui.components.ChartMath
import org.junit.Assert.assertEquals
import org.junit.Test

class ChartMathTest {
    @Test
    fun `nice ceiling snaps to 1-2-2_5-5 steps`() {
        assertEquals(1f, ChartMath.niceCeiling(0f))
        assertEquals(1000f, ChartMath.niceCeiling(0f, floor = 1000f))
        assertEquals(100f, ChartMath.niceCeiling(87f))
        assertEquals(250f, ChartMath.niceCeiling(201f))
        assertEquals(500f, ChartMath.niceCeiling(251f))
        assertEquals(2000f, ChartMath.niceCeiling(1500f))
        assertEquals(1000f, ChartMath.niceCeiling(1000f))
    }

    @Test
    fun `x positions anchor newest sample at the right edge`() {
        // capacity 5 over width 100 => slot 25
        assertEquals(100f, ChartMath.xFor(i = 2, n = 3, capacity = 5, width = 100f))
        assertEquals(50f, ChartMath.xFor(i = 0, n = 3, capacity = 5, width = 100f))
        assertEquals(0f, ChartMath.xFor(i = 0, n = 5, capacity = 5, width = 100f))
    }

    @Test
    fun `indexAt inverts xFor and clamps`() {
        for (i in 0 until 3) {
            assertEquals(i, ChartMath.indexAt(ChartMath.xFor(i, 3, 5, 100f), 3, 5, 100f))
        }
        assertEquals(0, ChartMath.indexAt(-50f, 3, 5, 100f))
        assertEquals(2, ChartMath.indexAt(500f, 3, 5, 100f))
        assertEquals(-1, ChartMath.indexAt(10f, 0, 5, 100f))
    }

    @Test
    fun `y is clamped into the plot`() {
        assertEquals(100f, ChartMath.yFor(0f, 10f, 100f))
        assertEquals(0f, ChartMath.yFor(10f, 10f, 100f))
        assertEquals(0f, ChartMath.yFor(50f, 10f, 100f))
        assertEquals(50f, ChartMath.yFor(5f, 10f, 100f))
    }

    @Test
    fun `bar fractions keep small non-zero bars visible`() {
        assertEquals(1f, ChartMath.barFraction(100, 100))
        assertEquals(0.01f, ChartMath.barFraction(1, 1_000_000))
        assertEquals(0f, ChartMath.barFraction(0, 100))
    }
}
