package dev.vessel.console.model.flow

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class BackoffPolicyTest {

    @Test
    fun `grows exponentially and caps without jitter`() {
        val p = BackoffPolicy(initialDelayMs = 100, maxDelayMs = 1_000, multiplier = 2.0, jitter = 0.0)
        assertEquals(listOf(100L, 200L, 400L, 800L, 1_000L, 1_000L), (0..5).map(p::delayFor))
    }

    @Test
    fun `huge attempt numbers do not overflow`() {
        val p = BackoffPolicy(initialDelayMs = 100, maxDelayMs = 5_000, jitter = 0.0)
        assertEquals(5_000L, p.delayFor(10_000))
    }

    @Test
    fun `jitter stays within bounds`() {
        val p = BackoffPolicy(initialDelayMs = 1_000, maxDelayMs = 1_000, jitter = 0.2, random = Random(42))
        repeat(1_000) {
            val d = p.delayFor(it)
            assertTrue("delay $d out of bounds", d in 800..1_200)
        }
    }
}
