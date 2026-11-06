package dev.vessel.console.model.flow

import app.cash.turbine.test
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.currentTime
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException
import kotlin.test.assertFailsWith

@OptIn(ExperimentalCoroutinesApi::class)
class FlowOperatorsTest {

    private val noJitter = BackoffPolicy(initialDelayMs = 100, maxDelayMs = 1_000, jitter = 0.0)

    @Test
    fun `retries transient failures with growing delays and reports them in-band`() = runTest {
        var subscriptions = 0
        val upstream = flow {
            subscriptions++
            if (subscriptions <= 2) throw IOException("blip $subscriptions")
            emit("ok")
        }

        val events = upstream.retryWithBackoff(noJitter) { it is IOException }.toList()

        assertEquals(3, subscriptions)
        assertEquals(3, events.size)
        val r1 = events[0] as StreamEvent.Reconnecting
        val r2 = events[1] as StreamEvent.Reconnecting
        assertEquals(1 to 100L, r1.attempt to r1.delayMs)
        assertEquals(2 to 200L, r2.attempt to r2.delayMs)
        assertEquals(StreamEvent.Item("ok"), events[2])
        assertEquals(300L, currentTime) // virtual time: 100 + 200
    }

    @Test
    fun `attempt counter resets after a successful item`() = runTest {
        var subscriptions = 0
        val upstream = flow {
            subscriptions++
            emit(subscriptions)
            if (subscriptions < 3) throw IOException("drop")
        }
        val reconnects = upstream.retryWithBackoff(noJitter) { true }
            .toList()
            .filterIsInstance<StreamEvent.Reconnecting>()
        // Each failure followed a delivered item, so every reconnect is "attempt 1".
        assertEquals(listOf(1, 1), reconnects.map { it.attempt })
        assertEquals(listOf(100L, 100L), reconnects.map { it.delayMs })
    }

    @Test
    fun `non-retryable failures propagate immediately`() = runTest {
        val upstream = flow<Int> { throw IllegalStateException("auth") }
        assertFailsWith<IllegalStateException> {
            upstream.retryWithBackoff(noJitter) { it is IOException }.toList()
        }
    }

    @Test
    fun `gives up after maxAttempts consecutive failures`() = runTest {
        var subscriptions = 0
        val upstream = flow<Int> {
            subscriptions++
            throw IOException("down")
        }
        assertFailsWith<IOException> {
            upstream.retryWithBackoff(noJitter.copy(maxAttempts = 3)) { true }.toList()
        }
        assertEquals(4, subscriptions) // initial + 3 retries
    }

    @Test
    fun `downstream exceptions are not retried`() = runTest {
        var subscriptions = 0
        val upstream = flow {
            subscriptions++
            emit(1)
        }
        assertFailsWith<ArithmeticException> {
            upstream.retryWithBackoff(noJitter) { true }.collect { throw ArithmeticException() }
        }
        assertEquals(1, subscriptions)
    }

    @Test
    fun `items filters out reconnect notices`() = runTest {
        var first = true
        val upstream = flow {
            if (first) { first = false; throw IOException() }
            emit(7)
        }
        assertEquals(listOf(7), upstream.retryWithBackoff(noJitter) { true }.items().toList())
    }

    @Test
    fun `batchByTime groups items per period and flushes the tail`() = runTest {
        val upstream = flow {
            emit(1); emit(2)
            kotlinx.coroutines.delay(150)
            emit(3)
            kotlinx.coroutines.delay(10)
        }
        val batches = upstream.batchByTime(periodMs = 100, maxBatch = 100).toList()
        assertEquals(listOf(listOf(1, 2), listOf(3)), batches)
    }

    @Test
    fun `batchByTime keeps only the newest maxBatch items when the producer outruns the period`() = runTest {
        val upstream = flow { (1..10).forEach { emit(it) } }
        val batches = upstream.batchByTime(periodMs = 1_000, maxBatch = 3).toList()
        assertEquals(listOf(listOf(8, 9, 10)), batches)
    }

    @Test
    fun `batchByTime emits nothing for quiet periods`() = runTest {
        val source = MutableSharedFlow<Int>()
        source.batchByTime(periodMs = 100, maxBatch = 10).test {
            kotlinx.coroutines.delay(1_000)
            expectNoEvents()
            source.emit(5)
            assertEquals(listOf(5), awaitItem())
            cancelAndIgnoreRemainingEvents()
        }
        assertTrue(true)
    }
}
