package dev.vessel.console.model.flow

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.retryWhen
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Re-subscribes to the upstream on retryable failures using [policy], emitting
 * [StreamEvent.Reconnecting] before each wait.
 *
 * Built on [retryWhen] so only *upstream* exceptions are retried — an exception
 * thrown by a downstream collector propagates untouched (flow exception
 * transparency). The attempt counter is per-collection and resets after every
 * successfully delivered item, so a stream that ran fine for an hour and then
 * blipped starts again at the initial (short) delay.
 *
 * Normal completion of the upstream completes this flow: the daemon ends
 * telemetry streams deliberately when the container goes away.
 */
fun <T> Flow<T>.retryWithBackoff(
    policy: BackoffPolicy,
    isRetryable: (Throwable) -> Boolean,
): Flow<StreamEvent<T>> = flow {
    var attempt = 0
    emitAll(
        this@retryWithBackoff
            .map<T, StreamEvent<T>> {
                attempt = 0
                StreamEvent.Item(it)
            }
            .retryWhen { cause, _ ->
                if (!isRetryable(cause) || attempt >= policy.maxAttempts) return@retryWhen false
                val wait = policy.delayFor(attempt)
                attempt++
                emit(StreamEvent.Reconnecting(attempt, wait, cause))
                delay(wait)
                true
            },
    )
}

/** Drops reconnect notices, keeping only data. */
fun <T> Flow<StreamEvent<T>>.items(): Flow<T> = flow {
    collect { if (it is StreamEvent.Item) emit(it.value) }
}

/**
 * Groups a high-rate flow into batches emitted at most once per [periodMs].
 *
 * The syscall tracer can deliver thousands of events per second; rebuilding
 * UI state per event would burn the main thread. Batching turns N events into
 * one state update per frame-ish period.
 *
 * Memory stays bounded even if the collector is slow: the pending batch keeps
 * only the newest [maxBatch] items (drop-oldest — the same policy the daemon
 * uses server-side). Pick [maxBatch] equal to the consumer's ring buffer
 * capacity: anything older would be evicted from the ring anyway.
 *
 * Empty periods emit nothing. A final partial batch is flushed on completion.
 */
fun <T> Flow<T>.batchByTime(periodMs: Long, maxBatch: Int): Flow<List<T>> {
    require(periodMs > 0) { "periodMs must be > 0" }
    require(maxBatch > 0) { "maxBatch must be > 0" }
    return channelFlow {
        val lock = Mutex()
        val pending = ArrayDeque<T>()

        suspend fun drain(): List<T>? = lock.withLock {
            if (pending.isEmpty()) null else pending.toList().also { pending.clear() }
        }

        val ticker = launch {
            while (true) {
                delay(periodMs)
                drain()?.let { send(it) }
            }
        }
        collect { item ->
            lock.withLock {
                if (pending.size == maxBatch) pending.removeFirst()
                pending.addLast(item)
            }
        }
        ticker.cancel()
        drain()?.let { send(it) }
    }
}
