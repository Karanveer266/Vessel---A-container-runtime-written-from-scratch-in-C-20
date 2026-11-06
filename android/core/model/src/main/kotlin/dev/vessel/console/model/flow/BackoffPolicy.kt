package dev.vessel.console.model.flow

import kotlin.math.min
import kotlin.math.pow
import kotlin.random.Random

/**
 * Exponential backoff with bounded jitter.
 *
 * delay(n) = min(maxDelay, initialDelay * multiplier^n) * (1 ± jitter)
 *
 * Jitter de-synchronizes many consoles reconnecting after a daemon restart
 * (thundering herd), and the cap keeps a phone that left Wi-Fi for an hour
 * from waiting minutes after it comes back. [random] is injectable so tests
 * are deterministic.
 */
data class BackoffPolicy(
    val initialDelayMs: Long = 500,
    val maxDelayMs: Long = 15_000,
    val multiplier: Double = 2.0,
    val jitter: Double = 0.2,
    /** Consecutive failed attempts before giving up; reset by any successful item. */
    val maxAttempts: Int = Int.MAX_VALUE,
    val random: Random = Random.Default,
) {
    init {
        require(initialDelayMs >= 0) { "initialDelayMs must be >= 0" }
        require(maxDelayMs >= initialDelayMs) { "maxDelayMs must be >= initialDelayMs" }
        require(multiplier >= 1.0) { "multiplier must be >= 1" }
        require(jitter in 0.0..1.0) { "jitter must be in [0, 1]" }
        require(maxAttempts >= 0) { "maxAttempts must be >= 0" }
    }

    /** Delay before retry number [attempt] (0-based). */
    fun delayFor(attempt: Int): Long {
        // Clamp the exponent so pow() never overflows to Infinity for long outages.
        val exp = initialDelayMs * multiplier.pow(min(attempt, 62).toDouble())
        val base = min(exp, maxDelayMs.toDouble())
        val factor = if (jitter == 0.0) 1.0 else 1.0 + jitter * (random.nextDouble() * 2 - 1)
        return (base * factor).toLong().coerceAtLeast(0)
    }

    companion object {
        /** Streams: retry forever, the UI shows "reconnecting (attempt n)". */
        val Streaming = BackoffPolicy()

        /** Idempotent unary reads: a couple of quick retries, then surface the error. */
        val UnaryRead = BackoffPolicy(initialDelayMs = 250, maxDelayMs = 2_000, maxAttempts = 2)
    }
}
