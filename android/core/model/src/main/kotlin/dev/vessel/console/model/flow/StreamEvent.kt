package dev.vessel.console.model.flow

/**
 * What a long-lived telemetry stream delivers to its collector: either data,
 * or an in-band notice that the transport dropped and a reconnect is
 * scheduled. Surfacing reconnects as values (instead of hiding them) lets the
 * UI show a "reconnecting…" banner and lets consumers reset state that would
 * otherwise be duplicated on resubscribe (e.g. log tails).
 */
sealed interface StreamEvent<out T> {
    data class Item<out T>(val value: T) : StreamEvent<T>

    data class Reconnecting(
        /** 1-based consecutive attempt number. */
        val attempt: Int,
        val delayMs: Long,
        val cause: Throwable,
    ) : StreamEvent<Nothing>
}
