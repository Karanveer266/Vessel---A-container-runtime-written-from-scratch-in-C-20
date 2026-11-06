package dev.vessel.console.ui.common

import dev.vessel.console.model.flow.StreamEvent
import kotlinx.coroutines.flow.SharingStarted

/** Async value for one-shot loads. */
sealed interface Load<out T> {
    data object Idle : Load<Nothing>
    data object Loading : Load<Nothing>
    data class Ready<T>(val value: T) : Load<T>
    data class Failed(val message: String) : Load<Nothing>

    val valueOrNull: T? get() = (this as? Ready<T>)?.value
}

/** Connection health of a live stream, for banners. */
sealed interface StreamStatus {
    data object Connecting : StreamStatus
    data object Live : StreamStatus
    data class Reconnecting(val attempt: Int, val reason: String) : StreamStatus
    data class Failed(val message: String) : StreamStatus
}

fun StreamEvent.Reconnecting.toStatus(): StreamStatus.Reconnecting =
    StreamStatus.Reconnecting(attempt, cause.message ?: cause::class.simpleName.orEmpty())

/**
 * Sharing policy for every screen-level StateFlow that is backed by a gRPC
 * stream.
 *
 * Paired with `collectAsStateWithLifecycle()` in the UI, which stops
 * collecting below STARTED: when the app goes to background (or the user
 * switches detail tabs), the subscriber count drops to zero and after
 * [STOP_TIMEOUT_MS] the upstream — and therefore the gRPC call — is cancelled.
 * The grace period means a configuration change (rotation, ~hundreds of ms)
 * does *not* tear down and re-open the stream.
 */
object Sharing {
    const val STOP_TIMEOUT_MS = 5_000L
    val WhileUiVisible: SharingStarted = SharingStarted.WhileSubscribed(stopTimeoutMillis = STOP_TIMEOUT_MS)
}
