package dev.vessel.console.model

/**
 * Transport-agnostic error vocabulary. `:core:api` translates gRPC `Status`
 * codes into these so ViewModels never import `io.grpc` and can render
 * actionable messages ("check your token" vs "daemon unreachable").
 */
sealed class VesselException(message: String, cause: Throwable? = null) : Exception(message, cause) {
    /** Transient — worth retrying (daemon restarting, Wi-Fi flapping, deadline). */
    abstract val isRetryable: Boolean

    class Unavailable(message: String, cause: Throwable? = null) : VesselException(message, cause) {
        override val isRetryable = true
    }

    class Unauthenticated(message: String, cause: Throwable? = null) : VesselException(message, cause) {
        override val isRetryable = false
    }

    class NotFound(message: String, cause: Throwable? = null) : VesselException(message, cause) {
        override val isRetryable = false
    }

    class InvalidArgument(message: String, cause: Throwable? = null) : VesselException(message, cause) {
        override val isRetryable = false
    }

    class FailedPrecondition(message: String, cause: Throwable? = null) : VesselException(message, cause) {
        override val isRetryable = false
    }

    class Unimplemented(message: String, cause: Throwable? = null) : VesselException(message, cause) {
        override val isRetryable = false
    }

    class Other(message: String, override val isRetryable: Boolean, cause: Throwable? = null) :
        VesselException(message, cause)
}

/** Short, user-facing description for snackbars / error banners. */
fun Throwable.userMessage(): String = when (this) {
    is VesselException.Unauthenticated -> "Authentication failed — check the bearer token"
    is VesselException.Unavailable -> "Daemon unreachable: ${message.orEmpty()}"
    is VesselException.NotFound -> "Not found: ${message.orEmpty()}"
    is VesselException -> message ?: this::class.simpleName.orEmpty()
    else -> message ?: this::class.simpleName.orEmpty()
}
