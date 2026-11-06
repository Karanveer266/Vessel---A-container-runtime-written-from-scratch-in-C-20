package dev.vessel.console.api

import dev.vessel.console.model.VesselException
import io.grpc.Status
import io.grpc.StatusException
import io.grpc.StatusRuntimeException
import kotlinx.coroutines.CancellationException

/**
 * Translates gRPC failures into the domain [VesselException] hierarchy.
 *
 * Retryability follows the gRPC status-code guide: UNAVAILABLE is the
 * canonical transient error; DEADLINE_EXCEEDED, ABORTED and
 * RESOURCE_EXHAUSTED are worth retrying for idempotent reads and streams.
 * Everything else (auth, invalid argument, not found…) is a caller problem
 * that retrying cannot fix.
 *
 * [CancellationException] is never wrapped — doing so would break structured
 * concurrency (a cancelled ViewModel scope must stay cancelled).
 */
fun Throwable.toVesselException(): Throwable {
    if (this is VesselException || this is CancellationException) return this
    val status = when (this) {
        is StatusException -> status
        is StatusRuntimeException -> status
        else -> return VesselException.Other(message ?: javaClass.simpleName, isRetryable = false, cause = this)
    }
    val description = status.description ?: status.cause?.message ?: status.code.name
    return when (status.code) {
        Status.Code.UNAVAILABLE -> VesselException.Unavailable(description, this)
        Status.Code.UNAUTHENTICATED, Status.Code.PERMISSION_DENIED ->
            VesselException.Unauthenticated(description, this)
        Status.Code.NOT_FOUND -> VesselException.NotFound(description, this)
        Status.Code.INVALID_ARGUMENT, Status.Code.OUT_OF_RANGE, Status.Code.ALREADY_EXISTS ->
            VesselException.InvalidArgument(description, this)
        Status.Code.FAILED_PRECONDITION -> VesselException.FailedPrecondition(description, this)
        Status.Code.UNIMPLEMENTED -> VesselException.Unimplemented(description, this)
        Status.Code.DEADLINE_EXCEEDED, Status.Code.ABORTED, Status.Code.RESOURCE_EXHAUSTED ->
            VesselException.Other("${status.code}: $description", isRetryable = true, cause = this)
        else -> VesselException.Other("${status.code}: $description", isRetryable = false, cause = this)
    }
}

internal fun Throwable.isRetryable(): Boolean = (toVesselException() as? VesselException)?.isRetryable == true
