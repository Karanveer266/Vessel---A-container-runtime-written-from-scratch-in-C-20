package dev.vessel.console.api

import dev.vessel.console.model.VesselException
import io.grpc.Status
import kotlinx.coroutines.CancellationException
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class ErrorsTest {
    @Test
    fun `maps status codes to domain errors with retryability`() {
        val unavailable = Status.UNAVAILABLE.withDescription("conn refused").asException().toVesselException()
        assertTrue(unavailable is VesselException.Unavailable && unavailable.isRetryable)

        val auth = Status.UNAUTHENTICATED.asRuntimeException().toVesselException()
        assertTrue(auth is VesselException.Unauthenticated)
        assertFalse((auth as VesselException).isRetryable)

        assertTrue(Status.NOT_FOUND.asException().toVesselException() is VesselException.NotFound)
        assertTrue((Status.DEADLINE_EXCEEDED.asException().toVesselException() as VesselException).isRetryable)
        assertFalse((Status.INTERNAL.asException().toVesselException() as VesselException).isRetryable)
    }

    @Test
    fun `cancellation is never wrapped`() {
        val c = CancellationException("scope gone")
        assertSame(c, c.toVesselException())
    }
}
