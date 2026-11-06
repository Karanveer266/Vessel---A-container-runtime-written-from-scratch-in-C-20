package dev.vessel.console.api

import dev.vessel.console.model.ConnectionSettings
import io.grpc.Channel
import io.grpc.ClientInterceptors
import io.grpc.ManagedChannel
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.Closeable
import java.util.concurrent.TimeUnit

/**
 * Owns at most one live [ManagedChannel], keyed by the full [ConnectionSettings].
 *
 * gRPC channels are expensive (TCP + TLS + HTTP/2 handshake) and designed to be
 * long-lived and shared, so every RPC reuses the cached one. When the settings
 * change (host, port, TLS *or* token), a new channel is built and the old one
 * is shut down gracefully: in-flight unary calls are allowed to finish, while
 * long-lived streams are restarted by VesselClient anyway.
 */
class ChannelManager(private val factory: ChannelFactory) : Closeable {
    private val lock = Mutex()
    private var current: Entry? = null

    private class Entry(val settings: ConnectionSettings, val managed: ManagedChannel, val channel: Channel)

    suspend fun channelFor(settings: ConnectionSettings): Channel = lock.withLock {
        current?.takeIf { it.settings == settings }?.let { return it.channel }
        current?.managed?.shutdown()
        val managed = factory.create(settings.endpoint)
        val channel = ClientInterceptors.intercept(managed, AuthInterceptor(settings.token))
        current = Entry(settings, managed, channel)
        channel
    }

    /** Fast-fails pending connection attempts (e.g. the user hit "Reconnect"). */
    fun resetConnectBackoff() {
        current?.managed?.resetConnectBackoff()
    }

    override fun close() {
        val entry = current ?: return
        current = null
        entry.managed.shutdown()
        if (!entry.managed.awaitTermination(1, TimeUnit.SECONDS)) entry.managed.shutdownNow()
    }
}
