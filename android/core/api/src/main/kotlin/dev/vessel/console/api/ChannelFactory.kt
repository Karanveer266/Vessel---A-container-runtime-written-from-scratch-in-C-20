package dev.vessel.console.api

import dev.vessel.console.model.Endpoint
import io.grpc.ManagedChannel
import io.grpc.okhttp.OkHttpChannelBuilder
import java.util.concurrent.TimeUnit

/**
 * Seam for transport construction. Production uses OkHttp (the recommended
 * gRPC transport on Android: small, no Netty, uses the platform TLS stack);
 * :app decorates it with network-change awareness; tests substitute an
 * in-process channel.
 */
fun interface ChannelFactory {
    fun create(endpoint: Endpoint): ManagedChannel
}

/**
 * OkHttp channel tuned for a phone talking to a daemon over Wi-Fi / USB.
 *
 * Keepalive: a phone that walks out of Wi-Fi range leaves TCP connections
 * half-open for minutes; HTTP/2 PINGs detect that within ~[keepAliveSec] +
 * [keepAliveTimeoutSec] so the stream fails fast and the reconnect logic takes
 * over. Pings are sent only while calls are active (`keepAliveWithoutCalls =
 * false`) to spare the radio/battery. grpc-core servers by default tolerate
 * only one data-less ping per 5 min; vesseld lowers that to 10 s
 * (GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS in src/daemon/main.cpp)
 * so an idle trace stream is never answered with GOAWAY(too_many_pings).
 * Against a third-party server that keeps the default, that GOAWAY surfaces
 * as UNAVAILABLE and the normal reconnect path handles it.
 */
class OkHttpChannelFactory(
    private val keepAliveSec: Long = 60,
    private val keepAliveTimeoutSec: Long = 20,
    private val idleTimeoutSec: Long = 300,
    private val maxInboundMessageBytes: Int = 16 * 1024 * 1024,
) : ChannelFactory {

    override fun create(endpoint: Endpoint): ManagedChannel = builder(endpoint).build()

    /** Exposed so platform layers can wrap the configured builder (e.g. AndroidChannelBuilder). */
    fun builder(endpoint: Endpoint): OkHttpChannelBuilder =
        OkHttpChannelBuilder.forAddress(endpoint.host, endpoint.port).apply {
            if (endpoint.useTls) useTransportSecurity() else usePlaintext()
            keepAliveTime(keepAliveSec, TimeUnit.SECONDS)
            keepAliveTimeout(keepAliveTimeoutSec, TimeUnit.SECONDS)
            keepAliveWithoutCalls(false)
            idleTimeout(idleTimeoutSec, TimeUnit.SECONDS)
            maxInboundMessageSize(maxInboundMessageBytes)
            userAgent("vessel-console")
        }
}
