package dev.vessel.console.api

import io.grpc.CallOptions
import io.grpc.Channel
import io.grpc.ClientCall
import io.grpc.ClientInterceptor
import io.grpc.ForwardingClientCall.SimpleForwardingClientCall
import io.grpc.Metadata
import io.grpc.MethodDescriptor

/**
 * Attaches `authorization: Bearer <token>` to every call, as vesseld expects.
 *
 * The token is read through [tokenProvider] at call start rather than captured
 * once, so the interceptor stays correct even if a caller wires it to a
 * mutable source. (VesselClient rebuilds the channel on any settings change
 * anyway, which also restarts long-lived streams with the new credentials.)
 *
 * A blank token sends no header at all: a daemon running without auth (e.g.
 * bound to localhost on the same device) should not see a malformed
 * `Bearer ` value.
 */
class AuthInterceptor(private val tokenProvider: () -> String) : ClientInterceptor {

    constructor(token: String) : this({ token })

    override fun <ReqT, RespT> interceptCall(
        method: MethodDescriptor<ReqT, RespT>,
        callOptions: CallOptions,
        next: Channel,
    ): ClientCall<ReqT, RespT> = object : SimpleForwardingClientCall<ReqT, RespT>(next.newCall(method, callOptions)) {
        override fun start(responseListener: Listener<RespT>, headers: Metadata) {
            val token = tokenProvider().trim()
            if (token.isNotEmpty()) headers.put(AUTHORIZATION, "Bearer $token")
            super.start(responseListener, headers)
        }
    }

    companion object {
        val AUTHORIZATION: Metadata.Key<String> =
            Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER)
    }
}
