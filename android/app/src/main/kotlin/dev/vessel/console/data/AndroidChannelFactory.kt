package dev.vessel.console.data

import android.content.Context
import dev.vessel.console.api.ChannelFactory
import dev.vessel.console.api.OkHttpChannelFactory
import dev.vessel.console.model.Endpoint
import io.grpc.ManagedChannel
import io.grpc.android.AndroidChannelBuilder

/**
 * Decorates the OkHttp channel with grpc-android's network monitor: on a
 * network switch (Wi-Fi -> cellular, Wi-Fi reconnect) the channel drops dead
 * connections and resets its connect backoff immediately instead of waiting
 * out a TCP timeout. Our stream-level retry then reconnects in ~500 ms.
 */
class AndroidChannelFactory(
    context: Context,
    private val okHttp: OkHttpChannelFactory = OkHttpChannelFactory(),
) : ChannelFactory {
    private val appContext = context.applicationContext

    override fun create(endpoint: Endpoint): ManagedChannel =
        AndroidChannelBuilder.usingBuilder(okHttp.builder(endpoint))
            .context(appContext)
            .build()
}
