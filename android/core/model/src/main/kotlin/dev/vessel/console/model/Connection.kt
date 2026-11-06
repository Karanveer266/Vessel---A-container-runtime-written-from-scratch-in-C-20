package dev.vessel.console.model

/**
 * Where and how to reach `vesseld`. Persisted by the app (DataStore) and read
 * by the API client on every (re)connect, so edits take effect without a restart.
 */
data class ConnectionSettings(
    val host: String = DEFAULT_HOST,
    val port: Int = DEFAULT_PORT,
    val token: String = "",
    val useTls: Boolean = false,
) {
    val endpoint: Endpoint get() = Endpoint(host.trim(), port, useTls)

    /** Bearer token sent in clear text to a non-loopback host is worth a warning. */
    val isInsecureRemote: Boolean
        get() = !useTls && token.isNotEmpty() && !endpoint.isLoopback

    fun validate(): List<String> = buildList {
        if (host.isBlank()) add("Host must not be empty")
        if (host.trim().any { it.isWhitespace() }) add("Host must not contain spaces")
        if (port !in 1..65535) add("Port must be in 1..65535")
    }

    companion object {
        const val DEFAULT_HOST = "127.0.0.1"
        const val DEFAULT_PORT = 50051
    }
}

/** The part of the settings that identifies a transport; the token is per-call metadata. */
data class Endpoint(val host: String, val port: Int, val useTls: Boolean) {
    val isLoopback: Boolean
        get() = host == "localhost" || host == "::1" || host.startsWith("127.")

    override fun toString(): String = "${if (useTls) "https" else "http"}://$host:$port"
}
