package dev.vessel.console.api

import dev.vessel.api.v1.Container as PbContainer
import dev.vessel.api.v1.ContainerRef
import dev.vessel.api.v1.ContainerState as PbContainerState
import dev.vessel.api.v1.GetHostInfoRequest
import dev.vessel.api.v1.HostInfo as PbHostInfo
import dev.vessel.api.v1.ListContainersRequest
import dev.vessel.api.v1.ListContainersResponse
import dev.vessel.api.v1.StatsSample as PbStatsSample
import dev.vessel.api.v1.StreamStatsRequest
import dev.vessel.api.v1.StreamTraceRequest
import dev.vessel.api.v1.TraceEvent as PbTraceEvent
import dev.vessel.api.v1.VesselGrpcKt.VesselCoroutineImplBase
import dev.vessel.api.v1.container
import dev.vessel.api.v1.hostInfo
import dev.vessel.api.v1.listContainersResponse
import dev.vessel.api.v1.statsSample
import dev.vessel.api.v1.traceEvent
import dev.vessel.console.model.ConnectionSettings
import dev.vessel.console.model.ContainerState
import dev.vessel.console.model.TraceFilter
import dev.vessel.console.model.VesselException
import dev.vessel.console.model.flow.BackoffPolicy
import dev.vessel.console.model.flow.StreamEvent
import dev.vessel.console.model.flow.items
import io.grpc.Metadata
import io.grpc.Server
import io.grpc.ServerCall
import io.grpc.ServerCallHandler
import io.grpc.ServerInterceptor
import io.grpc.ServerInterceptors
import io.grpc.Status
import io.grpc.StatusException
import io.grpc.inprocess.InProcessChannelBuilder
import io.grpc.inprocess.InProcessServerBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertFailsWith

/**
 * End-to-end tests of [VesselClient] against a real gRPC server running
 * in-process: real stubs, real interceptors, real status propagation — only
 * the socket is replaced. Each "host" in [ConnectionSettings] resolves to an
 * in-process server of the same name, which lets us test reconnection to a
 * different daemon when settings change.
 */
class VesselClientInProcessTest {

    private val servers = mutableListOf<Server>()
    private val seenAuth = ConcurrentLinkedQueue<String>()
    private lateinit var channels: ChannelManager

    private val fastRetry = BackoffPolicy(initialDelayMs = 5, maxDelayMs = 20, jitter = 0.0, maxAttempts = 2)
    private val fastStreamRetry = BackoffPolicy(initialDelayMs = 5, maxDelayMs = 20, jitter = 0.0)

    /** Records the authorization header of every incoming call. */
    private val authRecorder = object : ServerInterceptor {
        override fun <ReqT, RespT> interceptCall(
            call: ServerCall<ReqT, RespT>,
            headers: Metadata,
            next: ServerCallHandler<ReqT, RespT>,
        ): ServerCall.Listener<ReqT> {
            seenAuth += headers.get(AuthInterceptor.AUTHORIZATION) ?: "<none>"
            return next.startCall(call, headers)
        }
    }

    private fun serve(name: String, impl: VesselCoroutineImplBase) {
        servers += InProcessServerBuilder.forName(name)
            .directExecutor()
            .addService(ServerInterceptors.intercept(impl, authRecorder))
            .build()
            .start()
    }

    private fun client(settings: Flow<ConnectionSettings>) = VesselClient(
        settings = settings,
        channels = channels,
        unaryDeadlineMs = 5_000,
        readPolicy = fastRetry,
        streamPolicy = fastStreamRetry,
        ioDispatcher = Dispatchers.IO,
    )

    private fun settings(host: String = "daemon-a", token: String = "s3cret") =
        MutableStateFlow(ConnectionSettings(host = host, port = 50051, token = token))

    @Before
    fun setUp() {
        channels = ChannelManager { endpoint -> InProcessChannelBuilder.forName(endpoint.host).directExecutor().build() }
    }

    @After
    fun tearDown() {
        channels.close()
        servers.forEach { it.shutdownNow().awaitTermination() }
    }

    @Test
    fun `unary call maps response and sends bearer token`() = runBlocking {
        serve("daemon-a", object : VesselCoroutineImplBase() {
            override suspend fun getHostInfo(request: GetHostInfoRequest): PbHostInfo =
                hostInfo { kernelRelease = "6.6.30-android15"; isAndroid = true; onlineCpus = 8 }
        })

        val info = client(settings()).hostInfo()

        assertEquals("6.6.30-android15", info.kernelRelease)
        assertTrue(info.isAndroid)
        assertEquals(listOf("Bearer s3cret"), seenAuth.toList())
    }

    @Test
    fun `blank token sends no authorization header`() = runBlocking {
        serve("daemon-a", object : VesselCoroutineImplBase() {
            override suspend fun getHostInfo(request: GetHostInfoRequest): PbHostInfo = hostInfo { }
        })
        client(settings(token = "  ")).hostInfo()
        assertEquals(listOf("<none>"), seenAuth.toList())
    }

    @Test
    fun `unauthenticated is surfaced as a domain error and not retried`() = runBlocking {
        val calls = AtomicInteger()
        serve("daemon-a", object : VesselCoroutineImplBase() {
            override suspend fun getHostInfo(request: GetHostInfoRequest): PbHostInfo {
                calls.incrementAndGet()
                throw StatusException(Status.UNAUTHENTICATED.withDescription("bad token"))
            }
        })
        val e = assertFailsWith<VesselException.Unauthenticated> { client(settings()).hostInfo() }
        assertEquals("bad token", e.message)
        assertEquals(1, calls.get())
    }

    @Test
    fun `idempotent reads retry transient failures`() = runBlocking {
        val calls = AtomicInteger()
        serve("daemon-a", object : VesselCoroutineImplBase() {
            override suspend fun listContainers(request: ListContainersRequest): ListContainersResponse {
                if (calls.incrementAndGet() == 1) throw StatusException(Status.UNAVAILABLE)
                return listContainersResponse {
                    containers += container { id = "aaa"; state = PbContainerState.CONTAINER_STATE_RUNNING }
                }
            }
        })
        val list = client(settings()).listContainers()
        assertEquals(2, calls.get())
        assertEquals(ContainerState.Running, list.single().state)
    }

    @Test
    fun `mutations are never retried automatically`() = runBlocking {
        val calls = AtomicInteger()
        serve("daemon-a", object : VesselCoroutineImplBase() {
            override suspend fun startContainer(request: ContainerRef): PbContainer {
                calls.incrementAndGet()
                throw StatusException(Status.UNAVAILABLE)
            }
        })
        assertFailsWith<VesselException.Unavailable> { client(settings()).startContainer("aaa") }
        assertEquals(1, calls.get())
    }

    @Test
    fun `stats stream reconnects after the server drops it`() = runBlocking {
        val subscriptions = AtomicInteger()
        serve("daemon-a", object : VesselCoroutineImplBase() {
            override fun streamStats(request: StreamStatsRequest): Flow<PbStatsSample> = flow {
                val n = subscriptions.incrementAndGet()
                emit(statsSample { containerId = request.id; cpuMillicoresUsed = n * 100 })
                if (n == 1) throw StatusException(Status.UNAVAILABLE.withDescription("daemon restarting"))
                emit(statsSample { containerId = request.id; cpuMillicoresUsed = n * 100 + 1 })
            }
        })

        val events = withTimeout(5_000) { client(settings()).streamStats("c1").take(4).toList() }

        assertEquals(2, subscriptions.get())
        val reconnect = events[1] as StreamEvent.Reconnecting
        assertEquals(1, reconnect.attempt)
        assertTrue(reconnect.cause is VesselException.Unavailable)
        assertEquals(
            listOf(100, 200, 201),
            events.filterIsInstance<StreamEvent.Item<*>>().map { (it.value as dev.vessel.console.model.StatsSample).cpuMillicoresUsed },
        )
    }

    @Test
    fun `trace stream assigns monotonically increasing keys across reconnects`() = runBlocking {
        val subscriptions = AtomicInteger()
        serve("daemon-a", object : VesselCoroutineImplBase() {
            override fun streamTrace(request: StreamTraceRequest): Flow<PbTraceEvent> = flow {
                val n = subscriptions.incrementAndGet()
                emit(traceEvent { syscallName = "read"; detail = "sub$n" })
                emit(traceEvent { syscallName = "write"; detail = "sub$n"; droppedEvents = n.toLong() })
                if (n == 1) throw StatusException(Status.UNAVAILABLE)
            }
        })

        val items = withTimeout(5_000) { client(settings()).streamTrace(TraceFilter("c1")).items().take(4).toList() }

        assertEquals(listOf(0L, 1L, 2L, 3L), items.map { it.seq })
        assertEquals(listOf("sub1", "sub1", "sub2", "sub2"), items.map { it.detail })
    }

    @Test
    fun `trace keys stay unique across separate subscriptions`() = runBlocking {
        serve("daemon-a", object : VesselCoroutineImplBase() {
            override fun streamTrace(request: StreamTraceRequest): Flow<PbTraceEvent> = flow {
                repeat(3) { emit(traceEvent { syscallName = "read" }) }
            }
        })
        val client = client(settings())
        val first = withTimeout(5_000) { client.streamTrace(TraceFilter("c1")).items().take(3).toList() }
        val second = withTimeout(5_000) { client.streamTrace(TraceFilter("c1", errorsOnly = true)).items().take(3).toList() }
        assertEquals(6, (first + second).map { it.seq }.toSet().size)
        assertTrue(second.first().seq > first.last().seq)
    }

    @Test
    fun `changing settings resubscribes the stream against the new daemon with new credentials`() = runBlocking {
        fun daemon(tag: Int) = object : VesselCoroutineImplBase() {
            override fun streamStats(request: StreamStatsRequest): Flow<PbStatsSample> = flow {
                while (true) {
                    emit(statsSample { cpuMillicoresUsed = tag })
                    kotlinx.coroutines.delay(10)
                }
            }
        }
        serve("daemon-a", daemon(1))
        serve("daemon-b", daemon(2))
        val settings = settings(host = "daemon-a", token = "token-a")
        val stream = client(settings).streamStats("").items()

        val seen = withTimeout(5_000) {
            val out = mutableListOf<Int>()
            stream.first { sample ->
                out += sample.cpuMillicoresUsed
                if (out.size == 2) settings.value = settings.value.copy(host = "daemon-b", token = "token-b")
                sample.cpuMillicoresUsed == 2
            }
            out
        }

        assertEquals(1, seen.first())
        assertEquals(2, seen.last())
        assertTrue("Bearer token-a" in seenAuth)
        assertEquals("Bearer token-b", seenAuth.last())
    }

}
