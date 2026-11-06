package dev.vessel.console.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class StatsHistoryTest {

    @Test
    fun `keeps a bounded window and the latest sample`() {
        val h = StatsHistory(capacity = 3)
        (1..5).forEach { h.add(sample(cpu = it * 100, mem = it * 1_000L)) }
        val s = h.snapshot()
        assertEquals(3, s.cpuMillicores.size)
        assertEquals(300f, s.cpuMillicores[0])
        assertEquals(500f, s.cpuMillicores.lastOrNull())
        assertEquals(5_000f, s.memoryCurrentBytes.lastOrNull())
        assertEquals(500, s.latest?.cpuMillicoresUsed)
    }

    @Test
    fun `snapshots are immutable and equal by content`() {
        val h = StatsHistory(capacity = 4)
        h.add(sample(cpu = 1))
        val a = h.snapshot()
        assertEquals(a, h.snapshot())
        h.add(sample(cpu = 2))
        assertEquals(1, a.cpuMillicores.size)
        assertFalse(a == h.snapshot())
        h.clear()
        assertNull(h.snapshot().latest)
    }
}

class TraceFilterTest {

    private fun ev(name: String, ret: Long = 0, latency: Long = 0, kind: TraceEventKind = TraceEventKind.Syscall) =
        TraceEvent(0, "c", kind, 0, 1, 1, "sh", 0, name, ret, latency, 0, false, "", 0, TraceBackend.Ebpf)

    @Test
    fun `combines all criteria`() {
        val f = TraceFilter("c", errorsOnly = true, minLatencyNs = 1_000, syscalls = listOf("openat"))
        assertTrue(f.matches(ev("openat", ret = -2, latency = 5_000)))
        assertFalse(f.matches(ev("openat", ret = 3, latency = 5_000)))
        assertFalse(f.matches(ev("openat", ret = -2, latency = 10)))
        assertFalse(f.matches(ev("read", ret = -2, latency = 5_000)))
    }

    @Test
    fun `seccomp denials count as errors`() {
        assertTrue(TraceFilter("c", errorsOnly = true).matches(ev("ptrace", kind = TraceEventKind.SeccompDenied)))
    }
}

class ConnectionSettingsTest {
    @Test
    fun `validates and flags clear-text tokens to remote hosts`() {
        assertTrue(ConnectionSettings().validate().isEmpty())
        assertEquals(2, ConnectionSettings(host = " ", port = 0).validate().size)
        assertFalse(ConnectionSettings(host = "127.0.0.1", token = "t").isInsecureRemote)
        assertTrue(ConnectionSettings(host = "192.168.1.5", token = "t").isInsecureRemote)
        assertFalse(ConnectionSettings(host = "192.168.1.5", token = "t", useTls = true).isInsecureRemote)
    }

    @Test
    fun `state machine exposes sensible actions`() {
        assertEquals(setOf(ContainerAction.Stop, ContainerAction.Pause, ContainerAction.Remove), ContainerState.Running.allowedActions)
        assertTrue(ContainerAction.Resume in ContainerState.Paused.allowedActions)
        assertFalse(ContainerAction.Stop in ContainerState.Stopped.allowedActions)
    }
}

internal fun sample(cpu: Int = 0, mem: Long = 0, id: String = "c1") = StatsSample(
    containerId = id, timestampUnixNs = 0, cpuUsageUsec = 0, cpuNrThrottled = 0, cpuThrottledUsec = 0,
    cpuMillicoresUsed = cpu, memoryCurrentBytes = mem, memoryMaxBytes = 0, memoryPeakBytes = 0,
    memoryEventsHigh = 0, memoryEventsMax = 0, memoryEventsOom = 0, memoryEventsOomKill = 0,
    pidsCurrent = 1, pidsMax = 0, cpuPressure = Psi.Zero, memoryPressure = Psi.Zero, ioPressure = Psi.Zero,
    netRxBytes = 0, netTxBytes = 0, droppedSamples = 0,
)
