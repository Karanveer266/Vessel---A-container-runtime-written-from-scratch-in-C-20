package dev.vessel.console.ui

import dev.vessel.console.model.ContainerAction
import dev.vessel.console.model.ContainerState
import dev.vessel.console.model.ExecResult
import dev.vessel.console.model.LogStream
import dev.vessel.console.model.SyscallCount
import dev.vessel.console.model.SyscallSummary
import dev.vessel.console.model.TraceBackend
import dev.vessel.console.model.flow.StreamEvent
import dev.vessel.console.testing.FakeVesselRepository
import dev.vessel.console.testing.MainDispatcherRule
import dev.vessel.console.testing.container
import dev.vessel.console.testing.item
import dev.vessel.console.testing.logChunk
import dev.vessel.console.testing.stats
import dev.vessel.console.testing.subscribe
import dev.vessel.console.ui.common.Load
import dev.vessel.console.ui.common.StreamStatus
import dev.vessel.console.ui.detail.ContainerHeaderViewModel
import dev.vessel.console.ui.detail.ExecViewModel
import dev.vessel.console.ui.detail.LiveStatsViewModel
import dev.vessel.console.ui.detail.LogsViewModel
import dev.vessel.console.ui.detail.SyscallSummaryViewModel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import java.io.IOException

@OptIn(ExperimentalCoroutinesApi::class)
class LiveStatsViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    private val repo = FakeVesselRepository()

    @Test
    fun `builds a bounded chart window from the stream`() = runTest {
        val vm = LiveStatsViewModel("c1", repo, capacity = 4)
        subscribe(vm.state)
        runCurrent()
        assertEquals(listOf("c1"), repo.statsRefs)
        (1..10).forEach { repo.stats.emit(item(stats(cpu = it * 100, mem = it * 1_000L))) }
        runCurrent()
        val s = vm.state.value.stats
        assertEquals(4, s.cpuMillicores.size)
        assertEquals(700f, s.cpuMillicores[0])
        assertEquals(1_000f, s.cpuMillicores.lastOrNull())
        assertEquals(StreamStatus.Live, vm.state.value.status)
    }

    @Test
    fun `reconnect keeps history and shows status`() = runTest {
        val vm = LiveStatsViewModel("c1", repo)
        subscribe(vm.state)
        runCurrent()
        repo.stats.emit(item(stats(cpu = 5, oomKill = 1)))
        repo.stats.emit(StreamEvent.Reconnecting(1, 500, IOException("x")))
        runCurrent()
        assertEquals(1, vm.state.value.stats.cpuMillicores.size)
        assertTrue(vm.state.value.stats.latest!!.hasOomEvents)
        assertTrue(vm.state.value.status is StreamStatus.Reconnecting)
    }

    @Test
    fun `stops the stream in background`() = runTest {
        val vm = LiveStatsViewModel("c1", repo)
        val ui = subscribe(vm.state)
        runCurrent()
        assertEquals(1, repo.statsSubscribers)
        ui.cancel()
        advanceTimeBy(5_001)
        runCurrent()
        assertEquals(0, repo.statsSubscribers)
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class LogsViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    private val repo = FakeVesselRepository()

    @Test
    fun `assembles lines per stream across chunk boundaries`() = runTest {
        val vm = LogsViewModel("c1", repo)
        subscribe(vm.state)
        runCurrent()
        repo.logs.emit(item(logChunk("hel")))
        repo.logs.emit(item(logChunk("boom\n", LogStream.Stderr)))
        repo.logs.emit(item(logChunk("lo\nwor")))
        advanceTimeBy(250)
        assertEquals(listOf("boom" to LogStream.Stderr, "hello" to LogStream.Stdout), vm.state.value.lines.map { it.text to it.stream })
        repo.logs.emit(item(logChunk("ld")))
        repo.endLogs()
        advanceTimeBy(250)
        runCurrent()
        assertEquals("world", vm.state.value.lines.last().text)
        assertTrue(vm.state.value.ended)
    }

    @Test
    fun `reconnect clears buffer since the daemon replays the tail`() = runTest {
        val vm = LogsViewModel("c1", repo)
        subscribe(vm.state)
        runCurrent()
        repo.logs.emit(item(logChunk("a\nb\n")))
        advanceTimeBy(250)
        repo.logs.emit(StreamEvent.Reconnecting(1, 10, IOException()))
        repo.logs.emit(item(logChunk("a\nb\nc\n")))
        advanceTimeBy(250)
        assertEquals(listOf("a", "b", "c"), vm.state.value.lines.map { it.text })
    }

    @Test
    fun `keeps at most capacity lines`() = runTest {
        val vm = LogsViewModel("c1", repo, capacity = 10)
        subscribe(vm.state)
        runCurrent()
        repo.logs.emit(item(logChunk((1..100).joinToString("") { "$it\n" })))
        advanceTimeBy(250)
        assertEquals((91..100).map { "$it" }, vm.state.value.lines.map { it.text })
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class HeaderExecSummaryViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    private val repo = FakeVesselRepository().apply { containers += container("c1", state = ContainerState.Running) }

    @Test
    fun `header loads container and applies actions`() = runTest {
        val vm = ContainerHeaderViewModel("c1", repo)
        runCurrent()
        assertEquals(ContainerState.Running, vm.state.value.container.valueOrNull?.state)
        vm.perform(ContainerAction.Stop)
        runCurrent()
        assertEquals(ContainerState.Stopped, vm.state.value.container.valueOrNull?.state)
        assertNull(vm.state.value.pending)
        vm.perform(ContainerAction.Remove)
        runCurrent()
        assertTrue(vm.state.value.removed)
    }

    @Test
    fun `header reports missing container`() = runTest {
        val vm = ContainerHeaderViewModel("nope", repo)
        runCurrent()
        assertTrue(vm.state.value.container is Load.Failed)
    }

    @Test
    fun `exec splits argv and shows the result`() = runTest {
        repo.execResult = ExecResult(2, "", "no such file\n", false)
        var now = 1_000L
        val vm = ExecViewModel("c1", repo, clock = { now.also { now += 42 } })
        vm.onCommandChange("sh -c 'ls /missing'")
        vm.run()
        runCurrent()
        assertEquals(listOf("sh", "-c", "ls /missing"), repo.execArgs)
        assertEquals(2, vm.state.value.result?.exitCode)
        assertEquals(42L, vm.state.value.durationMs)
    }

    @Test
    fun `exec rejects malformed command without calling the daemon`() = runTest {
        val vm = ExecViewModel("c1", repo)
        vm.onCommandChange("echo 'oops")
        vm.run()
        runCurrent()
        assertNull(repo.execArgs)
        assertTrue(vm.state.value.error!!.contains("quote"))
    }

    @Test
    fun `summary polls while visible and on manual refresh`() = runTest {
        repo.summary = SyscallSummary("c1", listOf(SyscallCount("read", 0, 10, 0, 100, 50)), 10, 0, TraceBackend.Ebpf)
        val vm = SyscallSummaryViewModel("c1", repo, pollIntervalMs = 1_000)
        val ui = subscribe(vm.state)
        runCurrent()
        assertEquals("read", vm.state.value.summary.valueOrNull?.syscalls?.single()?.name)
        assertEquals(1, repo.summaryCalls)
        vm.refresh()
        runCurrent()
        assertEquals(2, repo.summaryCalls)
        advanceTimeBy(1_001)
        assertEquals(3, repo.summaryCalls)
        ui.cancel()
        advanceTimeBy(5_500) // sharing grace period
        val afterGrace = repo.summaryCalls
        advanceTimeBy(20_000)
        assertEquals("no polling in background", afterGrace, repo.summaryCalls)
    }
}
