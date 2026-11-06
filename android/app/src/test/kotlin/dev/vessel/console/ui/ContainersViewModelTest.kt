package dev.vessel.console.ui

import dev.vessel.console.model.ContainerAction
import dev.vessel.console.model.ContainerState
import dev.vessel.console.model.VesselException
import dev.vessel.console.model.flow.StreamEvent
import dev.vessel.console.testing.FakeVesselRepository
import dev.vessel.console.testing.MainDispatcherRule
import dev.vessel.console.testing.container
import dev.vessel.console.testing.item
import dev.vessel.console.testing.stats
import dev.vessel.console.testing.subscribe
import dev.vessel.console.ui.common.Load
import dev.vessel.console.ui.common.StreamStatus
import dev.vessel.console.ui.containers.ContainersViewModel
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
class ContainersViewModelTest {
    @get:Rule val main = MainDispatcherRule()

    private val repo = FakeVesselRepository().apply {
        containers += container("stopped1", state = ContainerState.Stopped, createdAt = 3)
        containers += container("run-old", state = ContainerState.Running, createdAt = 1)
        containers += container("run-new", state = ContainerState.Running, createdAt = 2)
    }

    private fun rows(vm: ContainersViewModel) = (vm.state.value.rows as Load.Ready).value

    @Test
    fun `lists running containers first, newest first`() = runTest {
        val vm = ContainersViewModel(repo)
        subscribe(vm.state)
        runCurrent()
        assertEquals(listOf("run-new", "run-old", "stopped1"), rows(vm).map { it.container.id })
    }

    @Test
    fun `multiplexed stats stream feeds bounded per-row sparklines`() = runTest {
        val vm = ContainersViewModel(repo, sparkCapacity = 3)
        subscribe(vm.state)
        runCurrent()
        assertEquals(listOf(""), repo.statsRefs) // ONE stream for all containers
        repeat(5) { i ->
            repo.stats.emit(item(stats("run-new", cpu = i * 10)))
            repo.stats.emit(item(stats("run-old", cpu = 1)))
        }
        advanceTimeBy(600) // one batch period
        val sparks = rows(vm).first { it.container.id == "run-new" }.sparks
        assertEquals(3, sparks.cpuMillicores.size)
        assertEquals(listOf(20f, 30f, 40f), (0 until 3).map { sparks.cpuMillicores[it] })
        assertEquals(40, sparks.latest?.cpuMillicoresUsed)
        assertEquals(StreamStatus.Live, vm.state.value.statsStatus)
    }

    @Test
    fun `reconnect notices surface as stream status`() = runTest {
        val vm = ContainersViewModel(repo)
        subscribe(vm.state)
        runCurrent()
        repo.stats.emit(StreamEvent.Reconnecting(attempt = 2, delayMs = 1000, cause = IOException("wifi")))
        advanceTimeBy(600)
        assertEquals(2, (vm.state.value.statsStatus as StreamStatus.Reconnecting).attempt)
    }

    @Test
    fun `stream and polling stop after the UI goes away, with a grace period`() = runTest {
        val vm = ContainersViewModel(repo, pollIntervalMs = 1_000)
        val ui = subscribe(vm.state)
        runCurrent()
        assertEquals(1, repo.statsSubscribers)

        ui.cancel() // app backgrounded
        advanceTimeBy(4_000)
        assertEquals("still subscribed during grace period", 1, repo.statsSubscribers)
        advanceTimeBy(1_500)
        assertEquals(0, repo.statsSubscribers)

        val callsWhenStopped = repo.listCalls
        advanceTimeBy(10_000)
        assertEquals("no polling in background", callsWhenStopped, repo.listCalls)
    }

    @Test
    fun `polls the list while visible`() = runTest {
        val vm = ContainersViewModel(repo, pollIntervalMs = 1_000)
        subscribe(vm.state)
        runCurrent()
        val before = repo.listCalls
        advanceTimeBy(3_100)
        assertTrue(repo.listCalls >= before + 3)
    }

    @Test
    fun `actions update the row and refresh`() = runTest {
        val vm = ContainersViewModel(repo)
        subscribe(vm.state)
        runCurrent()
        val target = rows(vm).first { it.container.id == "run-new" }.container
        vm.perform(target, ContainerAction.Pause)
        runCurrent()
        assertEquals(listOf("pause" to "run-new"), repo.actions)
        assertEquals(ContainerState.Paused, rows(vm).first { it.container.id == "run-new" }.container.state)
        assertNull(rows(vm).first { it.container.id == "run-new" }.pending)
    }

    @Test
    fun `removing a running container forces it and drops the row`() = runTest {
        val vm = ContainersViewModel(repo)
        subscribe(vm.state)
        runCurrent()
        vm.perform(rows(vm).first { it.container.id == "run-old" }.container, ContainerAction.Remove)
        runCurrent()
        assertEquals("remove(force=true)" to "run-old", repo.actions.single())
        assertTrue(rows(vm).none { it.container.id == "run-old" })
    }

    @Test
    fun `failed action keeps state, clears pending and reports`() = runTest {
        repo.actionError = VesselException.FailedPrecondition("not running")
        val vm = ContainersViewModel(repo)
        subscribe(vm.state)
        runCurrent()
        vm.perform(rows(vm).first { it.container.id == "stopped1" }.container, ContainerAction.Start)
        runCurrent()
        assertEquals(ContainerState.Stopped, rows(vm).first { it.container.id == "stopped1" }.container.state)
        assertTrue(vm.state.value.message!!.startsWith("Start failed"))
        vm.messageShown()
        runCurrent()
        assertNull(vm.state.value.message)
    }

    @Test
    fun `background refresh failure keeps stale rows`() = runTest {
        val vm = ContainersViewModel(repo, pollIntervalMs = 1_000)
        subscribe(vm.state)
        runCurrent()
        repo.listError = VesselException.Unavailable("down")
        advanceTimeBy(1_100)
        assertEquals(3, rows(vm).size)
    }
}
