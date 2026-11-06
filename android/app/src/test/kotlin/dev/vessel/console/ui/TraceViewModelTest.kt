package dev.vessel.console.ui

import dev.vessel.console.model.flow.StreamEvent
import dev.vessel.console.testing.FakeVesselRepository
import dev.vessel.console.testing.MainDispatcherRule
import dev.vessel.console.testing.item
import dev.vessel.console.testing.subscribe
import dev.vessel.console.testing.traceEvent
import dev.vessel.console.ui.detail.TraceViewModel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class TraceViewModelTest {
    @get:Rule val main = MainDispatcherRule()

    private val repo = FakeVesselRepository()

    private fun TestScope.started(capacity: Int = 2_000): TraceViewModel {
        val vm = TraceViewModel("c1", repo, capacity = capacity, batchMs = 100, filterDebounceMs = 300)
        subscribe(vm.state)
        advanceTimeBy(301) // initial debounce
        runCurrent()
        return vm
    }

    @Test
    fun `buffer is bounded and newest first under a flood of events`() = runTest {
        val vm = started(capacity = 2_000)
        for (i in 0 until 50_000L) repo.trace.emit(item(traceEvent(i)))
        advanceTimeBy(150)
        val s = vm.state.value
        assertEquals(2_000, s.events.size)
        assertEquals(49_999L, s.events.first().seq)
        assertEquals(48_000L, s.events.last().seq)
        // Keys must be unique for LazyColumn.
        assertEquals(s.events.size, s.events.map { it.seq }.toSet().size)
    }

    @Test
    fun `accumulates dropped_events reported by the daemon`() = runTest {
        val vm = started()
        repo.trace.emit(item(traceEvent(0, dropped = 5)))
        repo.trace.emit(item(traceEvent(1, dropped = 7)))
        advanceTimeBy(150)
        assertEquals(12L, vm.state.value.droppedUpstream)
        assertEquals(2L, vm.state.value.received)
    }

    @Test
    fun `filter edits are debounced into a single new server subscription`() = runTest {
        val vm = started()
        assertEquals(1, repo.traceFilters.size)
        "openat".forEachIndexed { i, _ ->
            vm.onFilterChange { it.copy(syscalls = "openat".take(i + 1)) }
            advanceTimeBy(50)
        }
        vm.onFilterChange { it.copy(errorsOnly = true, minLatencyUs = "10") }
        advanceTimeBy(400)
        runCurrent()
        assertEquals(2, repo.traceFilters.size)
        val f = repo.traceFilters.last()
        assertEquals(listOf("openat"), f.syscalls)
        assertTrue(f.errorsOnly)
        assertEquals(10_000L, f.minLatencyNs)
        assertEquals("previous RPC cancelled", 1, repo.traceSubscribers)
    }

    @Test
    fun `existing events are re-filtered locally when the filter changes`() = runTest {
        val vm = started()
        repo.trace.emit(item(traceEvent(0, name = "read")))
        repo.trace.emit(item(traceEvent(1, name = "openat", ret = -2)))
        advanceTimeBy(150)
        assertEquals(2, vm.state.value.events.size)
        vm.onFilterChange { it.copy(errorsOnly = true) }
        advanceTimeBy(400)
        runCurrent()
        assertEquals(listOf("openat"), vm.state.value.events.map { it.syscallName })
    }

    @Test
    fun `pause freezes the list while counters keep running`() = runTest {
        val vm = started()
        repo.trace.emit(item(traceEvent(0)))
        advanceTimeBy(150)
        vm.togglePause()
        repo.trace.emit(item(traceEvent(1)))
        advanceTimeBy(150)
        assertEquals(1, vm.state.value.events.size)
        assertEquals(2L, vm.state.value.received)
        vm.togglePause()
        runCurrent()
        assertEquals(2, vm.state.value.events.size)
    }

    @Test
    fun `clear empties buffer and counters`() = runTest {
        val vm = started()
        repo.trace.emit(item(traceEvent(0, dropped = 3)))
        advanceTimeBy(150)
        vm.clear()
        runCurrent()
        assertEquals(0, vm.state.value.events.size)
        assertEquals(0L, vm.state.value.droppedUpstream)
    }

    @Test
    fun `reconnecting is surfaced`() = runTest {
        val vm = started()
        repo.trace.emit(StreamEvent.Reconnecting(1, 500, RuntimeException("x")))
        advanceTimeBy(150)
        assertTrue(vm.state.value.status is dev.vessel.console.ui.common.StreamStatus.Reconnecting)
    }

    @Test
    fun `stream is cancelled after the tab is hidden`() = runTest {
        val vm = TraceViewModel("c1", repo, filterDebounceMs = 0)
        val ui = subscribe(vm.state)
        advanceTimeBy(10)
        assertEquals(1, repo.traceSubscribers)
        ui.cancel()
        advanceTimeBy(5_100)
        assertEquals(0, repo.traceSubscribers)
    }
}
