package dev.vessel.console.ui.detail

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.vessel.console.model.Container
import dev.vessel.console.model.ContainerAction
import dev.vessel.console.model.ExecResult
import dev.vessel.console.model.StatsHistory
import dev.vessel.console.model.StatsSnapshot
import dev.vessel.console.model.SyscallSummary
import dev.vessel.console.model.VesselRepository
import dev.vessel.console.model.flow.StreamEvent
import dev.vessel.console.model.format.CommandLine
import dev.vessel.console.model.perform
import dev.vessel.console.model.userMessage
import dev.vessel.console.ui.common.Load
import dev.vessel.console.ui.common.Sharing
import dev.vessel.console.ui.common.StreamStatus
import dev.vessel.console.ui.common.toStatus
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull

// ============================================================================
// Header: container metadata + lifecycle actions
// ============================================================================

data class DetailHeaderState(
    val container: Load<Container> = Load.Loading,
    val pending: ContainerAction? = null,
    val removed: Boolean = false,
    val message: String? = null,
)

class ContainerHeaderViewModel(
    private val containerId: String,
    private val repository: VesselRepository,
) : ViewModel() {
    private val _state = MutableStateFlow(DetailHeaderState())
    val state: StateFlow<DetailHeaderState> = _state.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            val result = runCatching { repository.getContainer(containerId) }
            result.exceptionOrNull()?.let { if (it is CancellationException) throw it }
            _state.update { s ->
                result.fold(
                    onSuccess = { s.copy(container = Load.Ready(it)) },
                    onFailure = { e -> if (s.container is Load.Ready) s.copy(message = e.userMessage()) else s.copy(container = Load.Failed(e.userMessage())) },
                )
            }
        }
    }

    fun perform(action: ContainerAction) {
        val current = _state.value.container.valueOrNull ?: return
        if (_state.value.pending != null) return
        _state.update { it.copy(pending = action) }
        viewModelScope.launch {
            try {
                val updated = repository.perform(current, action)
                _state.update {
                    if (updated == null) it.copy(pending = null, removed = true)
                    else it.copy(pending = null, container = Load.Ready(updated))
                }
            } catch (c: CancellationException) {
                throw c
            } catch (t: Throwable) {
                _state.update { it.copy(pending = null, message = "${action.label} failed: ${t.userMessage()}") }
            }
        }
    }

    fun messageShown() = _state.update { it.copy(message = null) }
}

// ============================================================================
// Live stats tab
// ============================================================================

data class LiveStatsState(
    val stats: StatsSnapshot = StatsSnapshot(),
    val capacity: Int = StatsHistory.DEFAULT_CAPACITY,
    val status: StreamStatus = StreamStatus.Connecting,
)

/**
 * Feeds the live charts from StreamStats(id).
 *
 * Samples go into a [StatsHistory] (fixed-size primitive ring buffers) and
 * each produces an immutable [StatsSnapshot]. `conflate()` sits *after* the
 * accumulation step: no sample is lost from the history, but if the UI falls
 * behind (e.g. 100 ms interval on a slow device) intermediate snapshots are
 * skipped and only the newest is rendered.
 */
class LiveStatsViewModel(
    private val containerId: String,
    private val repository: VesselRepository,
    private val intervalMs: Int = 1_000,
    private val capacity: Int = StatsHistory.DEFAULT_CAPACITY,
) : ViewModel() {

    val state: StateFlow<LiveStatsState> = flow {
        // History is per-subscription: after the stream is stopped in the
        // background, the window restarts fresh rather than drawing a
        // misleading straight line across the gap.
        val history = StatsHistory(capacity)
        emit(LiveStatsState(capacity = capacity))
        repository.streamStats(containerId, intervalMs).collect { event ->
            when (event) {
                is StreamEvent.Item -> {
                    history.add(event.value)
                    emit(LiveStatsState(history.snapshot(), capacity, StreamStatus.Live))
                }
                is StreamEvent.Reconnecting ->
                    emit(LiveStatsState(history.snapshot(), capacity, event.toStatus()))
            }
        }
    }
        .conflate()
        .catch { emit(LiveStatsState(capacity = capacity, status = StreamStatus.Failed(it.userMessage()))) }
        .stateIn(viewModelScope, Sharing.WhileUiVisible, LiveStatsState(capacity = capacity))
}

// ============================================================================
// Syscall summary tab
// ============================================================================

data class SummaryState(val summary: Load<SyscallSummary> = Load.Loading, val refreshing: Boolean = false)

/** Polls GetSyscallSummary while visible (counters are exact, cheap to fetch). */
class SyscallSummaryViewModel(
    private val containerId: String,
    private val repository: VesselRepository,
    private val pollIntervalMs: Long = 5_000,
) : ViewModel() {
    private val manual = MutableStateFlow(0)

    val state: StateFlow<SummaryState> = flow {
        var last: Load<SyscallSummary> = Load.Loading
        while (true) {
            last = try {
                Load.Ready(repository.syscallSummary(containerId))
            } catch (c: CancellationException) {
                throw c
            } catch (t: Throwable) {
                if (last is Load.Ready) last else Load.Failed(t.userMessage())
            }
            emit(SummaryState(last))
            // Wake early on manual refresh.
            val seen = manual.value
            withTimeoutOrNull(pollIntervalMs) { manual.first { it != seen } }
        }
    }.stateIn(viewModelScope, Sharing.WhileUiVisible, SummaryState())

    fun refresh() = manual.update { it + 1 }
}

// ============================================================================
// Exec tab
// ============================================================================

data class ExecState(
    val command: String = "ps aux",
    val running: Boolean = false,
    val result: ExecResult? = null,
    val error: String? = null,
    val durationMs: Long = 0,
)

class ExecViewModel(
    private val containerId: String,
    private val repository: VesselRepository,
    private val clock: () -> Long = System::currentTimeMillis,
) : ViewModel() {
    private val _state = MutableStateFlow(ExecState())
    val state: StateFlow<ExecState> = _state.asStateFlow()

    fun onCommandChange(command: String) = _state.update { it.copy(command = command, error = null) }

    fun run() {
        val s = _state.value
        if (s.running) return
        val args = try {
            CommandLine.split(s.command)
        } catch (e: CommandLine.ParseException) {
            _state.update { it.copy(error = e.message) }
            return
        }
        if (args.isEmpty()) {
            _state.update { it.copy(error = "Enter a command") }
            return
        }
        _state.update { it.copy(running = true, error = null) }
        viewModelScope.launch {
            val start = clock()
            try {
                val result = repository.exec(containerId, args)
                _state.update { it.copy(running = false, result = result, durationMs = clock() - start) }
            } catch (c: CancellationException) {
                throw c
            } catch (t: Throwable) {
                _state.update { it.copy(running = false, error = t.userMessage()) }
            }
        }
    }
}
