package dev.vessel.console.ui.detail

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.vessel.console.model.TraceEvent
import dev.vessel.console.model.TraceFilter
import dev.vessel.console.model.VesselRepository
import dev.vessel.console.model.flow.StreamEvent
import dev.vessel.console.model.flow.batchByTime
import dev.vessel.console.model.userMessage
import dev.vessel.console.model.util.RingBuffer
import dev.vessel.console.ui.common.Sharing
import dev.vessel.console.ui.common.StreamStatus
import dev.vessel.console.ui.common.toStatus
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.merge
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update

/** User-editable filter inputs (raw text for the text fields). */
data class TraceFilterInput(
    val errorsOnly: Boolean = false,
    /** Microseconds, as typed. Empty = no minimum. */
    val minLatencyUs: String = "",
    /** Comma/space separated syscall names. */
    val syscalls: String = "",
) {
    fun toFilter(containerId: String) = TraceFilter(
        containerId = containerId,
        errorsOnly = errorsOnly,
        minLatencyNs = (minLatencyUs.trim().toLongOrNull() ?: 0L).coerceAtLeast(0) * 1_000,
        syscalls = syscalls.split(',', ' ').map { it.trim() }.filter { it.isNotEmpty() }.distinct(),
    )
}

data class TraceUiState(
    val input: TraceFilterInput = TraceFilterInput(),
    /** Newest first. Immutable snapshot of the ring buffer; at most `capacity` items. */
    val events: List<TraceEvent> = emptyList(),
    val capacity: Int = TraceViewModel.DEFAULT_CAPACITY,
    /** Events the daemon/kernel reported as lost (sum of `dropped_events`). */
    val droppedUpstream: Long = 0,
    /** Older events discarded locally to keep the buffer bounded. */
    val evictedLocally: Long = 0,
    val received: Long = 0,
    val paused: Boolean = false,
    val status: StreamStatus = StreamStatus.Connecting,
)

/**
 * Live syscall trace.
 *
 * Pipeline (per active filter):
 * ```
 * StreamTrace(filter)          gRPC server stream, server-side filtering
 *   -> batchByTime(250 ms)     ≤ 4 UI updates/s regardless of event rate;
 *                              pending batch capped at `capacity` (drop-oldest)
 *   -> RingBuffer(capacity)    memory flat: newest 2,000 events, O(1) insert
 *   -> immutable snapshot      newest-first List for the LazyColumn
 * ```
 * Filter edits are debounced (typing "openat" must not open six RPCs) and
 * applied with `flatMapLatest`, which cancels the previous RPC. Events
 * already in the buffer are re-filtered locally so the list reacts
 * instantly, before the new server stream delivers.
 *
 * Pause freezes the *displayed* list while ingestion continues, so counters
 * stay truthful and unpausing jumps to the present.
 */
@OptIn(ExperimentalCoroutinesApi::class, FlowPreview::class)
class TraceViewModel(
    private val containerId: String,
    private val repository: VesselRepository,
    private val capacity: Int = DEFAULT_CAPACITY,
    private val batchMs: Long = 250,
    private val filterDebounceMs: Long = 400,
) : ViewModel() {

    private val input = MutableStateFlow(TraceFilterInput())
    private val paused = MutableStateFlow(false)

    /** Raw ingest result, before pause is applied. */
    private data class Ingest(
        val events: List<TraceEvent>,
        val droppedUpstream: Long,
        val evicted: Long,
        val received: Long,
        val status: StreamStatus,
    )

    /**
     * Confined to the single active inner flow of `flatMapLatest` (the
     * previous one is cancelled before the next starts), so no locking needed.
     */
    private val buffer = RingBuffer<TraceEvent>(capacity)
    private var evictedBase = 0L
    private var droppedUpstream = 0L
    private var received = 0L

    private val ingest: Flow<Ingest> = input
        .map { it.toFilter(containerId) }
        .distinctUntilChanged()
        .debounce(filterDebounceMs)
        .flatMapLatest { filter -> ingest(filter) }

    /** Inputs to the single-owner buffer: either a batch from the wire or a user "clear". */
    private sealed interface Signal {
        data class Batch(val events: List<StreamEvent<TraceEvent>>) : Signal
        data object Clear : Signal
    }

    private val clearRequests = MutableSharedFlow<Signal>(extraBufferCapacity = 1, onBufferOverflow = BufferOverflow.DROP_OLDEST)

    private fun ingest(filter: TraceFilter): Flow<Ingest> = flow {
        retainMatching(filter)
        emit(snapshot(StreamStatus.Connecting))
        var status: StreamStatus = StreamStatus.Connecting
        val batches = repository.streamTrace(filter)
            .batchByTime(batchMs, maxBatch = capacity)
            .map<List<StreamEvent<TraceEvent>>, Signal> { Signal.Batch(it) }
        // Both sources are consumed by this one collector, so the buffer has a single writer.
        merge(batches, clearRequests).collect { signal ->
            when (signal) {
                Signal.Clear -> {
                    buffer.clear()
                    evictedBase = buffer.evicted
                    droppedUpstream = 0
                    received = 0
                }
                is Signal.Batch -> for (event in signal.events) {
                    when (event) {
                        is StreamEvent.Item -> {
                            val e = event.value
                            received++
                            droppedUpstream += e.droppedEvents
                            // Belt and braces: an older daemon may ignore some filter fields.
                            if (filter.matches(e)) buffer.add(e)
                            status = StreamStatus.Live
                        }
                        is StreamEvent.Reconnecting -> status = event.toStatus()
                    }
                }
            }
            emit(snapshot(status))
        }
    }.catch { emit(snapshot(StreamStatus.Failed(it.userMessage()))) }

    private var frozen: List<TraceEvent> = emptyList()

    val state: StateFlow<TraceUiState> = combine(ingest, input, paused) { i, inp, p ->
        if (!p) frozen = i.events
        TraceUiState(
            input = inp,
            events = frozen,
            capacity = capacity,
            droppedUpstream = i.droppedUpstream,
            evictedLocally = i.evicted,
            received = i.received,
            paused = p,
            status = i.status,
        )
    }.stateIn(viewModelScope, Sharing.WhileUiVisible, TraceUiState(capacity = capacity))

    fun onFilterChange(transform: (TraceFilterInput) -> TraceFilterInput) = input.update(transform)

    fun togglePause() = paused.update { !it }

    /** Clears the buffer and counters. Routed through the ingest collector to keep a single writer. */
    fun clear() {
        frozen = emptyList()
        clearRequests.tryEmit(Signal.Clear)
    }

    private fun retainMatching(filter: TraceFilter) {
        val keep = buffer.toList().filter(filter::matches)
        buffer.clear()
        buffer.addAll(keep)
    }

    private fun snapshot(status: StreamStatus) = Ingest(
        events = List(buffer.size) { buffer[buffer.size - 1 - it] },
        droppedUpstream = droppedUpstream,
        evicted = buffer.evicted - evictedBase,
        received = received,
        status = status,
    )

    companion object {
        const val DEFAULT_CAPACITY = 2_000
    }
}
