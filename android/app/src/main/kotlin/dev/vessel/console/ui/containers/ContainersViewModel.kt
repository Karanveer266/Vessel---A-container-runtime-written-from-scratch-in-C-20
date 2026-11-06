package dev.vessel.console.ui.containers

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.vessel.console.model.Container
import dev.vessel.console.model.ContainerAction
import dev.vessel.console.model.StatsSample
import dev.vessel.console.model.VesselRepository
import dev.vessel.console.model.flow.StreamEvent
import dev.vessel.console.model.flow.batchByTime
import dev.vessel.console.model.perform
import dev.vessel.console.model.userMessage
import dev.vessel.console.model.util.FloatRingBuffer
import dev.vessel.console.model.util.Series
import dev.vessel.console.model.util.snapshot
import dev.vessel.console.ui.common.Load
import dev.vessel.console.ui.common.Sharing
import dev.vessel.console.ui.common.StreamStatus
import dev.vessel.console.ui.common.toStatus
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.onStart
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** Trend data for one row. Immutable snapshot; equality by content. */
data class Sparks(
    val cpuMillicores: Series = Series.Empty,
    val memoryBytes: Series = Series.Empty,
    val latest: StatsSample? = null,
)

data class ContainerRow(
    val container: Container,
    val sparks: Sparks,
    /** Action currently in flight for this row (disables the menu, shows progress). */
    val pending: ContainerAction?,
)

data class ContainersUiState(
    val rows: Load<List<ContainerRow>> = Load.Loading,
    val refreshing: Boolean = false,
    val statsStatus: StreamStatus = StreamStatus.Connecting,
    val showStopped: Boolean = true,
    val message: String? = null,
)

/**
 * Containers list.
 *
 * Two inputs are combined into one immutable [ContainersUiState]:
 *  * the container list (unary ListContainers; pull-to-refresh, after every
 *    action, and a slow poll while visible to catch exits);
 *  * ONE multiplexed StreamStats("") for every running container, feeding
 *    per-row sparklines. One stream instead of N keeps the connection count
 *    flat no matter how many rows are on screen.
 *
 * Both are only active while the UI is subscribed ([Sharing.WhileUiVisible]),
 * so backgrounding the app stops polling and cancels the stream.
 */
class ContainersViewModel(
    private val repository: VesselRepository,
    private val pollIntervalMs: Long = 10_000,
    private val sparkCapacity: Int = SPARK_CAPACITY,
) : ViewModel() {

    private data class ListState(
        val containers: Load<List<Container>> = Load.Loading,
        val refreshing: Boolean = false,
        val pending: Map<String, ContainerAction> = emptyMap(),
        val showStopped: Boolean = true,
        val message: String? = null,
    )

    private val list = MutableStateFlow(ListState())

    private data class SparkState(val byId: Map<String, Sparks> = emptyMap(), val status: StreamStatus = StreamStatus.Connecting)

    /** Accumulates the multiplexed stats stream into bounded per-container ring buffers. */
    private val sparks: Flow<SparkState> = flow {
        val cpu = LinkedHashMap<String, FloatRingBuffer>()
        val mem = LinkedHashMap<String, FloatRingBuffer>()
        val latest = HashMap<String, StatsSample>()
        var status: StreamStatus = StreamStatus.Connecting
        emit(SparkState())
        repository.streamStats(ref = "")
            // N containers × 1 Hz -> at most one UI update per period.
            .batchByTime(periodMs = BATCH_MS, maxBatch = MAX_BATCH)
            .collect { batch ->
                for (event in batch) {
                    when (event) {
                        is StreamEvent.Item -> {
                            val s = event.value
                            cpu.getOrPut(s.containerId) { FloatRingBuffer(sparkCapacity) }.add(s.cpuMillicoresUsed.toFloat())
                            mem.getOrPut(s.containerId) { FloatRingBuffer(sparkCapacity) }.add(s.memoryCurrentBytes.toFloat())
                            latest[s.containerId] = s
                            status = StreamStatus.Live
                        }
                        is StreamEvent.Reconnecting -> status = event.toStatus()
                    }
                }
                prune(cpu, mem, latest)
                emit(
                    SparkState(
                        byId = cpu.keys.associateWith { id ->
                            Sparks(cpu.getValue(id).snapshot(), mem.getValue(id).snapshot(), latest[id])
                        },
                        status = status,
                    ),
                )
            }
    }.catch { emit(SparkState(status = StreamStatus.Failed(it.userMessage()))) }

    /** Slow poll while visible; its emissions just keep `combine` alive. */
    private val poller: Flow<Unit> = flow {
        while (true) {
            emit(Unit)
            delay(pollIntervalMs)
            load(userInitiated = false)
        }
    }.onStart { load(userInitiated = false) }

    val state: StateFlow<ContainersUiState> = combine(list, sparks, poller) { l, s, _ ->
        ContainersUiState(
            rows = when (val c = l.containers) {
                is Load.Ready -> Load.Ready(
                    c.value.map { ContainerRow(it, s.byId[it.id] ?: Sparks(), l.pending[it.id]) },
                )
                is Load.Failed -> c
                Load.Idle, Load.Loading -> Load.Loading
            },
            refreshing = l.refreshing,
            statsStatus = s.status,
            showStopped = l.showStopped,
            message = l.message,
        )
    }.stateIn(viewModelScope, Sharing.WhileUiVisible, ContainersUiState())

    fun refresh() {
        viewModelScope.launch { load(userInitiated = true) }
    }

    fun setShowStopped(show: Boolean) {
        list.update { it.copy(showStopped = show) }
        refresh()
    }

    fun perform(container: Container, action: ContainerAction) {
        if (list.value.pending.containsKey(container.id)) return
        list.update { it.copy(pending = it.pending + (container.id to action)) }
        viewModelScope.launch {
            val result = runCatching { repository.perform(container, action) }
            result.exceptionOrNull()?.let { if (it is CancellationException) throw it }
            list.update { s ->
                val updated = when (val c = s.containers) {
                    is Load.Ready -> Load.Ready(applyResult(c.value, container, action, result.getOrNull()))
                    else -> c
                }
                s.copy(
                    containers = if (result.isSuccess) updated else s.containers,
                    pending = s.pending - container.id,
                    message = result.exceptionOrNull()?.let { "${action.label} failed: ${it.userMessage()}" },
                )
            }
            load(userInitiated = false)
        }
    }

    fun messageShown() = list.update { it.copy(message = null) }

    private suspend fun load(userInitiated: Boolean) {
        if (userInitiated) list.update { it.copy(refreshing = true) }
        try {
            val containers = repository.listContainers(includeStopped = list.value.showStopped)
                .sortedWith(compareBy<Container> { it.state.ordinal != RUNNING_ORDINAL }.thenByDescending { it.createdAtUnixNs })
            list.update { it.copy(containers = Load.Ready(containers), refreshing = false) }
        } catch (c: CancellationException) {
            throw c
        } catch (t: Throwable) {
            list.update { s ->
                // Keep showing stale data on a failed background refresh; only report.
                if (s.containers is Load.Ready) {
                    s.copy(refreshing = false, message = if (userInitiated) t.userMessage() else s.message)
                } else {
                    s.copy(containers = Load.Failed(t.userMessage()), refreshing = false)
                }
            }
        }
    }

    /** Optimistically reflect the daemon's answer before the follow-up list refresh lands. */
    private fun applyResult(list: List<Container>, target: Container, action: ContainerAction, updated: Container?) =
        if (action == ContainerAction.Remove) list.filterNot { it.id == target.id }
        else list.map { if (it.id == target.id && updated != null) updated else it }

    /** Bound the per-container maps: forget the earliest-seen containers beyond the cap (LinkedHashMap insertion order). */
    private fun prune(vararg maps: MutableMap<String, *>) {
        val first = maps.first()
        while (first.size > MAX_TRACKED) {
            val oldest = first.keys.first()
            maps.forEach { it.remove(oldest) }
        }
    }

    companion object {
        const val SPARK_CAPACITY = 30
        private const val BATCH_MS = 500L
        private const val MAX_BATCH = 512
        private const val MAX_TRACKED = 128
        private val RUNNING_ORDINAL = dev.vessel.console.model.ContainerState.Running.ordinal
    }
}
