package dev.vessel.console.ui.detail

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.vessel.console.model.LineAssembler
import dev.vessel.console.model.LogStream
import dev.vessel.console.model.VesselRepository
import dev.vessel.console.model.flow.StreamEvent
import dev.vessel.console.model.flow.batchByTime
import dev.vessel.console.model.userMessage
import dev.vessel.console.model.util.RingBuffer
import dev.vessel.console.ui.common.Sharing
import dev.vessel.console.ui.common.StreamStatus
import dev.vessel.console.ui.common.toStatus
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.stateIn

data class LogLine(val seq: Long, val stream: LogStream, val text: String)

data class LogsUiState(
    /** Oldest first (terminal order). */
    val lines: List<LogLine> = emptyList(),
    val status: StreamStatus = StreamStatus.Connecting,
    /** The stream completed: container exited and the daemon closed the follow. */
    val ended: Boolean = false,
)

/**
 * StreamLogs(follow) -> complete lines -> bounded buffer.
 *
 * stdout and stderr get separate [LineAssembler]s: chunks of the two pipes
 * interleave, and a partial stdout line must not be glued to stderr bytes.
 *
 * On reconnect the daemon replays `tail_bytes` of history, so the buffer is
 * cleared when a [StreamEvent.Reconnecting] arrives — the replay then
 * repopulates it without duplicates.
 */
class LogsViewModel(
    private val containerId: String,
    private val repository: VesselRepository,
    private val capacity: Int = DEFAULT_CAPACITY,
) : ViewModel() {

    val state: StateFlow<LogsUiState> = flow {
        val lines = RingBuffer<LogLine>(capacity)
        val assemblers = mutableMapOf<LogStream, LineAssembler>()
        var seq = 0L
        var status: StreamStatus = StreamStatus.Connecting
        emit(LogsUiState())
        repository.streamLogs(containerId, follow = true)
            .batchByTime(periodMs = 200, maxBatch = 1_024)
            .collect { batch ->
                for (event in batch) {
                    when (event) {
                        is StreamEvent.Item -> {
                            val chunk = event.value
                            val assembler = assemblers.getOrPut(chunk.stream) { LineAssembler() }
                            assembler.feed(chunk.data).forEach { lines.add(LogLine(seq++, chunk.stream, it)) }
                            status = StreamStatus.Live
                        }
                        is StreamEvent.Reconnecting -> {
                            lines.clear()
                            assemblers.clear()
                            status = event.toStatus()
                        }
                    }
                }
                emit(LogsUiState(lines.toList(), status))
            }
        // Normal completion: flush unterminated trailing output.
        assemblers.forEach { (stream, a) -> a.flush().forEach { lines.add(LogLine(seq++, stream, it)) } }
        emit(LogsUiState(lines.toList(), StreamStatus.Live, ended = true))
    }
        .catch { emit(LogsUiState(status = StreamStatus.Failed(it.userMessage()))) }
        .stateIn(viewModelScope, Sharing.WhileUiVisible, LogsUiState())

    companion object {
        const val DEFAULT_CAPACITY = 5_000
    }
}
