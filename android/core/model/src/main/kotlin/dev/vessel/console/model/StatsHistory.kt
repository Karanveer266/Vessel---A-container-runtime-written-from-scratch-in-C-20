package dev.vessel.console.model

import dev.vessel.console.model.util.FloatRingBuffer
import dev.vessel.console.model.util.Series
import dev.vessel.console.model.util.snapshot

/**
 * Bounded rolling window of stats for one container, feeding the live charts.
 *
 * Memory is O(capacity) forever: at the default 1 Hz and 120 points this is a
 * 2-minute window costing ~2.4 KB of floats regardless of how long the
 * detail screen stays open. Mutable and single-owner; call [snapshot] to get
 * immutable data for UI state.
 */
class StatsHistory(val capacity: Int = DEFAULT_CAPACITY) {
    private val cpuMillicores = FloatRingBuffer(capacity)
    private val memoryCurrent = FloatRingBuffer(capacity)
    private val memoryMax = FloatRingBuffer(capacity)
    private val pids = FloatRingBuffer(capacity)
    private val psiCpuSome10 = FloatRingBuffer(capacity)
    private val psiMemSome10 = FloatRingBuffer(capacity)
    private val psiIoSome10 = FloatRingBuffer(capacity)

    var latest: StatsSample? = null
        private set

    fun add(s: StatsSample) {
        cpuMillicores.add(s.cpuMillicoresUsed.toFloat())
        memoryCurrent.add(s.memoryCurrentBytes.toFloat())
        memoryMax.add(s.memoryMaxBytes.toFloat())
        pids.add(s.pidsCurrent.toFloat())
        psiCpuSome10.add(s.cpuPressure.someAvg10)
        psiMemSome10.add(s.memoryPressure.someAvg10)
        psiIoSome10.add(s.ioPressure.someAvg10)
        latest = s
    }

    fun clear() {
        listOf(cpuMillicores, memoryCurrent, memoryMax, pids, psiCpuSome10, psiMemSome10, psiIoSome10)
            .forEach(FloatRingBuffer::clear)
        latest = null
    }

    fun snapshot(): StatsSnapshot = StatsSnapshot(
        cpuMillicores = cpuMillicores.snapshot(),
        memoryCurrentBytes = memoryCurrent.snapshot(),
        memoryMaxBytes = memoryMax.snapshot(),
        pids = pids.snapshot(),
        psiCpuSome10 = psiCpuSome10.snapshot(),
        psiMemorySome10 = psiMemSome10.snapshot(),
        psiIoSome10 = psiIoSome10.snapshot(),
        latest = latest,
    )

    companion object {
        const val DEFAULT_CAPACITY = 120
    }
}

data class StatsSnapshot(
    val cpuMillicores: Series = Series.Empty,
    val memoryCurrentBytes: Series = Series.Empty,
    val memoryMaxBytes: Series = Series.Empty,
    val pids: Series = Series.Empty,
    val psiCpuSome10: Series = Series.Empty,
    val psiMemorySome10: Series = Series.Empty,
    val psiIoSome10: Series = Series.Empty,
    val latest: StatsSample? = null,
)
