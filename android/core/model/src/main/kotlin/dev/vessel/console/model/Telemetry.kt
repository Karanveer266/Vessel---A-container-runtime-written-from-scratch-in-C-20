package dev.vessel.console.model

/** Pressure Stall Information for one resource (cpu / memory / io). */
data class Psi(
    val someAvg10: Float = 0f,
    val someAvg60: Float = 0f,
    val fullAvg10: Float = 0f,
    val someTotalUsec: Long = 0,
) {
    companion object {
        val Zero = Psi()
    }
}

data class StatsSample(
    val containerId: String,
    val timestampUnixNs: Long,
    val cpuUsageUsec: Long,
    val cpuNrThrottled: Long,
    val cpuThrottledUsec: Long,
    val cpuMillicoresUsed: Int,
    val memoryCurrentBytes: Long,
    /** 0 = unlimited. */
    val memoryMaxBytes: Long,
    val memoryPeakBytes: Long,
    val memoryEventsHigh: Long,
    val memoryEventsMax: Long,
    val memoryEventsOom: Long,
    val memoryEventsOomKill: Long,
    val pidsCurrent: Int,
    /** 0 = unlimited. */
    val pidsMax: Int,
    val cpuPressure: Psi,
    val memoryPressure: Psi,
    val ioPressure: Psi,
    val netRxBytes: Long,
    val netTxBytes: Long,
    val droppedSamples: Long,
) {
    val hasOomEvents: Boolean get() = memoryEventsOom > 0 || memoryEventsOomKill > 0
}

enum class TraceEventKind(val label: String) {
    Unspecified("?"),
    Syscall("syscall"),
    Memory("memory"),
    Net("net"),
    ProcessExec("exec"),
    ProcessExit("exit"),
    OomKill("oom-kill"),
    MemoryPressure("mem-pressure"),
    SeccompDenied("seccomp-denied"),
    Lifecycle("lifecycle"),
}

data class TraceEvent(
    /**
     * Client-assigned, strictly increasing and unique across *all* trace
     * subscriptions of one repository instance (survives reconnects and
     * resubscriptions). The wire format has no event id, but lazy lists need
     * stable unique keys — this is it.
     */
    val seq: Long,
    val containerId: String,
    val kind: TraceEventKind,
    val timestampUnixNs: Long,
    val hostPid: Int,
    val hostTid: Int,
    val comm: String,
    val syscallNr: Int,
    val syscallName: String,
    val ret: Long,
    val latencyNs: Long,
    val bytes: Long,
    val egress: Boolean,
    val detail: String,
    /** Events lost (kernel ring buffer + server queue) right before this one. */
    val droppedEvents: Long,
    val backend: TraceBackend,
) {
    val isError: Boolean get() = kind == TraceEventKind.Syscall && ret < 0 ||
        kind == TraceEventKind.SeccompDenied || kind == TraceEventKind.OomKill
}

/** Server-side filter for StreamTrace (also re-applied client-side to the buffer). */
data class TraceFilter(
    val containerId: String,
    val errorsOnly: Boolean = false,
    val minLatencyNs: Long = 0,
    /** Exact syscall names, empty = all. */
    val syscalls: List<String> = emptyList(),
    val kinds: Set<TraceEventKind> = emptySet(),
) {
    fun matches(e: TraceEvent): Boolean =
        (!errorsOnly || e.isError) &&
            (minLatencyNs <= 0 || e.latencyNs >= minLatencyNs) &&
            (syscalls.isEmpty() || e.syscallName in syscalls) &&
            (kinds.isEmpty() || e.kind in kinds)
}

data class SyscallCount(
    val name: String,
    val nr: Int,
    val count: Long,
    val errors: Long,
    val totalLatencyNs: Long,
    val maxLatencyNs: Long,
) {
    val avgLatencyNs: Long get() = if (count == 0L) 0 else totalLatencyNs / count
}

data class SyscallSummary(
    val containerId: String,
    /** Sorted by count, descending (daemon guarantee; mapper re-sorts defensively). */
    val syscalls: List<SyscallCount>,
    val totalEvents: Long,
    val droppedEvents: Long,
    val backend: TraceBackend,
)

enum class LogStream { Unspecified, Stdout, Stderr }

data class LogChunk(
    val stream: LogStream,
    val data: ByteArray,
    val timestampUnixNs: Long,
) {
    // ByteArray needs content equality for a meaningful data class.
    override fun equals(other: Any?): Boolean =
        other is LogChunk && stream == other.stream &&
            timestampUnixNs == other.timestampUnixNs && data.contentEquals(other.data)

    override fun hashCode(): Int =
        (stream.hashCode() * 31 + data.contentHashCode()) * 31 + timestampUnixNs.hashCode()
}
