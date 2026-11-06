package dev.vessel.console.model.format

import java.util.Locale
import kotlin.math.abs

/**
 * Pure, locale-stable formatting helpers. They live in the platform-free model
 * module so they are unit tested on the JVM and shared by every screen.
 *
 * All output uses [Locale.ROOT] — these are operator-facing technical figures
 * ("1.5 GiB", "250m") that should look the same as `vesselctl` output, not
 * localized prose.
 */
object Formatters {
    private val BINARY_UNITS = arrayOf("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")

    /** IEC binary units, as reported by cgroups: 1536 -> "1.5 KiB". */
    fun bytes(bytes: Long): String {
        if (bytes < 0) return "-" + bytes(-bytes)
        if (bytes < 1024) return "$bytes B"
        var value = bytes.toDouble()
        var unit = 0
        while (value >= 1024 && unit < BINARY_UNITS.lastIndex) {
            value /= 1024
            unit++
        }
        return "${trim(value)} ${BINARY_UNITS[unit]}"
    }

    /** Limits use 0 as "unlimited" throughout the API. */
    fun bytesOrUnlimited(bytes: Long): String = if (bytes <= 0) "unlimited" else bytes(bytes)

    /**
     * Kubernetes-style CPU quantity: 250 -> "250m", 1000 -> "1", 1500 -> "1.5".
     * 0 means unlimited when used for limits; pass [zeroLabel] for that case.
     */
    fun millicores(millicores: Int, zeroLabel: String = "0m"): String = when {
        millicores == 0 -> zeroLabel
        millicores % 1000 == 0 -> "${millicores / 1000}"
        millicores > 1000 -> trim(millicores / 1000.0)
        else -> "${millicores}m"
    }

    /** Nanoseconds to the most readable unit: 850 -> "850 ns", 1_500 -> "1.5 µs", 2_500_000 -> "2.5 ms". */
    fun durationNs(ns: Long): String {
        val a = abs(ns)
        return when {
            a < 1_000 -> "$ns ns"
            a < 1_000_000 -> "${trim(ns / 1_000.0)} µs"
            a < 1_000_000_000 -> "${trim(ns / 1_000_000.0)} ms"
            else -> durationSeconds(ns / 1_000_000_000.0)
        }
    }

    fun durationMs(ms: Double): String = when {
        ms < 1.0 -> "${trim(ms * 1000)} µs"
        ms < 1000.0 -> "${trim(ms)} ms"
        else -> durationSeconds(ms / 1000.0)
    }

    /** Uptime-style: 42 -> "42s", 125 -> "2m 5s", 3_725 -> "1h 2m", 90_000 -> "1d 1h". */
    fun uptime(seconds: Long): String {
        if (seconds < 60) return "${seconds}s"
        val d = seconds / 86_400
        val h = (seconds % 86_400) / 3_600
        val m = (seconds % 3_600) / 60
        val s = seconds % 60
        return when {
            d > 0 -> "${d}d ${h}h"
            h > 0 -> "${h}h ${m}m"
            else -> "${m}m ${s}s"
        }
    }

    /** Large counters (syscall counts, events): 1_234 -> "1.2k", 5_600_000 -> "5.6M". */
    fun count(n: Long): String {
        val a = abs(n)
        return when {
            a < 1_000 -> "$n"
            a < 1_000_000 -> "${trim(n / 1_000.0)}k"
            a < 1_000_000_000 -> "${trim(n / 1_000_000.0)}M"
            else -> "${trim(n / 1_000_000_000.0)}G"
        }
    }

    fun percent(fraction: Double): String = "${trim(fraction * 100)}%"

    /** PSI avg values are already percentages (0..100). */
    fun psi(value: Float): String = String.format(Locale.ROOT, "%.2f%%", value)

    fun gflops(value: Double): String = if (value >= 1000) "${trim(value / 1000)} TFLOP/s" else "${trim(value)} GFLOP/s"

    fun bandwidth(gbps: Double): String = "${trim(gbps)} GB/s"

    /** Syscall return values: negative values are -errno. */
    fun syscallReturn(ret: Long): String = if (ret < 0) {
        val name = Errno.name((-ret).toInt())
        if (name != null) "-$name" else "$ret"
    } else {
        "$ret"
    }

    private fun durationSeconds(s: Double): String = when {
        s < 60 -> "${trim(s)} s"
        else -> uptime(s.toLong())
    }

    /** One decimal below 10, none above 100 — keeps chips and axes narrow. */
    private fun trim(v: Double): String {
        val a = abs(v)
        val s = when {
            a >= 100 -> String.format(Locale.ROOT, "%.0f", v)
            a >= 10 -> String.format(Locale.ROOT, "%.1f", v)
            else -> String.format(Locale.ROOT, "%.2f", v)
        }
        return if (s.contains('.')) s.trimEnd('0').trimEnd('.') else s
    }
}

/** The handful of errno values that dominate syscall traces (Linux generic numbering). */
object Errno {
    private val names = mapOf(
        1 to "EPERM", 2 to "ENOENT", 3 to "ESRCH", 4 to "EINTR", 5 to "EIO", 9 to "EBADF",
        10 to "ECHILD", 11 to "EAGAIN", 12 to "ENOMEM", 13 to "EACCES", 14 to "EFAULT",
        16 to "EBUSY", 17 to "EEXIST", 20 to "ENOTDIR", 21 to "EISDIR", 22 to "EINVAL",
        24 to "EMFILE", 25 to "ENOTTY", 28 to "ENOSPC", 32 to "EPIPE", 38 to "ENOSYS",
        110 to "ETIMEDOUT", 111 to "ECONNREFUSED", 115 to "EINPROGRESS",
    )

    fun name(errno: Int): String? = names[errno]
}
