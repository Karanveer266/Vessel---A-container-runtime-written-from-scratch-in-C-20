package dev.vessel.console.model.format

import dev.vessel.console.model.format.Formatters.bytes
import dev.vessel.console.model.format.Formatters.count
import dev.vessel.console.model.format.Formatters.durationMs
import dev.vessel.console.model.format.Formatters.durationNs
import dev.vessel.console.model.format.Formatters.millicores
import dev.vessel.console.model.format.Formatters.uptime
import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Locale

class FormattersTest {

    @Test
    fun `bytes uses IEC units with trimmed precision`() {
        assertEquals("0 B", bytes(0))
        assertEquals("1023 B", bytes(1023))
        assertEquals("1 KiB", bytes(1024))
        assertEquals("1.5 KiB", bytes(1536))
        assertEquals("512 MiB", bytes(512L * 1024 * 1024))
        assertEquals("1.25 GiB", bytes(1_342_177_280))
        assertEquals("15.6 GiB", bytes(16_750_372_454))
        assertEquals("8 EiB", bytes(Long.MAX_VALUE))
        assertEquals("-2 KiB", bytes(-2048))
    }

    @Test
    fun `zero limits read as unlimited`() {
        assertEquals("unlimited", Formatters.bytesOrUnlimited(0))
        assertEquals("64 MiB", Formatters.bytesOrUnlimited(64L shl 20))
        assertEquals("unlimited", millicores(0, zeroLabel = "unlimited"))
    }

    @Test
    fun `millicores follow kubernetes notation`() {
        assertEquals("250m", millicores(250))
        assertEquals("999m", millicores(999))
        assertEquals("1", millicores(1000))
        assertEquals("1.5", millicores(1500))
        assertEquals("4", millicores(4000))
        assertEquals("0m", millicores(0))
    }

    @Test
    fun `durations pick the most readable unit`() {
        assertEquals("850 ns", durationNs(850))
        assertEquals("1.5 µs", durationNs(1_500))
        assertEquals("12.3 µs", durationNs(12_345))
        assertEquals("2.5 ms", durationNs(2_500_000))
        assertEquals("250 ms", durationNs(250_000_000))
        assertEquals("1.2 s", durationNs(1_200_000_000))
        assertEquals("2m 5s", durationNs(125_000_000_000))
        assertEquals("500 µs", durationMs(0.5))
        assertEquals("3.14 ms", durationMs(3.14159))
    }

    @Test
    fun `uptime is compact`() {
        assertEquals("42s", uptime(42))
        assertEquals("2m 5s", uptime(125))
        assertEquals("1h 2m", uptime(3_725))
        assertEquals("1d 1h", uptime(90_000))
    }

    @Test
    fun `counts abbreviate`() {
        assertEquals("999", count(999))
        assertEquals("1.23k", count(1_234))
        assertEquals("5.6M", count(5_600_000))
        assertEquals("2G", count(2_000_000_000))
    }

    @Test
    fun `syscall return renders errno names`() {
        assertEquals("-ENOENT", Formatters.syscallReturn(-2))
        assertEquals("-EAGAIN", Formatters.syscallReturn(-11))
        assertEquals("-999", Formatters.syscallReturn(-999))
        assertEquals("4096", Formatters.syscallReturn(4096))
    }

    @Test
    fun `output is locale independent`() {
        val saved = Locale.getDefault()
        try {
            Locale.setDefault(Locale.GERMANY) // decimal comma locale
            assertEquals("1.5 KiB", bytes(1536))
            assertEquals("12.50%", Formatters.psi(12.5f))
        } finally {
            Locale.setDefault(saved)
        }
    }

    @Test
    fun `gpu throughput units`() {
        assertEquals("850 GFLOP/s", Formatters.gflops(850.0))
        assertEquals("19.5 TFLOP/s", Formatters.gflops(19_500.0))
        assertEquals("180 GB/s", Formatters.bandwidth(180.0))
    }
}
