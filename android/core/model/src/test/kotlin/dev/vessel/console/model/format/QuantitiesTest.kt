package dev.vessel.console.model.format

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.test.assertFailsWith

class QuantitiesTest {

    @Test
    fun `parses byte quantities in k8s and docker notation`() {
        assertEquals(0L, Quantities.parseBytes(""))
        assertEquals(1024L, Quantities.parseBytes("1024"))
        assertEquals(512L shl 20, Quantities.parseBytes("512Mi"))
        assertEquals(512_000_000L, Quantities.parseBytes("512M"))
        assertEquals(1536L shl 20, Quantities.parseBytes("1.5Gi"))
        assertEquals(2_000_000_000L, Quantities.parseBytes("2g"))
        assertEquals(256L shl 10, Quantities.parseBytes(" 256 KiB "))
    }

    @Test
    fun `rejects malformed byte quantities`() {
        assertNull(Quantities.parseBytes("12x"))
        assertNull(Quantities.parseBytes("-5M"))
        assertNull(Quantities.parseBytes("M"))
        assertNull(Quantities.parseBytes("99999999Ti")) // overflows Long
    }

    @Test
    fun `parses cpu quantities`() {
        assertEquals(0, Quantities.parseMillicores(""))
        assertEquals(500, Quantities.parseMillicores("500m"))
        assertEquals(500, Quantities.parseMillicores("0.5"))
        assertEquals(2000, Quantities.parseMillicores("2"))
        assertEquals(1250, Quantities.parseMillicores("1.25"))
        assertNull(Quantities.parseMillicores("lots"))
        assertNull(Quantities.parseMillicores("-1"))
    }

    @Test
    fun `validates cpuset syntax`() {
        listOf("", "0", "0-3", "0,2,4", "0-1,6").forEach { assertTrue(it, Quantities.isValidCpuset(it)) }
        listOf("a", "3-1", "0-", "0--2", "1,,2", "-1").forEach { assertFalse(it, Quantities.isValidCpuset(it)) }
    }

    @Test
    fun `counts reject negatives`() {
        assertEquals(0, Quantities.parseCount(""))
        assertEquals(64, Quantities.parseCount("64"))
        assertNull(Quantities.parseCount("-1"))
    }
}

class CommandLineTest {

    @Test
    fun `splits on whitespace`() {
        assertEquals(listOf("ls", "-la", "/"), CommandLine.split("  ls   -la /  "))
        assertEquals(emptyList<String>(), CommandLine.split("   "))
    }

    @Test
    fun `honours quotes and escapes`() {
        assertEquals(listOf("sh", "-c", "echo hi && sleep 1"), CommandLine.split("sh -c 'echo hi && sleep 1'"))
        assertEquals(listOf("echo", "a \"b\" \\c"), CommandLine.split("""echo "a \"b\" \\c""""))
        assertEquals(listOf("a b"), CommandLine.split("""a\ b"""))
        assertEquals(listOf("pre'fix'post"), CommandLine.split("""pre"'fix'"post"""))
        assertEquals(listOf(""), CommandLine.split("''"))
    }

    @Test
    fun `reports unterminated quotes`() {
        assertFailsWith<CommandLine.ParseException> { CommandLine.split("echo 'oops") }
        assertFailsWith<CommandLine.ParseException> { CommandLine.split("echo \"oops") }
        assertFailsWith<CommandLine.ParseException> { CommandLine.split("echo \\") }
    }
}
