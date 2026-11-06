package dev.vessel.console.model

import org.junit.Assert.assertEquals
import org.junit.Test

class LineAssemblerTest {

    @Test
    fun `joins lines split across chunks`() {
        val a = LineAssembler()
        assertEquals(emptyList<String>(), a.feed("hel".toByteArray()))
        assertEquals(listOf("hello"), a.feed("lo\nwor".toByteArray()))
        assertEquals(listOf("world"), a.feed("ld\n".toByteArray()))
        assertEquals(emptyList<String>(), a.flush())
    }

    @Test
    fun `keeps multibyte utf-8 sequences intact across chunk boundaries`() {
        val bytes = "µs → ok\n".toByteArray(Charsets.UTF_8)
        val a = LineAssembler()
        val out = bytes.toList().chunked(1).flatMap { a.feed(it.toByteArray()) }
        assertEquals(listOf("µs → ok"), out)
    }

    @Test
    fun `strips CRLF and flushes trailing partial line`() {
        val a = LineAssembler()
        assertEquals(listOf("one", "two"), a.feed("one\r\ntwo\nthr".toByteArray()))
        assertEquals(listOf("thr"), a.flush())
        assertEquals(emptyList<String>(), a.flush())
    }

    @Test
    fun `force-splits pathological long lines`() {
        val a = LineAssembler(maxLineChars = 4)
        assertEquals(listOf("abcd", "efgh"), a.feed("abcdefghij".toByteArray()))
        assertEquals(listOf("ij"), a.flush())
    }
}
