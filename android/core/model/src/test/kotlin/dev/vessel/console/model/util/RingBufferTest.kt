package dev.vessel.console.model.util

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.test.assertFailsWith

class RingBufferTest {

    @Test
    fun `fills up to capacity in insertion order`() {
        val rb = RingBuffer<Int>(3)
        rb.add(1); rb.add(2)
        assertEquals(listOf(1, 2), rb.toList())
        assertEquals(2, rb.size)
        assertFalse(rb.isFull)
    }

    @Test
    fun `overwrites oldest when full and counts evictions`() {
        val rb = RingBuffer<Int>(3)
        (1..7).forEach(rb::add)
        assertEquals(listOf(5, 6, 7), rb.toList())
        assertTrue(rb.isFull)
        assertEquals(4L, rb.evicted)
        assertEquals(5, rb[0])
        assertEquals(7, rb[2])
    }

    @Test
    fun `iterator matches toList across wrap-around`() {
        val rb = RingBuffer<String>(4)
        ('a'..'j').forEach { rb.add(it.toString()) }
        assertEquals(rb.toList(), rb.iterator().asSequence().toList())
        assertEquals(listOf("g", "h", "i", "j"), rb.toList())
    }

    @Test
    fun `memory stays bounded under sustained load`() {
        val rb = RingBuffer<Long>(2_000)
        for (i in 0 until 1_000_000L) rb.add(i)
        assertEquals(2_000, rb.size)
        assertEquals(998_000L, rb[0])
        assertEquals(999_999L, rb[1_999])
    }

    @Test
    fun `clear resets but snapshot taken before is unaffected`() {
        val rb = RingBuffer<Int>(2)
        rb.addAll(listOf(1, 2, 3))
        val snapshot = rb.toList()
        rb.clear()
        assertEquals(0, rb.size)
        assertEquals(listOf(2, 3), snapshot)
        rb.add(9)
        assertEquals(listOf(9), rb.toList())
    }

    @Test
    fun `rejects bad capacity and out-of-range index`() {
        assertFailsWith<IllegalArgumentException> { RingBuffer<Int>(0) }
        val rb = RingBuffer<Int>(2).apply { add(1) }
        assertFailsWith<IndexOutOfBoundsException> { rb[1] }
    }
}

class FloatRingBufferTest {

    @Test
    fun `snapshot is ordered oldest-first across the wrap point`() {
        val rb = FloatRingBuffer(4)
        (1..6).forEach { rb.add(it.toFloat()) }
        assertArrayEquals(floatArrayOf(3f, 4f, 5f, 6f), rb.toFloatArray(), 0f)
        assertEquals(6f, rb.last())
    }

    @Test
    fun `snapshot is a defensive copy`() {
        val rb = FloatRingBuffer(2)
        rb.add(1f)
        val snap = rb.snapshot()
        rb.add(2f); rb.add(3f)
        assertEquals(1, snap.size)
        assertEquals(1f, snap[0])
    }

    @Test
    fun `empty buffer has no last value`() {
        assertEquals(null, FloatRingBuffer(3).last())
        assertEquals(0, FloatRingBuffer(3).toFloatArray().size)
    }

    @Test
    fun `Series equality is by content so StateFlow can dedupe`() {
        assertEquals(Series.of(1f, 2f), Series.of(1f, 2f))
        assertEquals(Series.of(1f, 2f).hashCode(), Series.of(1f, 2f).hashCode())
        assertFalse(Series.of(1f) == Series.of(2f))
        assertEquals(2f, Series.of(1f, 2f, 0.5f).max())
        assertEquals(0f, Series.Empty.max())
    }
}
