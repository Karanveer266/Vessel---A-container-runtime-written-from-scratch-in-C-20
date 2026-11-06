package dev.vessel.console.model.util

/**
 * Fixed-capacity FIFO that overwrites the oldest element when full.
 *
 * Used for the live trace list: a busy container can emit tens of thousands of
 * syscalls per second, and the UI must keep memory flat no matter how long the
 * screen stays open. O(1) add, O(n) snapshot, zero allocation per add.
 *
 * Not thread-safe by design: each buffer is owned by exactly one coroutine
 * (the ViewModel's collector), which publishes immutable snapshots.
 */
class RingBuffer<T : Any>(val capacity: Int) : Iterable<T> {
    init {
        require(capacity > 0) { "capacity must be > 0, was $capacity" }
    }

    private val items = arrayOfNulls<Any>(capacity)
    private var head = 0 // index of the oldest element
    var size: Int = 0
        private set

    /** Total elements ever evicted because the buffer was full. */
    var evicted: Long = 0
        private set

    val isFull: Boolean get() = size == capacity

    fun add(item: T) {
        val tail = (head + size) % capacity
        items[tail] = item
        if (size < capacity) {
            size++
        } else {
            head = (head + 1) % capacity
            evicted++
        }
    }

    fun addAll(batch: Iterable<T>) = batch.forEach(::add)

    /** i = 0 is the oldest element. */
    @Suppress("UNCHECKED_CAST")
    operator fun get(i: Int): T {
        if (i !in 0 until size) throw IndexOutOfBoundsException("index $i, size $size")
        return items[(head + i) % capacity] as T
    }

    fun clear() {
        items.fill(null)
        head = 0
        size = 0
    }

    /** Immutable copy, oldest first. */
    fun toList(): List<T> = List(size) { get(it) }

    override fun iterator(): Iterator<T> = object : Iterator<T> {
        private var i = 0
        override fun hasNext() = i < size
        override fun next(): T = get(i++)
    }
}

/**
 * Primitive-float specialization for chart series: avoids boxing ~4 bytes into
 * a 16-byte `java.lang.Float` per sample, which matters at 1 Hz × 5 series ×
 * N containers on a phone.
 */
class FloatRingBuffer(val capacity: Int) {
    init {
        require(capacity > 0) { "capacity must be > 0, was $capacity" }
    }

    private val values = FloatArray(capacity)
    private var head = 0
    var size: Int = 0
        private set

    fun add(value: Float) {
        values[(head + size) % capacity] = value
        if (size < capacity) size++ else head = (head + 1) % capacity
    }

    operator fun get(i: Int): Float {
        if (i !in 0 until size) throw IndexOutOfBoundsException("index $i, size $size")
        return values[(head + i) % capacity]
    }

    fun last(): Float? = if (size == 0) null else get(size - 1)

    fun clear() {
        head = 0
        size = 0
    }

    /** Copy-out snapshot, oldest first. The returned array is never mutated by the buffer. */
    fun toFloatArray(): FloatArray {
        val out = FloatArray(size)
        val firstChunk = minOf(size, capacity - head)
        values.copyInto(out, 0, head, head + firstChunk)
        if (firstChunk < size) values.copyInto(out, firstChunk, 0, size - firstChunk)
        return out
    }
}
