package dev.vessel.console.model.util

/**
 * Immutable snapshot of a numeric time series for rendering.
 *
 * Wraps a defensively-owned [FloatArray] so the type is safe to put in UI state:
 * nobody can mutate it after publication, and equality is by content, so a
 * StateFlow will not re-emit (and Compose will not redraw) identical data.
 */
class Series private constructor(private val values: FloatArray) {
    val size: Int get() = values.size
    operator fun get(i: Int): Float = values[i]
    fun lastOrNull(): Float? = if (values.isEmpty()) null else values[values.size - 1]
    fun isEmpty(): Boolean = values.isEmpty()

    fun max(): Float = if (values.isEmpty()) 0f else values.max()
    fun min(): Float = if (values.isEmpty()) 0f else values.min()

    override fun equals(other: Any?): Boolean = other is Series && values.contentEquals(other.values)
    override fun hashCode(): Int = values.contentHashCode()
    override fun toString(): String = "Series(${values.joinToString(limit = 8)})"

    companion object {
        val Empty = Series(FloatArray(0))

        /** Takes ownership of [values]; callers must pass a fresh copy (e.g. from a ring buffer snapshot). */
        fun wrap(values: FloatArray): Series = if (values.isEmpty()) Empty else Series(values)

        fun of(vararg values: Float): Series = wrap(values.copyOf())
    }
}

fun FloatRingBuffer.snapshot(): Series = Series.wrap(toFloatArray())
