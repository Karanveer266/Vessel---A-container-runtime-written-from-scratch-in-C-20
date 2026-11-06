package dev.vessel.console.model

import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.nio.charset.CharsetDecoder
import java.nio.charset.CodingErrorAction

/**
 * Turns arbitrary byte chunks (as delivered by `StreamLogs`) into complete
 * text lines.
 *
 * The daemon chunks container output at pipe-read boundaries, which means a
 * chunk can end in the middle of a line *and* in the middle of a multi-byte
 * UTF-8 sequence. Decoding each chunk independently would produce U+FFFD
 * garbage at every boundary; this keeps both the partial code point (inside
 * the stateful [CharsetDecoder]) and the partial line between calls.
 *
 * Lines longer than [maxLineChars] are force-split so one binary blob without
 * newlines cannot grow the pending buffer without bound.
 */
class LineAssembler(private val maxLineChars: Int = 4_096) {
    private val decoder: CharsetDecoder = Charsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPLACE)
        .onUnmappableCharacter(CodingErrorAction.REPLACE)
    private val partial = StringBuilder()
    private var leftover: ByteArray = ByteArray(0)

    /** Feeds a chunk, returning the lines it completed (without terminators). */
    fun feed(bytes: ByteArray): List<String> {
        val input = ByteBuffer.wrap(if (leftover.isEmpty()) bytes else leftover + bytes)
        val out = CharBuffer.allocate(input.remaining() + 1)
        decoder.decode(input, out, false)
        // Bytes of an incomplete trailing code point stay in `input`; keep them for next time.
        leftover = ByteArray(input.remaining()).also { input.get(it) }
        out.flip()
        return split(out)
    }

    /** Emits whatever is pending as a final line (stream ended without trailing newline). */
    fun flush(): List<String> {
        val tail = if (leftover.isNotEmpty()) "�" else ""
        leftover = ByteArray(0)
        decoder.reset()
        val line = partial.toString() + tail
        partial.clear()
        return if (line.isEmpty()) emptyList() else listOf(line)
    }

    private fun split(chars: CharBuffer): List<String> {
        val lines = mutableListOf<String>()
        while (chars.hasRemaining()) {
            val c = chars.get()
            when (c) {
                '\n' -> {
                    if (partial.endsWith('\r')) partial.setLength(partial.length - 1)
                    lines += partial.toString()
                    partial.clear()
                }
                else -> {
                    partial.append(c)
                    if (partial.length >= maxLineChars) {
                        lines += partial.toString()
                        partial.clear()
                    }
                }
            }
        }
        return lines
    }
}
