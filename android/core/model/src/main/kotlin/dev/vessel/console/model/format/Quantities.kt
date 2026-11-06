package dev.vessel.console.model.format

/**
 * Parsers for the human-entered quantities on the create-container and GPU
 * forms. They accept the notations operators already know from Kubernetes /
 * Docker so the form feels familiar:
 *
 *  * memory: `536870912`, `512Mi`, `512M`, `1.5Gi`, `2g`, `256k`
 *  * cpu:    `500m`, `0.5`, `2`
 *
 * Empty input is valid and means "unlimited / daemon default" (0 in the proto).
 * Invalid input returns null so the form can show a field error.
 */
object Quantities {
    private val BYTE_PATTERN = Regex("""^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$""")

    private const val KI = 1L shl 10
    private const val MI = 1L shl 20
    private const val GI = 1L shl 30
    private const val TI = 1L shl 40

    private val BYTE_SUFFIXES = mapOf(
        "" to 1L, "b" to 1L,
        "k" to 1_000L, "kb" to 1_000L, "ki" to KI, "kib" to KI,
        "m" to 1_000_000L, "mb" to 1_000_000L, "mi" to MI, "mib" to MI,
        "g" to 1_000_000_000L, "gb" to 1_000_000_000L, "gi" to GI, "gib" to GI,
        "t" to 1_000_000_000_000L, "tb" to 1_000_000_000_000L, "ti" to TI, "tib" to TI,
    )

    fun parseBytes(input: String): Long? {
        if (input.isBlank()) return 0
        val match = BYTE_PATTERN.matchEntire(input) ?: return null
        val (number, suffix) = match.destructured
        val multiplier = BYTE_SUFFIXES[suffix.lowercase()] ?: return null
        val value = number.toBigDecimal().multiply(multiplier.toBigDecimal())
        return runCatching { value.setScale(0, java.math.RoundingMode.DOWN).longValueExact() }.getOrNull()
    }

    fun parseMillicores(input: String): Int? {
        val s = input.trim()
        if (s.isEmpty()) return 0
        val millicores = if (s.endsWith("m")) {
            s.dropLast(1).toLongOrNull()
        } else {
            s.toBigDecimalOrNull()?.movePointRight(3)?.let {
                runCatching { it.setScale(0, java.math.RoundingMode.DOWN).longValueExact() }.getOrNull()
            }
        } ?: return null
        return millicores.takeIf { it in 0..Int.MAX_VALUE }?.toInt()
    }

    /** Non-negative integer, empty = 0. */
    fun parseCount(input: String): Int? {
        val s = input.trim()
        if (s.isEmpty()) return 0
        return s.toIntOrNull()?.takeIf { it >= 0 }
    }

    /** cpuset list syntax: "0-3", "0,2,4", "0-1,6". Empty = inherit. */
    fun isValidCpuset(input: String): Boolean {
        val s = input.trim()
        if (s.isEmpty()) return true
        return s.split(',').all { part ->
            val range = part.split('-')
            when (range.size) {
                1 -> range[0].toIntOrNull()?.let { it >= 0 } == true
                2 -> {
                    val a = range[0].toIntOrNull()
                    val b = range[1].toIntOrNull()
                    a != null && b != null && a in 0..b
                }
                else -> false
            }
        }
    }
}

/**
 * Minimal POSIX-shell-like argv splitter for the "command" fields: handles
 * whitespace, single quotes (literal), double quotes (with \" and \\ escapes)
 * and backslash escapes outside quotes. No globbing, no variables — the
 * container runtime execs argv directly, there is no shell unless you ask for
 * one with `sh -c '…'`.
 */
object CommandLine {
    class ParseException(message: String) : IllegalArgumentException(message)

    fun split(input: String): List<String> {
        val args = mutableListOf<String>()
        val cur = StringBuilder()
        var inArg = false
        var i = 0
        while (i < input.length) {
            val c = input[i]
            when {
                c == '\'' -> {
                    val end = input.indexOf('\'', i + 1)
                    if (end < 0) throw ParseException("Unterminated single quote")
                    cur.append(input, i + 1, end)
                    inArg = true
                    i = end
                }
                c == '"' -> {
                    i++
                    while (true) {
                        if (i >= input.length) throw ParseException("Unterminated double quote")
                        val d = input[i]
                        if (d == '"') break
                        if (d == '\\' && i + 1 < input.length && input[i + 1] in "\"\\$`") {
                            cur.append(input[i + 1])
                            i += 2
                            continue
                        }
                        cur.append(d)
                        i++
                    }
                    inArg = true
                }
                c == '\\' -> {
                    if (i + 1 >= input.length) throw ParseException("Trailing backslash")
                    cur.append(input[i + 1])
                    inArg = true
                    i++
                }
                c.isWhitespace() -> if (inArg) {
                    args += cur.toString()
                    cur.clear()
                    inArg = false
                }
                else -> {
                    cur.append(c)
                    inArg = true
                }
            }
            i++
        }
        if (inArg) args += cur.toString()
        return args
    }
}
