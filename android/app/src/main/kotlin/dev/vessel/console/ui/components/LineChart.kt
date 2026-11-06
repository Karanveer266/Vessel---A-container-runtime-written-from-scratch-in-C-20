package dev.vessel.console.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.background
import dev.vessel.console.model.util.Series
import dev.vessel.console.ui.theme.VesselTheme

/** One plotted line. [dashed] is used for limits so identity never relies on colour alone. */
@Immutable
data class ChartSeries(
    val label: String,
    val values: Series,
    val color: Color,
    val dashed: Boolean = false,
)

/**
 * Hand-rolled strip chart on [Canvas] — no third-party chart library.
 *
 * Why custom: the data is a fixed-size rolling window updated at 1 Hz, so the
 * job is tiny (polyline + 5 gridlines + labels) and a library would add
 * 100s of KB and its own recomposition model. Here:
 *  * input is an immutable [Series] snapshot from a ring buffer, so the chart
 *    redraws only when data actually changed (content equality);
 *  * the draw phase allocates one [Path] per series and nothing per point;
 *  * the y-axis maximum snaps to a "nice" ceiling ([ChartMath.niceCeiling]) so
 *    gridlines don't jump on every sample;
 *  * a touch/drag crosshair shows the exact values at that sample (the
 *    "hover layer"), and a legend is shown for ≥ 2 series;
 *  * one y-axis only — measures of different scale go in separate charts.
 */
@Composable
fun LineChart(
    series: List<ChartSeries>,
    capacity: Int,
    formatY: (Float) -> String,
    modifier: Modifier = Modifier,
    height: Dp = 140.dp,
    /** Lower bound for the y-axis max so an idle container's chart isn't all noise. */
    minYMax: Float = 1f,
    /** Optional fixed y max (e.g. 100 for PSI percentages). */
    fixedYMax: Float? = null,
    summary: String = "",
) {
    val colors = VesselTheme.chartColors
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = MaterialTheme.typography.labelSmall.copy(color = colors.axisText)
    var selected by remember { mutableIntStateOf(-1) }

    val n = series.maxOfOrNull { it.values.size } ?: 0
    val dataMax = series.maxOfOrNull { it.values.max() } ?: 0f
    val yMax = fixedYMax ?: ChartMath.niceCeiling(dataMax, minYMax)

    Column(modifier) {
        if (series.size >= 2) Legend(series)
        Box(
            Modifier
                .fillMaxWidth()
                .height(height)
                .semantics { contentDescription = summary },
        ) {
            Canvas(
                Modifier
                    .matchParentSize()
                    .pointerInput(n, capacity) {
                        val plotWidth = size.width - AXIS_GUTTER_PX
                        detectTapGestures(
                            onPress = {
                                selected = ChartMath.indexAt(it.x - AXIS_GUTTER_PX, n, capacity, plotWidth)
                                tryAwaitRelease()
                                selected = -1
                            },
                        )
                    }
                    .pointerInput(n, capacity) {
                        val plotWidth = size.width - AXIS_GUTTER_PX
                        detectDragGestures(
                            onDragEnd = { selected = -1 },
                            onDragCancel = { selected = -1 },
                        ) { change, _ ->
                            selected = ChartMath.indexAt(change.position.x - AXIS_GUTTER_PX, n, capacity, plotWidth)
                        }
                    },
            ) {
                val plotLeft = AXIS_GUTTER_PX
                val plotWidth = size.width - plotLeft
                val plotHeight = size.height - LABEL_PAD_PX
                drawGrid(yMax, plotLeft, plotWidth, plotHeight, colors.grid, textMeasurer, labelStyle, formatY)
                series.forEach { s -> drawSeries(s, capacity, yMax, plotLeft, plotWidth, plotHeight) }
                if (selected in 0 until n) {
                    drawCrosshair(series, selected, n, capacity, yMax, plotLeft, plotWidth, plotHeight, colors.axisText)
                }
            }
            if (selected in 0 until n) {
                Tooltip(series, selected, formatY, Modifier.align(Alignment.TopEnd).padding(4.dp))
            }
        }
    }
}

private const val AXIS_GUTTER_PX = 96f
private const val LABEL_PAD_PX = 8f

private fun DrawScope.drawGrid(
    yMax: Float,
    left: Float,
    width: Float,
    height: Float,
    gridColor: Color,
    textMeasurer: TextMeasurer,
    style: androidx.compose.ui.text.TextStyle,
    formatY: (Float) -> String,
) {
    // Recessive grid: hairlines, no vertical lines, labels in muted ink.
    for (value in ChartMath.gridValues(yMax)) {
        val y = ChartMath.yFor(value, yMax, height) + LABEL_PAD_PX / 2
        drawLine(gridColor, Offset(left, y), Offset(left + width, y), strokeWidth = 1f)
        if (value == 0f || value == yMax || value == yMax / 2) {
            val layout = textMeasurer.measure(formatY(value), style)
            drawText(layout, topLeft = Offset(0f, (y - layout.size.height / 2f).coerceIn(0f, size.height - layout.size.height)))
        }
    }
}

private fun DrawScope.drawSeries(s: ChartSeries, capacity: Int, yMax: Float, left: Float, width: Float, height: Float) {
    val values = s.values
    if (values.isEmpty()) return
    val path = Path()
    for (i in 0 until values.size) {
        val x = left + ChartMath.xFor(i, values.size, capacity, width)
        val y = ChartMath.yFor(values[i], yMax, height) + LABEL_PAD_PX / 2
        if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
    }
    drawPath(
        path,
        color = s.color,
        style = Stroke(
            width = 2.dp.toPx(),
            cap = StrokeCap.Round,
            join = StrokeJoin.Round,
            pathEffect = if (s.dashed) PathEffect.dashPathEffect(floatArrayOf(10f, 8f)) else null,
        ),
    )
    // End-point marker on the newest sample makes "now" obvious.
    if (!s.dashed) {
        val lastX = left + ChartMath.xFor(values.size - 1, values.size, capacity, width)
        val lastY = ChartMath.yFor(values[values.size - 1], yMax, height) + LABEL_PAD_PX / 2
        drawCircle(s.color, radius = 3.dp.toPx(), center = Offset(lastX, lastY))
    }
}

private fun DrawScope.drawCrosshair(
    series: List<ChartSeries>,
    index: Int,
    n: Int,
    capacity: Int,
    yMax: Float,
    left: Float,
    width: Float,
    height: Float,
    color: Color,
) {
    val x = left + ChartMath.xFor(index, n, capacity, width)
    drawLine(color.copy(alpha = 0.6f), Offset(x, 0f), Offset(x, size.height), strokeWidth = 1.dp.toPx())
    series.forEach { s ->
        // Series may differ in length right after a reset; align on the newest sample.
        val i = index - (n - s.values.size)
        if (i in 0 until s.values.size) {
            val y = ChartMath.yFor(s.values[i], yMax, height) + LABEL_PAD_PX / 2
            drawCircle(Color.White, radius = 5.dp.toPx(), center = Offset(x, y))
            drawCircle(s.color, radius = 4.dp.toPx(), center = Offset(x, y))
        }
    }
}

@Composable
private fun Tooltip(series: List<ChartSeries>, index: Int, formatY: (Float) -> String, modifier: Modifier) {
    val n = series.maxOf { it.values.size }
    Column(
        modifier
            .clip(RoundedCornerShape(6.dp))
            .background(MaterialTheme.colorScheme.inverseSurface.copy(alpha = 0.92f))
            .padding(horizontal = 8.dp, vertical = 4.dp),
    ) {
        Text(
            "${n - 1 - index}s ago",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.inverseOnSurface,
        )
        series.forEach { s ->
            val i = index - (n - s.values.size)
            if (i in 0 until s.values.size) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Swatch(s)
                    Spacer(Modifier.width(4.dp))
                    Text(
                        "${s.label}: ${formatY(s.values[i])}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.inverseOnSurface,
                    )
                }
            }
        }
    }
}

@Composable
private fun Legend(series: List<ChartSeries>) {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.padding(bottom = 4.dp)) {
        series.forEach { s ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Swatch(s)
                Spacer(Modifier.width(4.dp))
                Text(s.label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun Swatch(s: ChartSeries) {
    Canvas(Modifier.size(width = 14.dp, height = 8.dp)) {
        drawLine(
            s.color,
            Offset(0f, size.height / 2),
            Offset(size.width, size.height / 2),
            strokeWidth = 2.dp.toPx(),
            pathEffect = if (s.dashed) PathEffect.dashPathEffect(floatArrayOf(6f, 4f)) else null,
        )
    }
}

/**
 * Minimal sparkline for list rows: one line, no axes, no interaction.
 * Scales to its own max (with [minYMax] floor) — it shows trend, not magnitude;
 * the exact value is printed next to it.
 */
@Composable
fun Sparkline(
    values: Series,
    capacity: Int,
    color: Color,
    modifier: Modifier = Modifier,
    minYMax: Float = 1f,
) {
    Canvas(modifier) {
        if (values.isEmpty()) return@Canvas
        val yMax = maxOf(values.max(), minYMax) * 1.1f
        val path = Path()
        for (i in 0 until values.size) {
            val x = ChartMath.xFor(i, values.size, capacity, size.width)
            val y = ChartMath.yFor(values[i], yMax, size.height)
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, color, style = Stroke(width = 1.5.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
    }
}

/** Horizontal bar with a rounded data-end, anchored at the start (baseline). */
@Composable
fun HBar(fraction: Float, color: Color, modifier: Modifier = Modifier, track: Color = Color.Transparent) {
    Canvas(modifier) {
        if (track != Color.Transparent) {
            drawRoundRect(track, size = size, cornerRadius = CornerRadius(4.dp.toPx()))
        }
        val w = size.width * fraction.coerceIn(0f, 1f)
        if (w > 0f) {
            drawRoundRect(color, size = Size(w, size.height), cornerRadius = CornerRadius(4.dp.toPx()))
        }
    }
}
