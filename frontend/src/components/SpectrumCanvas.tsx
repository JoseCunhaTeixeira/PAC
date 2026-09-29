import { useEffect, useMemo, useRef, useState } from "react";
import { bone, boneR } from "./colormaps";
import { HoverTooltip } from "./HoverTooltip";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { nearestIndex, useCanvasHover } from "./useCanvasHover";
import { useContainerWidth } from "./useContainerWidth";
import { niceTicks, tickDecimals, useZoom, type PlotRect, type Range } from "./useZoom";
import { ZoomSelection } from "./ZoomOverlay";
import type { Tip } from "./tips";
import { num } from "./viz/format";
import { alongLine } from "./viz/line";
import { vizPalette } from "./viz/palette";

// A record's spectra as its saved figure draws them (sigpipe's plot_trace_spectra): each trace's
// amplitude spectrum at its receiver along the line, frequency up, in bone (reversed on a light
// page, white nothing and black the trace's largest, as the figure; on a dark one, black
// nothing), the whole of it, 0 to Nyquist; a band's bounds dashed.
// The wheel zooms; a drag does what its box's tools say. Given a zoom along the line, it follows
// the plots above it (a gather's), its margins theirs: its colour bar above it.

export interface TraceSpectra {
  freqs: number[];
  /** Each trace's, in the record's order (traces x freqs), 0 to 1 of its largest. */
  amplitude: number[][];
}

// A gather's margins (LineGather), the colour bar in the top one.
const ML = 84, MR = 16, MT = 30, MB = 46;
const BASE_W = 716; // the drawing's width until its card is measured
const FONT = CANVAS_FONT;
const LEGEND_W = 120, LEGEND_H = 8;

interface Column {
  trace: number;
  x: number;
  left: number;
  right: number;
}

/** The first index of ascending `values` at or above `value`. */
function lowerBound(values: number[], value: number): number {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** The spectra on show, put device pixel by device pixel in the plot (its top left at ML, MT):
 * each pixel column its trace's, each row the largest of the frequencies it covers (a narrow
 * peak kept, as the saved figure's groups keep it). No seams between the traces, no frequency
 * skipped: drawn scaled, a column a trace, they showed as a grid. */
function putSpectra(
  ctx: CanvasRenderingContext2D,
  spectra: TraceSpectra,
  columns: Column[],
  [x0, x1]: Range,
  [f0, f1]: Range,
  dpr: number,
  plotW: number,
  plotH: number,
  colour: (t: number) => [number, number, number],
) {
  const W = Math.max(1, Math.round(plotW * dpr));
  const H = Math.max(1, Math.round(plotH * dpr));
  const { freqs } = spectra;
  // Each device row's frequencies, as indices [from, to] into freqs; none past the spectra.
  const rows: [number, number][] = [];
  for (let py = 0; py < H; py++) {
    const top = f1 - (py / H) * (f1 - f0);
    const bottom = f1 - ((py + 1) / H) * (f1 - f0);
    let from = lowerBound(freqs, bottom);
    let to = lowerBound(freqs, top) - 1;
    if (to < from) {
      const middle = (top + bottom) / 2;
      from = to = middle < freqs[0] || middle > freqs[freqs.length - 1] ? -1 : nearestIndex(freqs, middle);
    }
    rows.push([from, to]);
  }
  // Each trace on show, its colours down the rows, once.
  const colours = new Map<number, Uint8ClampedArray>();
  const colourOf = (trace: number) => {
    const found = colours.get(trace);
    if (found) return found;
    const amplitude = spectra.amplitude[trace] ?? [];
    const rgb = new Uint8ClampedArray(H * 4);
    rows.forEach(([from, to], py) => {
      if (from < 0) return;
      let most = 0;
      for (let j = from; j <= to; j++) most = Math.max(most, amplitude[j] ?? 0);
      const [r, g, b] = colour(most);
      rgb.set([r, g, b, 255], py * 4);
    });
    colours.set(trace, rgb);
    return rgb;
  };
  const image = ctx.createImageData(W, H);
  const data = image.data;
  let c = 0;
  for (let px = 0; px < W; px++) {
    const x = x0 + ((px + 0.5) / W) * (x1 - x0);
    while (c < columns.length - 1 && columns[c].right < x) c++;
    const column = columns[c];
    if (!column || x < column.left || x > column.right) continue;
    const rgb = colourOf(column.trace);
    for (let py = 0, from = 0, to = px * 4; py < H; py++, from += 4, to += W * 4) {
      data[to] = rgb[from];
      data[to + 1] = rgb[from + 1];
      data[to + 2] = rgb[from + 2];
      data[to + 3] = rgb[from + 3];
    }
  }
  ctx.putImageData(image, Math.round(ML * dpr), Math.round(MT * dpr));
}

export function SpectrumCanvas({
  spectra,
  positions,
  band = null,
  outside = "cut by the filter",
  extent,
  xZoom = null,
  onXZoom,
}: {
  spectra: TraceSpectra;
  /** Each trace's receiver along the line, m, in the record's order. */
  positions: number[];
  /** The band dashed, Hz (a filter's cuts, a record's usable band); none, none drawn. */
  band?: [number, number] | null;
  /** What the hover says outside the band. */
  outside?: string;
  /** The line's extent along x, as the plots it follows have it; else its traces'. */
  extent?: Range;
  /** The zoom along the line it shares with them; null for the whole line. */
  xZoom?: Range | null;
  onXZoom?: (x: Range | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const palette = canvasPalette(theme);
  const pass = vizPalette(theme).status.pass;
  const colour = theme === "dark" ? bone : boneR;
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const TOTAL_W = Math.max(420, Math.round(containerWidth || BASE_W));
  const PLOT_W = TOTAL_W - ML - MR;
  const PLOT_H = Math.round(Math.min(420, Math.max(260, PLOT_W * 0.45)));
  const TOTAL_H = MT + PLOT_H + MB;
  const PLOTS: PlotRect[] = [{ left: ML, top: MT, width: PLOT_W, height: PLOT_H, xAxis: MB, yAxis: ML }];
  const { pos: hoverPos, onMouseMove, onMouseLeave } = useCanvasHover(1);

  // The traces along the line, each a column reaching halfway to its neighbours.
  const columns = useMemo((): Column[] => {
    const at = spectra.amplitude.map((_, i) => positions[i] ?? i);
    const order = at.map((_, i) => i).sort((a, b) => at[a] - at[b]);
    return order.map((trace, k) => {
      const x = at[trace];
      const before = k > 0 ? at[order[k - 1]] : null;
      const after = k < order.length - 1 ? at[order[k + 1]] : null;
      const half = (before !== null ? x - before : after !== null ? after - x : 2) / 2;
      const halfAfter = (after !== null ? after - x : before !== null ? x - before : 2) / 2;
      return { trace, x, left: x - half, right: x + halfAfter };
    });
  }, [spectra, positions]);

  const fMin = spectra.freqs[0] ?? 0;
  const fMax = spectra.freqs[spectra.freqs.length - 1] ?? 1;
  const xMin = columns[0]?.left ?? 0;
  const xMax = columns[columns.length - 1]?.right ?? 1;
  const full = useMemo(
    () => ({ x: extent ?? ([xMin, xMax] as Range), y: [fMin, fMax] as Range }),
    [extent, xMin, xMax, fMin, fMax],
  );
  // The frequencies, its own zoom; along the line, the plots' it follows when given.
  const [ownF, setOwnF] = useState<Range | null>(null);

  const zoom = useZoom({
    extent: full,
    plots: PLOTS,
    width: TOTAL_W,
    height: TOTAL_H,
    link: onXZoom ? alongLine(xZoom, onXZoom, ownF, setOwnF, full) : undefined,
    canvas: canvasRef,
  });
  const [x0, x1] = zoom.view.x;
  const [f0, f1] = zoom.view.y;

  // The colour bar's image, 0 on the left: drawn whole, no seams between columns.
  const legend = useMemo(() => {
    const off = document.createElement("canvas");
    off.width = 256;
    off.height = 1;
    const octx = off.getContext("2d");
    if (!octx) return off;
    const data = octx.createImageData(256, 1);
    for (let px = 0; px < 256; px++) {
      const [r, g, b] = colour(px / 255);
      data.data.set([r, g, b, 255], px * 4);
    }
    octx.putImageData(data, 0, 0);
    return off;
  }, [colour]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(TOTAL_W * dpr);
    canvas.height = Math.round(TOTAL_H * dpr);
    canvas.style.width = TOTAL_W + "px";
    canvas.style.height = TOTAL_H + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, TOTAL_W, TOTAL_H);
    const X = (x: number) => ML + ((x - x0) / (x1 - x0)) * PLOT_W;
    const Y = (f: number) => MT + PLOT_H - ((f - f0) / (f1 - f0)) * PLOT_H;
    putSpectra(ctx, spectra, columns, [x0, x1], [f0, f1], dpr, PLOT_W, PLOT_H, colour);

    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, MT, PLOT_W, PLOT_H);
    ctx.clip();
    if (band) {
      ctx.strokeStyle = pass;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      for (const f of band) {
        ctx.beginPath();
        ctx.moveTo(ML, Y(f));
        ctx.lineTo(ML + PLOT_W, Y(f));
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    ctx.restore();

    // The colour bar above the plot, on its right: 0 to each trace's largest.
    const legendX = ML + PLOT_W - LEGEND_W - 12;
    const legendY = (MT - LEGEND_H) / 2 - 2;
    ctx.drawImage(legend, 0, 0, legend.width, 1, legendX, legendY, LEGEND_W, LEGEND_H);
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, LEGEND_W, LEGEND_H);
    ctx.font = FONT;
    ctx.textBaseline = "middle";
    ctx.fillStyle = palette.tick;
    ctx.textAlign = "right";
    ctx.fillText("0", legendX - 5, legendY + LEGEND_H / 2);
    ctx.textAlign = "left";
    ctx.fillText("1", legendX + LEGEND_W + 5, legendY + LEGEND_H / 2);
    ctx.fillStyle = palette.title;
    ctx.textAlign = "right";
    ctx.fillText("Amplitude (normalized per trace)", legendX - 18, legendY + LEGEND_H / 2);

    // Axes.
    ctx.strokeStyle = palette.axis;
    ctx.beginPath();
    ctx.moveTo(ML, MT);
    ctx.lineTo(ML, MT + PLOT_H);
    ctx.lineTo(ML + PLOT_W, MT + PLOT_H);
    ctx.stroke();
    ctx.fillStyle = palette.tick;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const fTicks = niceTicks(f0, f1, Math.max(3, Math.floor(PLOT_H / 55)));
    const fDecimals = tickDecimals(fTicks.length > 1 ? fTicks[1] - fTicks[0] : f1 - f0);
    for (const f of fTicks) {
      ctx.beginPath();
      ctx.moveTo(ML - 4, Y(f));
      ctx.lineTo(ML, Y(f));
      ctx.stroke();
      ctx.fillText(f.toFixed(fDecimals), ML - 7, Y(f));
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xTicks = niceTicks(x0, x1, Math.max(3, Math.floor(PLOT_W / 90)));
    const xDecimals = tickDecimals(xTicks.length > 1 ? xTicks[1] - xTicks[0] : x1 - x0);
    for (const x of xTicks) {
      ctx.beginPath();
      ctx.moveTo(X(x), MT + PLOT_H);
      ctx.lineTo(X(x), MT + PLOT_H + 4);
      ctx.stroke();
      ctx.fillText(x.toFixed(xDecimals), X(x), MT + PLOT_H + 6);
    }
    ctx.fillStyle = palette.title;
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Position (m)", ML + PLOT_W / 2, TOTAL_H - 4);
    ctx.save();
    ctx.translate(16, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Frequency (Hz)", 0, 0);
    ctx.restore();
  }, [spectra, legend, colour, columns, band, pass, palette, x0, x1, f0, f1, TOTAL_W, TOTAL_H, PLOT_W, PLOT_H]);

  // Over the plot: the receiver under the pointer, the frequency, its amplitude there; whether
  // it lies outside the band.
  const tip = ((): Tip | null => {
    if (!hoverPos) return null;
    const { x, y } = hoverPos;
    if (x < ML || x > ML + PLOT_W || y < MT || y > MT + PLOT_H) return null;
    const at = x0 + ((x - ML) / PLOT_W) * (x1 - x0);
    const f = f0 + ((MT + PLOT_H - y) / PLOT_H) * (f1 - f0);
    const column = columns.find(({ left, right }) => at >= left && at <= right);
    if (!column) return null;
    const value = spectra.amplitude[column.trace]?.[nearestIndex(spectra.freqs, f)];
    const cut = band !== null && (f < band[0] || f > band[1]);
    return {
      title: `Receiver ${column.trace + 1}`,
      values: `${num(column.x, 4)} m; ${f.toFixed(1)} Hz${value !== undefined ? `; ${value.toFixed(2)}` : ""}`,
      notes: cut ? [outside] : [],
    };
  })();

  return (
    <div ref={containerRef} style={{ width: "100%", position: "relative" }}>
      <canvas
        ref={canvasRef}
        style={{ cursor: zoom.cursorAt(hoverPos), touchAction: "none", display: "block" }}
        onMouseDown={zoom.onMouseDown}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
      />
      <ZoomSelection box={zoom.selection} />
      {tip && hoverPos && <HoverTooltip x={hoverPos.x} y={hoverPos.y} tip={tip} />}
    </div>
  );
}
