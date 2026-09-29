import { useEffect, useMemo, useRef } from "react";
import { boneR } from "./colormaps";
import { HoverTooltip } from "./HoverTooltip";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { nearestIndex, useCanvasHover } from "./useCanvasHover";
import { useContainerWidth } from "./useContainerWidth";
import { evenTicks, tickDecimals, useZoom, visibleCells, type PlotRect } from "./useZoom";
import { ZoomSelection } from "./ZoomOverlay";
import type { Tip } from "./tips";
import { num } from "./viz/format";
import { vizPalette } from "./viz/palette";

// A record's spectra as its saved figure draws them (sigpipe's plot_trace_spectra): each trace's
// amplitude spectrum at its receiver along the line, frequency up, bone reversed (white:
// nothing, black: the trace's largest), the whole of it, 0 to Nyquist; a band's bounds dashed.
// The wheel zooms; a drag does what its box's tools say.

export interface TraceSpectra {
  freqs: number[];
  /** Each trace's, in the record's order (traces x freqs), 0 to 1 of its largest. */
  amplitude: number[][];
}

const ML = 60, MR = 76, MT = 16, MB = 38;
const BASE_W = 716; // the drawing's width until its card is measured
const FONT = CANVAS_FONT;
const LEGEND_W = 14;

export function SpectrumCanvas({
  spectra,
  positions,
  band = null,
}: {
  spectra: TraceSpectra;
  /** Each trace's receiver along the line, m, in the record's order. */
  positions: number[];
  /** The band a filter keeps, Hz: its bounds dashed; none, no filter. */
  band?: [number, number] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const palette = canvasPalette(theme);
  const pass = vizPalette(theme).status.pass;
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const TOTAL_W = Math.max(420, Math.round(containerWidth || BASE_W));
  const PLOT_W = TOTAL_W - ML - MR;
  const PLOT_H = Math.round(Math.min(420, Math.max(260, PLOT_W * 0.45)));
  const TOTAL_H = MT + PLOT_H + MB;
  const PLOTS: PlotRect[] = [{ left: ML, top: MT, width: PLOT_W, height: PLOT_H, xAxis: MB, yAxis: ML }];
  const { pos: hoverPos, onMouseMove, onMouseLeave } = useCanvasHover(1);

  // The traces along the line, each a column reaching halfway to its neighbours.
  const columns = useMemo(() => {
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

  const zoom = useZoom({
    extent: { x: [xMin, xMax], y: [fMin, fMax] },
    plots: PLOTS,
    width: TOTAL_W,
    height: TOTAL_H,
    canvas: canvasRef,
  });
  const [x0, x1] = zoom.view.x;
  const [f0, f1] = zoom.view.y;

  // The image once per record: a column a trace, in their order along the line, the highest
  // frequency on top.
  const image = useMemo(() => {
    const nf = spectra.freqs.length;
    const off = document.createElement("canvas");
    off.width = Math.max(1, columns.length);
    off.height = Math.max(1, nf);
    const octx = off.getContext("2d");
    if (!octx || nf === 0 || columns.length === 0) return off;
    const data = octx.createImageData(columns.length, nf);
    columns.forEach(({ trace }, c) => {
      const row = spectra.amplitude[trace] ?? [];
      for (let j = 0; j < nf; j++) {
        const [r, g, b] = boneR(row[j] ?? 0);
        const idx = ((nf - 1 - j) * columns.length + c) * 4;
        data.data[idx] = r;
        data.data[idx + 1] = g;
        data.data[idx + 2] = b;
        data.data[idx + 3] = 255;
      }
    });
    octx.putImageData(data, 0, 0);
    return off;
  }, [spectra, columns]);

  // The colour bar's image, 0 at the bottom: drawn whole, no seams between rows.
  const legend = useMemo(() => {
    const off = document.createElement("canvas");
    off.width = 1;
    off.height = 256;
    const octx = off.getContext("2d");
    if (!octx) return off;
    const data = octx.createImageData(1, 256);
    for (let py = 0; py < 256; py++) {
      const [r, g, b] = boneR(1 - py / 255);
      data.data.set([r, g, b, 255], py * 4);
    }
    octx.putImageData(data, 0, 0);
    return off;
  }, []);

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

    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, MT, PLOT_W, PLOT_H);
    ctx.clip();
    ctx.imageSmoothingEnabled = false;
    // The frequencies on show, then each trace's column on show.
    const nf = image.height;
    const [r0, r1] = visibleCells(nf, fMax, fMin, f0, f1);
    const fEdge = (r: number) => fMax - (r / nf) * (fMax - fMin);
    if (r1 > r0) {
      columns.forEach(({ left, right }, c) => {
        if (right < x0 || left > x1) return;
        const xa = X(left), xb = X(right);
        const ya = Y(fEdge(r0)), yb = Y(fEdge(r1));
        ctx.drawImage(image, c, r0, 1, r1 - r0, xa, ya, xb - xa, yb - ya);
      });
    }
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

    // The colour bar: 0 to each trace's largest.
    const legendX = ML + PLOT_W + 12;
    ctx.drawImage(legend, 0, 0, 1, legend.height, legendX, MT, LEGEND_W, PLOT_H);
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, MT, LEGEND_W, PLOT_H);
    ctx.font = FONT;
    ctx.fillStyle = palette.tick;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const t of [0, 0.5, 1]) ctx.fillText(String(t), legendX + LEGEND_W + 5, MT + PLOT_H - t * PLOT_H);

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
    const fDecimals = tickDecimals((f1 - f0) / 6);
    for (const f of evenTicks(f0, f1, 6)) {
      ctx.beginPath();
      ctx.moveTo(ML - 4, Y(f));
      ctx.lineTo(ML, Y(f));
      ctx.stroke();
      ctx.fillText(f.toFixed(fDecimals), ML - 7, Y(f));
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xDecimals = tickDecimals((x1 - x0) / 6);
    for (const x of evenTicks(x0, x1, 6)) {
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
    ctx.save();
    ctx.translate(TOTAL_W - 12, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Amplitude (of its largest)", 0, 0);
    ctx.restore();
  }, [image, legend, columns, band, pass, palette, fMin, fMax, x0, x1, f0, f1, TOTAL_W, TOTAL_H, PLOT_W, PLOT_H]);

  // Over the plot: the receiver under the pointer, the frequency, its amplitude there; whether
  // the filter cuts it.
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
      notes: cut ? ["cut by the filter"] : [],
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
