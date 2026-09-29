import { useEffect, useMemo, useRef, useState } from "react";
import { CANVAS_FONT, canvasPalette, useTheme } from "../../theme";
import { useContainerWidth } from "../useContainerWidth";
import { tickDecimals, useZoom, type PlotRect, type Range } from "../useZoom";
import { ZoomSelection } from "../ZoomOverlay";
import { TooltipLines } from "../HoverTooltip";
import { num } from "./format";
import { alongLine } from "./line";

// One plot of lines, points and shaded areas against two axes, sized to its column: the plots
// of a selected window's card (its Vs against depth, its curve against the model's). Zooms like
// every plot of PAC; its hover reads the nearest point.

export interface PlotSeries {
  label: string;
  color: string;
  points: [number, number][];
  dash?: number[];
  width?: number;
  /** Join the points (the default), and draw a dot at each, with its error bar along y. */
  line?: boolean;
  dots?: boolean;
  errors?: number[];
  /** Left out of the hover's read-out (a step drawing's corners). */
  quiet?: boolean;
}

export interface PlotArea {
  color: string;
  polygon: [number, number][];
}

export interface PlotRef {
  axis: "x" | "y";
  at: number;
  label: string;
  color: string;
  dash?: number[];
}

const ML = 54, MR = 12, MT = 10, MB = 38;

export function LinePlot({
  series,
  areas = [],
  refs = [],
  xLabel,
  yLabel,
  yDown = false,
  height = 300,
  xRange,
  yRange,
  resetKey,
  minWidth = 240,
  depthLink,
  yAxis = true,
}: {
  series: PlotSeries[];
  areas?: PlotArea[];
  refs?: PlotRef[];
  xLabel: string;
  yLabel: string;
  /** y grows downward (depth). */
  yDown?: boolean;
  height?: number;
  xRange?: Range;
  yRange?: Range;
  resetKey?: unknown;
  /** The narrowest the drawing gets, in CSS pixels. */
  minWidth?: number;
  /** A zoom along y shared with plots beside it (null: all of it); x the plot's own. */
  depthLink?: { y: Range | null; setY: (y: Range | null) => void };
  /** Its y axis drawn (ticks, title); without, a plot beside another sharing it. */
  yAxis?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const palette = canvasPalette(theme);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const width = Math.max(minWidth, Math.floor(containerWidth || 360));
  const ml = yAxis ? ML : 6;
  const plotW = width - ml - MR;
  const plotH = height - MT - MB;
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null);

  const extent = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const one of series) {
      one.points.forEach(([x, y], i) => {
        xs.push(x);
        ys.push(y);
        const error = one.errors?.[i];
        if (error != null) ys.push(y - error, y + error);
      });
    }
    for (const area of areas) for (const [x, y] of area.polygon) {
      xs.push(x);
      ys.push(y);
    }
    for (const ref of refs) (ref.axis === "x" ? xs : ys).push(ref.at);
    const fit = (values: number[], given?: Range): [number, number] => {
      if (given) return [given[0], given[1]];
      const finite = values.filter(Number.isFinite);
      if (finite.length === 0) return [0, 1];
      const lo = Math.min(...finite);
      const hi = Math.max(...finite);
      const pad = (hi - lo) * 0.06 || Math.abs(hi) * 0.1 || 1;
      return [lo - pad, hi + pad];
    };
    return { x: fit(xs, xRange), y: fit(ys, yRange) };
  }, [series, areas, refs, xRange, yRange]);
  const plots: PlotRect[] = [{ left: ml, top: MT, width: plotW, height: plotH, xAxis: MB, yAxis: yAxis ? ML : 0 }];
  // Sharing y: its x zoom its own, back to all of it with new data (`resetKey`).
  const [ownX, setOwnX] = useState<{ key: unknown; x: Range | null }>({ key: resetKey, x: null });
  const xZoom = Object.is(ownX.key, resetKey) ? ownX.x : null;
  const link = depthLink
    ? alongLine(xZoom, (next) => setOwnX({ key: resetKey, x: next }), depthLink.y, depthLink.setY, extent)
    : undefined;
  const zoom = useZoom({ extent, plots, width, height, yDown, resetKey, canvas: canvasRef, link });
  const [x0, x1] = zoom.view.x;
  const [y0, y1] = zoom.view.y;

  const hover = useMemo(() => {
    if (!mouse || mouse.x < ml || mouse.x > ml + plotW || mouse.y < MT || mouse.y > MT + plotH) return null;
    const X = (x: number) => ml + ((x - x0) / (x1 - x0)) * plotW;
    const Y = (y: number) =>
      yDown ? MT + ((y - y0) / (y1 - y0)) * plotH : MT + plotH - ((y - y0) / (y1 - y0)) * plotH;
    let best: { label: string; point: [number, number]; d: number } | null = null;
    for (const one of series) {
      if (one.quiet) continue;
      for (const point of one.points) {
        const d = Math.hypot(X(point[0]) - mouse.x, Y(point[1]) - mouse.y);
        if (d < 14 && (!best || d < best.d)) best = { label: one.label, point, d };
      }
    }
    const x = x0 + ((mouse.x - ml) / plotW) * (x1 - x0);
    const t = (mouse.y - MT) / plotH;
    const y = yDown ? y0 + t * (y1 - y0) : y1 - t * (y1 - y0);
    return best
      ? [best.label, `${xLabel}: ${num(best.point[0])}`, `${yLabel}: ${num(best.point[1])}`]
      : [`${xLabel}: ${num(x)}`, `${yLabel}: ${num(y)}`];
  }, [mouse, series, x0, x1, y0, y1, plotW, plotH, yDown, xLabel, yLabel, ml]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const X = (x: number) => ml + ((x - x0) / (x1 - x0)) * plotW;
    const Y = (y: number) =>
      yDown ? MT + ((y - y0) / (y1 - y0)) * plotH : MT + plotH - ((y - y0) / (y1 - y0)) * plotH;

    // Grid and ticks.
    const xTicks = ticks(x0, x1, Math.max(3, Math.floor(plotW / 70)));
    const yTicks = ticks(Math.min(y0, y1), Math.max(y0, y1), Math.max(3, Math.floor(plotH / 45)));
    ctx.strokeStyle = theme === "dark" ? "#2a2c31" : "#eeeef0";
    ctx.lineWidth = 1;
    for (const t of xTicks) {
      ctx.beginPath();
      ctx.moveTo(X(t), MT);
      ctx.lineTo(X(t), MT + plotH);
      ctx.stroke();
    }
    for (const t of yTicks) {
      ctx.beginPath();
      ctx.moveTo(ml, Y(t));
      ctx.lineTo(ml + plotW, Y(t));
      ctx.stroke();
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(ml, MT, plotW, plotH);
    ctx.clip();
    for (const area of areas) {
      if (area.polygon.length < 3) continue;
      ctx.fillStyle = area.color;
      ctx.beginPath();
      area.polygon.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(X(x), Y(y)) : ctx.lineTo(X(x), Y(y))));
      ctx.closePath();
      ctx.fill();
    }
    for (const ref of refs) {
      ctx.strokeStyle = ref.color;
      ctx.lineWidth = 1.3;
      ctx.setLineDash(ref.dash ?? [5, 4]);
      ctx.beginPath();
      if (ref.axis === "x") {
        ctx.moveTo(X(ref.at), MT);
        ctx.lineTo(X(ref.at), MT + plotH);
      } else {
        ctx.moveTo(ml, Y(ref.at));
        ctx.lineTo(ml + plotW, Y(ref.at));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = ref.color;
      ctx.font = CANVAS_FONT;
      if (ref.axis === "y") {
        ctx.textAlign = "right";
        ctx.textBaseline = "bottom";
        ctx.fillText(ref.label, ml + plotW - 4, Y(ref.at) - 2);
      } else {
        ctx.save();
        ctx.translate(X(ref.at) + 3, MT + 4);
        ctx.rotate(Math.PI / 2);
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(ref.label, 0, 0);
        ctx.restore();
      }
    }
    for (const one of series) {
      if (one.points.length === 0) continue;
      ctx.strokeStyle = one.color;
      ctx.fillStyle = one.color;
      ctx.lineWidth = one.width ?? 1.8;
      if (one.line !== false && one.points.length > 1) {
        ctx.setLineDash(one.dash ?? []);
        ctx.beginPath();
        one.points.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(X(x), Y(y)) : ctx.lineTo(X(x), Y(y))));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (one.dots) {
        one.points.forEach(([x, y], i) => {
          const error = one.errors?.[i];
          if (error != null && error > 0) {
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(X(x), Y(y - error));
            ctx.lineTo(X(x), Y(y + error));
            ctx.stroke();
          }
          ctx.beginPath();
          ctx.arc(X(x), Y(y), 2.6, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    }
    ctx.restore();

    // Axes.
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(ml + 0.5, MT + 0.5, plotW - 1, plotH - 1);
    ctx.font = CANVAS_FONT;
    ctx.fillStyle = palette.tick;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xDecimals = tickDecimals(xTicks.length > 1 ? xTicks[1] - xTicks[0] : 1);
    for (const t of xTicks) ctx.fillText(t.toFixed(xDecimals), X(t), MT + plotH + 4);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yDecimals = tickDecimals(yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1);
    if (yAxis) for (const t of yTicks) ctx.fillText(t.toFixed(yDecimals), ml - 5, Y(t));
    ctx.fillStyle = palette.title;
    ctx.font = CANVAS_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(xLabel, ml + plotW / 2, height - 2);
    if (yAxis) {
      ctx.save();
      ctx.translate(13, MT + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = "middle";
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }
  }, [series, areas, refs, width, height, plotW, plotH, x0, x1, y0, y1, yDown, palette, theme, xLabel, yLabel, ml, yAxis]);

  return (
    <div ref={containerRef} style={{ position: "relative", width: "100%" }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", cursor: zoom.cursorAt(mouse) }}
        onMouseMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          setMouse({ x: ((e.clientX - box.left) / box.width) * width, y: ((e.clientY - box.top) / box.height) * height });
        }}
        onMouseLeave={() => setMouse(null)}
        onMouseDown={zoom.onMouseDown}
      />
      <ZoomSelection box={zoom.selection} />
      {hover && mouse && (
        <div
          className="viz-tooltip"
          style={mouse.x > width / 2 ? { right: width - mouse.x + 12, top: mouse.y + 12 } : { left: mouse.x + 12, top: mouse.y + 12 }}
        >
          <TooltipLines lines={hover} />
        </div>
      )}
    </div>
  );
}

/** Round ticks about `count` apart over [lo, hi]. */
function ticks(lo: number, hi: number, count: number): number[] {
  const raw = (hi - lo) / count;
  if (!(raw > 0) || !Number.isFinite(raw)) return [lo];
  const power = 10 ** Math.floor(Math.log10(raw));
  const m = raw / power;
  const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * power;
  const found: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) found.push(+t.toFixed(10));
  return found;
}
