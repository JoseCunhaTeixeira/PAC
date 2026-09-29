import { useEffect, useMemo, useRef, useState } from "react";
import { CANVAS_FONT, canvasPalette, useTheme } from "../../theme";
import { useContainerWidth } from "../useContainerWidth";
import { tickDecimals, useZoom, type PlotRect, type Range } from "../useZoom";
import { ZoomSelection } from "../ZoomOverlay";
import { TipLines } from "../HoverTooltip";
import type { Tip } from "../tips";
import { num } from "./format";
import { alongLine } from "./line";
import { vizPalette } from "./palette";

// A record's traces as wiggles, each at its receiver's position along the line, under the line
// plot and aligned with it (the same margins, the same zoom along the line): the shot a star
// above its position, the traces the windows left out grey. Time runs down. A computing page's
// muting shows on it too: what it removes of each trace veiled, its tapers half-veiled.

export interface GatherData {
  positions: number[]; // each trace's receiver, m along the line
  source: number | null;
  dt: number; // s between samples
  traces: number[][]; // normalized: the largest |value| 1
  excluded: number[]; // traces left out of the windows
}

/** A muting to preview: each trace kept from the later of `tmin` and its offset over `vmax`
 * to the earlier of `tmax` and its offset over `vmin` (0: no limit), tapered over `taper`
 * seconds inside. */
export interface MuteOverlay {
  offsets: number[]; // each trace's distance from the shot, m
  tmin: number;
  tmax: number;
  vmin: number; // 0: none
  vmax: number; // 0: none
  width?: number; // s kept after the slowest arrival: the shot's pulse
  taper: number;
  /** The trigger's shift, s: the mute measured on the shifted record (a late trigger drops the
   * record's first `shift` seconds, an early one its last). */
  shift?: number;
}

/** What `mute` keeps of trace `i`: [from, to] in seconds of the record as recorded (to < from:
 * nothing). */
function kept(mute: MuteOverlay, i: number): [number, number] {
  const offset = mute.offsets[i] ?? 0;
  const shift = mute.shift ?? 0;
  const from = Math.max(mute.tmin, mute.vmax > 0 ? offset / mute.vmax : 0) + shift;
  const to = Math.min(mute.tmax, mute.vmin > 0 ? offset / mute.vmin + (mute.width ?? 0) : Infinity) + shift;
  return [Math.max(0, from), to];
}

const ML = 84;
const MR = 16;
const MT = 26;
const MB = 46; // the ticks' values, then the axis's title

function niceTicks(lo: number, hi: number, count: number): number[] {
  const raw = (hi - lo) / count;
  if (!(raw > 0)) return [lo];
  const power = 10 ** Math.floor(Math.log10(raw));
  const m = raw / power;
  const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * power;
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(+t.toFixed(9));
  return ticks;
}

export function LineGather({
  data,
  extent,
  xZoom,
  onXZoom,
  height = 440,
  mute,
}: {
  data: GatherData;
  /** The line's extent along x, as the line plot has it. */
  extent: Range;
  /** The zoom along the line, shared with the line plot; null for the whole line. */
  xZoom: Range | null;
  onXZoom: (x: Range | null) => void;
  height?: number;
  /** A muting to preview over the record. */
  mute?: MuteOverlay;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const axes = canvasPalette(theme);
  const palette = vizPalette(theme);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const width = Math.max(320, Math.floor(containerWidth || 800));
  const plotW = width - ML - MR;
  const plotH = height - MT - MB;
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null);
  const samples = data.traces[0]?.length ?? 0;
  const duration = Math.max(data.dt * Math.max(samples - 1, 1), 1e-6);
  // The whole record first (the user); the wheel and the box's tools zoom.
  const [ownY, setOwnY] = useState<Range | null>(null);
  const full = useMemo(() => ({ x: extent, y: [0, duration] as Range }), [extent, duration]);
  const plots: PlotRect[] = [{ left: ML, top: MT, width: plotW, height: plotH, xAxis: MB, yAxis: ML }];
  const zoom = useZoom({
    extent: full,
    plots,
    width,
    height,
    yDown: true,
    link: alongLine(xZoom, onXZoom, ownY, setOwnY, full),
    canvas: canvasRef,
  });
  const [x0, x1] = zoom.view.x;
  const [t0, t1] = zoom.view.y;
  const excluded = useMemo(() => new Set(data.excluded), [data.excluded]);

  // The trace under the pointer, or the sensor over the plot: its receiver, highlighted.
  const pointed = useMemo(() => {
    if (!mouse || mouse.x < ML || mouse.x > ML + plotW || mouse.y < 0 || mouse.y > MT + plotH) return null;
    const x = x0 + ((mouse.x - ML) / plotW) * (x1 - x0);
    let best = -1;
    data.positions.forEach((position, i) => {
      if (best < 0 || Math.abs(position - x) < Math.abs(data.positions[best] - x)) best = i;
    });
    return best < 0 ? null : best;
  }, [mouse, plotW, plotH, x0, x1, data]);

  const hover = useMemo((): Tip | null => {
    if (pointed === null || !mouse) return null;
    const position = data.positions[pointed];
    const notes: string[] = [];
    if (data.source !== null) notes.push(`${num(Math.abs(position - data.source), 4)} m from the shot`);
    if (excluded.has(pointed)) notes.push("left out of the windows");
    // In the plot, the time under the pointer, and whether the muting removes it.
    let time = "";
    if (mouse.y >= MT) {
      const t = t0 + ((mouse.y - MT) / plotH) * (t1 - t0);
      time = `; ${num(t * 1000, 4)} ms`;
      if (mute) {
        const [from, to] = kept(mute, pointed);
        if (t < from || t > to) notes.push("muted");
      }
    }
    return { title: `Receiver ${pointed + 1}`, values: `${num(position, 4)} m${time}`, notes };
  }, [pointed, mouse, plotH, t0, t1, data, excluded, mute]);

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
    const X = (x: number) => ML + ((x - x0) / (x1 - x0)) * plotW;
    const Y = (t: number) => MT + ((t - t0) / (t1 - t0)) * plotH;

    // Each trace's half-width: under half the tightest spacing on screen.
    let gap = Infinity;
    const sorted = [...data.positions].sort((a, b) => a - b);
    for (let i = 1; i < sorted.length; i++) gap = Math.min(gap, sorted[i] - sorted[i - 1]);
    const amplitude = Math.max(2, Math.min(40, ((Number.isFinite(gap) ? gap : 1) / (x1 - x0)) * plotW * 0.9));
    const first = Math.max(0, Math.floor(t0 / data.dt));
    const last = Math.min(samples - 1, Math.ceil(t1 / data.dt));

    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, MT, plotW, plotH);
    ctx.clip();
    const ink = theme === "dark" ? "#e6e6ea" : "#1c1c20";
    data.traces.forEach((trace, i) => {
      const base = X(data.positions[i]);
      if (base < ML - amplitude || base > ML + plotW + amplitude) return;
      const out = excluded.has(i);
      // Positive lobes filled (variable area), the whole wiggle stroked.
      if (!out) {
        ctx.fillStyle = ink;
        ctx.beginPath();
        ctx.moveTo(base, Y(first * data.dt));
        for (let j = first; j <= last; j++) ctx.lineTo(base + Math.max(0, trace[j]) * amplitude, Y(j * data.dt));
        ctx.lineTo(base, Y(last * data.dt));
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = out ? palette.status.fail : ink;
      ctx.globalAlpha = out ? 0.55 : 1;
      ctx.lineWidth = out ? 1 : 0.7;
      ctx.beginPath();
      for (let j = first; j <= last; j++) {
        const px = base + trace[j] * amplitude;
        const py = Y(j * data.dt);
        if (j === first) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    // The muting: what it removes veiled, trace by trace, its tapers half-veiled; each veil one
    // shape, its traces' slots overlapping a little, so that no seam shows between them.
    if (mute) {
      const slot = Math.max(2, ((Number.isFinite(gap) ? gap : 1) / (x1 - x0)) * plotW);
      const end = (samples - 1) * data.dt;
      const veil = new Path2D();
      const half = new Path2D();
      const band = (path: Path2D, left: number, from: number, to: number) =>
        path.rect(left, Y(from), slot + 1, Y(to) - Y(from));
      data.positions.forEach((position, i) => {
        const left = X(position) - slot / 2 - 0.5;
        const [from, to] = kept(mute, i);
        if (to <= from) {
          band(veil, left, 0, end);
          return;
        }
        band(veil, left, 0, from);
        if (to < end) band(veil, left, to, end);
        const taper = Math.min(mute.taper, (Math.min(to, end) - from) / 2);
        if (taper > 0) {
          band(half, left, from, from + taper);
          if (to < end) band(half, left, to - taper, to);
        }
      });
      ctx.fillStyle = theme === "dark" ? "rgba(33, 40, 51, 0.82)" : "rgba(214, 219, 227, 0.82)";
      ctx.fill(veil);
      ctx.fillStyle = theme === "dark" ? "rgba(33, 40, 51, 0.45)" : "rgba(214, 219, 227, 0.45)";
      ctx.fill(half);
    }
    ctx.restore();

    // Above the plot: the receivers (left-out ones red, the one pointed at bigger and darker)
    // and the shot.
    ctx.save();
    ctx.beginPath();
    ctx.rect(ML - 8, 0, plotW + 16, MT);
    ctx.clip();
    data.positions.forEach((position, i) => {
      const x = X(position);
      const on = i === pointed;
      const r = on ? 5 : 3.5;
      ctx.fillStyle = excluded.has(i) ? palette.status.fail : on ? palette.ink : palette.muted;
      ctx.beginPath();
      ctx.moveTo(x - r, MT - 4 - r * 1.7);
      ctx.lineTo(x + r, MT - 4 - r * 1.7);
      ctx.lineTo(x, MT - 4);
      ctx.closePath();
      ctx.fill();
    });
    if (data.source !== null) {
      const x = X(data.source);
      ctx.fillStyle = palette.series;
      ctx.beginPath();
      for (let k = 0; k < 10; k++) {
        const radius = k % 2 === 0 ? 7 : 3.2;
        const angle = -Math.PI / 2 + (k * Math.PI) / 5;
        const px = x + radius * Math.cos(angle);
        const py = 10 + radius * Math.sin(angle);
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();

    // Axes: position along the line, time down.
    ctx.strokeStyle = axes.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(ML + 0.5, MT + 0.5, plotW - 1, plotH - 1);
    ctx.font = CANVAS_FONT;
    ctx.fillStyle = axes.tick;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xs = niceTicks(x0, x1, Math.max(3, Math.floor(plotW / 90)));
    const xDecimals = tickDecimals((x1 - x0) / 8);
    for (const t of xs) {
      const x = X(t);
      if (x < ML - 1 || x > ML + plotW + 1) continue;
      ctx.beginPath();
      ctx.moveTo(x, MT + plotH);
      ctx.lineTo(x, MT + plotH + 4);
      ctx.stroke();
      ctx.fillText(t.toFixed(xDecimals), x, MT + plotH + 6);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const ts = niceTicks(t0, t1, Math.max(3, Math.floor(plotH / 55)));
    const tDecimals = tickDecimals((t1 - t0) / 6);
    for (const t of ts) {
      const y = Y(t);
      if (y < MT - 1 || y > MT + plotH + 1) continue;
      ctx.beginPath();
      ctx.moveTo(ML - 4, y);
      ctx.lineTo(ML, y);
      ctx.stroke();
      ctx.fillText(t.toFixed(tDecimals), ML - 7, y);
    }
    // The axes' titles, alike: position under, time left.
    ctx.font = CANVAS_FONT;
    ctx.fillStyle = axes.title;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Position (m)", ML + plotW / 2, height - 6);
    ctx.save();
    ctx.translate(18, MT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Time (s)", 0, 0);
    ctx.restore();
  }, [data, width, height, plotW, plotH, x0, x1, t0, t1, samples, excluded, axes, palette, theme, mute, pointed]);

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
          style={mouse.x > width * 0.6 ? { right: width - mouse.x + 14, top: mouse.y + 14 } : { left: mouse.x + 14, top: mouse.y + 14 }}
        >
          <TipLines tip={hover} />
        </div>
      )}
    </div>
  );
}
