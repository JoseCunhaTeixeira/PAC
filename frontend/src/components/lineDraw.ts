import { CANVAS_FONT, canvasFont, type Theme } from "../theme";
import { tickDecimals } from "./useZoom";
import type { PartState } from "./viz/types";

// What every plot of a line from above shares, so that the computing pages' geometry and
// Visualization's line look alike: the lanes, the line's extent (a window as wide on both), the
// symbols' sizes and shapes (a star a shot, an inverted triangle a receiver), the position axis
// under them, and for a line that is not flat, its elevation.

export const LANE = {
  left: 84, // the lanes' labels
  right: 16,
  shotY: 28,
  receiverY: 62,
  windowY: 86,
  windowH: 34, // a window's cell: as tall as the rails'
  axisH: 30, // under the windows: ticks, their values
  axisGap: 12, // between the last lane and the position axis
} as const;

/** The windows' lane's bottom, where the position axis starts. */
export const LANE_BOTTOM = LANE.windowY + LANE.windowH;

/** The line's extent along x: `xs` (its receivers, shots and windows), a margin around. */
export function lineSpan(xs: readonly number[]): [number, number] {
  if (xs.length === 0) return [0, 1];
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  const pad = (hi - lo) * 0.015 || 1;
  return [lo - pad, hi + pad];
}

// A window's cell, as the rails draw theirs (kit.css's .rail-cell), so that a window looks the
// same on every page: as tall, 2 px from the next, rounded 4 px; a state's colour mixed into the
// page, stronger under the pointer (and 12 % taller) and when selected (and edged); a cell telling
// its checks apart, a band each, 1 px apart, rounded 2 px.
export const CELL = { gap: 2, radius: 4, bandGap: 1, bandRadius: 2, edge: 1, maxW: 12 } as const;
// A figure in a cell (how many modes were picked in it, when more than one), as the rails write
// it: in a cell as wide as a digit at least.
const CELL_FONT = canvasFont(9, 600);
export const COUNT_MIN_W = 5;

/** A cell's tone: a state, a part's (by hand), picked but not judged (auto), or a window as the
 * computing pages show it (series). */
export type CellTone = PartState | "auto" | "series";
export type CellStrength = "rest" | "hover" | "active";

// index.css's tokens, which a canvas cannot read.
const TOKENS: Record<Theme, Record<string, string>> = {
  light: {
    ok: "#1a9e4b",
    warn: "#b87404",
    bad: "#d63c3c",
    accent: "#4f5bd5",
    muted: "#525d70",
    series: "#2a78d6",
    surfaceHover: "#eef1f5",
    borderStrong: "#cdd3dc",
    text: "#0f1728",
  },
  dark: {
    ok: "#34c46a",
    warn: "#f0b429",
    bad: "#f06a6a",
    accent: "#6671ec",
    muted: "#a1abbc",
    series: "#4f9cf5",
    surfaceHover: "#19212d",
    borderStrong: "#2d394a",
    text: "#e7ebf2",
  },
};
// kit.css's mixes of each tone into the page, in %: at rest, under the pointer, selected.
const MIXES: Record<Exclude<CellTone, "none">, readonly [number, number, number]> = {
  pass: [34, 58, 72],
  warn: [42, 66, 80],
  fail: [40, 64, 78],
  hand: [45, 70, 80],
  auto: [34, 55, 72],
  series: [34, 58, 80],
};
const BAND_MIXES = [38, 62, 80] as const;
const STRENGTHS: Record<CellStrength, 0 | 1 | 2> = { rest: 0, hover: 1, active: 2 };

/** A cell's colour, as the rails': `tone` at `strength`, a band's when `band`. */
export function cellColour(theme: Theme, tone: CellTone, strength: CellStrength, band = false): string {
  const tokens = TOKENS[theme];
  const i = STRENGTHS[strength];
  if (tone === "none") {
    return band ? withAlpha(tokens.borderStrong, BAND_MIXES[i] / 100) : i === 0 ? tokens.surfaceHover : tokens.borderStrong;
  }
  const base = {
    pass: tokens.ok,
    warn: tokens.warn,
    fail: tokens.bad,
    hand: tokens.accent,
    auto: tokens.muted,
    series: tokens.series,
  }[tone];
  return withAlpha(base, (band ? BAND_MIXES : MIXES[tone])[i] / 100);
}

/** A window's cell centred on `x`, `w` wide, as the rails draw theirs: its tones top down (one,
 * or a band each), at `strength`, `label` written small in the last (its curve's: how many modes
 * were picked in it, when more than one); selected, edged as a rail's (1 px of the text's colour and a faint ring);
 * under the pointer, 12 % taller. */
export function drawCell(
  ctx: CanvasRenderingContext2D,
  theme: Theme,
  x: number,
  w: number,
  tones: readonly CellTone[],
  strength: CellStrength,
  selected = false,
  label?: string,
): void {
  const h = LANE.windowH * (strength === "hover" ? 1.12 : 1);
  const top = LANE.windowY + (LANE.windowH - h) / 2;
  const left = x - w / 2;
  const band = tones.length <= 1 ? h : (h - CELL.bandGap * (tones.length - 1)) / tones.length;
  if (tones.length <= 1) {
    ctx.fillStyle = cellColour(theme, tones[0] ?? "none", strength);
    roundedRect(ctx, left, top, w, h, CELL.radius);
    ctx.fill();
  } else {
    tones.forEach((tone, i) => {
      ctx.fillStyle = cellColour(theme, tone, strength, true);
      roundedRect(ctx, left, top + i * (band + CELL.bandGap), w, band, CELL.bandRadius);
      ctx.fill();
    });
  }
  if (label && w >= COUNT_MIN_W) {
    ctx.font = CELL_FONT;
    ctx.fillStyle = TOKENS[theme].text;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x, top + h - band / 2 + 0.5);
  }
  if (selected) {
    // The edge inside the cell, the ring against it outside: each `CELL.edge` wide, as a rail's
    // border and box-shadow.
    const tokens = TOKENS[theme];
    const half = CELL.edge / 2;
    ctx.lineWidth = CELL.edge;
    ctx.strokeStyle = tokens.text;
    roundedRect(ctx, left + half, top + half, w - CELL.edge, h - CELL.edge, CELL.radius - half);
    ctx.stroke();
    ctx.strokeStyle = withAlpha(tokens.text, 0.16);
    roundedRect(ctx, left - half, top - half, w + CELL.edge, h + CELL.edge, CELL.radius + half);
    ctx.stroke();
  }
}

/** The tightest gap between `xs`; Infinity without two apart. */
function tightestGap(xs: readonly number[]): number {
  const sorted = [...xs].sort((a, b) => a - b);
  let gap = Infinity;
  for (let i = 1; i < sorted.length; i++) if (sorted[i] > sorted[i - 1]) gap = Math.min(gap, sorted[i] - sorted[i - 1]);
  return gap;
}

/** The receivers' usual spacing: the median of their gaps; Infinity without two apart. */
function receiverSpacing(receivers: readonly number[]): number {
  const xs = [...receivers].sort((a, b) => a - b);
  const gaps = xs.slice(1).map((x, i) => x - xs[i]).filter((gap) => gap > 0).sort((a, b) => a - b);
  return gaps.length ? gaps[Math.floor(gaps.length / 2)] : Infinity;
}

/** A cell's share of the line, in metres: a receiver's spacing, whatever the windows' step (the
 * tightest gap between windows at most, so that none overlap; that gap without receivers). */
export function cellMetres(xmids: readonly number[], receivers: readonly number[]): number {
  return Math.min(tightestGap(xmids), receiverSpacing(receivers));
}

/** The cells' width on screen: their share of the line, less the rails' gap, never wider than
 * `CELL.maxW`: small cells whatever the step, a larger one leaving larger gaps between them. */
export function cellWidth(xmids: readonly number[], receivers: readonly number[], span: number, plotW: number): number {
  const metres = cellMetres(xmids, receivers);
  const px = Number.isFinite(metres) && span > 0 ? (metres / span) * plotW : 14;
  return Math.max(2, Math.min(CELL.maxW, px - CELL.gap));
}

/** A rounded rectangle's path, its corners never rounder than half its side. */
export function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, Math.max(0, Math.min(r, w / 2, h / 2)));
}

/** A round step near `raw`: 1, 2 or 5 times a power of ten. */
function niceStep(raw: number): number {
  const power = 10 ** Math.floor(Math.log10(raw));
  const m = raw / power;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * power;
}

export function niceTicks(lo: number, hi: number, count: number): number[] {
  const raw = (hi - lo) / count;
  if (!(raw > 0)) return [lo];
  const step = niceStep(raw);
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(+t.toFixed(9));
  return ticks;
}

/** Symbols sized to their spacing on screen, so that a line of many keeps them apart. */
export function symbolSizes(
  shots: number[],
  receivers: number[],
  view: readonly [number, number],
  plotW: number,
): { starR: number; triangleR: number } {
  const spacing = (xs: number[]) => {
    const sorted = [...xs].sort((a, b) => a - b);
    let gap = Infinity;
    for (let i = 1; i < sorted.length; i++) gap = Math.min(gap, sorted[i] - sorted[i - 1]);
    return Number.isFinite(gap) && gap > 0 ? (gap / (view[1] - view[0])) * plotW : 20;
  };
  return {
    starR: Math.max(3, Math.min(7, spacing(shots) * 0.55)),
    triangleR: Math.max(2.5, Math.min(5.5, spacing(receivers) * 0.45)),
  };
}

/** A five-pointed star centred on (`x`, `y`), `r` its points' radius: a path to fill. */
export function starPath(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const radius = i % 2 === 0 ? r : r * 0.45;
    const angle = -Math.PI / 2 + (i * Math.PI) / 5;
    const px = x + radius * Math.cos(angle);
    const py = y + radius * Math.sin(angle);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

/** An inverted triangle `r` wide, its point at (`x`, `tip`): a path to fill. */
export function trianglePath(ctx: CanvasRenderingContext2D, x: number, tip: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x - r, tip - r * 1.7);
  ctx.lineTo(x + r, tip - r * 1.7);
  ctx.lineTo(x, tip);
  ctx.closePath();
}

/** A receiver on the receivers' lane: an inverted triangle `r` wide, its point where the lane's
 * size `size` puts every point, at the same height. */
export function receiverPath(ctx: CanvasRenderingContext2D, x: number, size: number, r: number) {
  trianglePath(ctx, x, LANE.receiverY + size * 0.9, r);
}

/** The lanes' labels, left of the plot. */
export function drawLaneLabels(ctx: CanvasRenderingContext2D, color: string, shots: boolean) {
  ctx.font = CANVAS_FONT;
  ctx.fillStyle = color;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  if (shots) ctx.fillText("Shots", LANE.left - 12, LANE.shotY);
  ctx.fillText("Receivers", LANE.left - 12, LANE.receiverY);
  ctx.fillText("Windows", LANE.left - 12, LANE.windowY + LANE.windowH / 2);
}

/** The position axis under the lanes (from `top`, the last lane's bottom): ticks at round
 * distances with their values, the first with its unit, and "Position" left of them as the
 * lanes' labels are written; no baseline. */
export function drawLineAxis(
  ctx: CanvasRenderingContext2D,
  view: readonly [number, number],
  plotW: number,
  colors: { axis: string; tick: string; title: string },
  top: number = LANE_BOTTOM,
) {
  const [x0, x1] = view;
  const y = top + 4;
  ctx.strokeStyle = colors.axis;
  ctx.lineWidth = 1;
  ctx.font = CANVAS_FONT;
  ctx.fillStyle = colors.tick;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const decimals = tickDecimals((x1 - x0) / 8);
  let first = true;
  for (const t of niceTicks(x0, x1, Math.max(3, Math.floor(plotW / 90)))) {
    const x = LANE.left + ((t - x0) / (x1 - x0)) * plotW;
    if (x < LANE.left - 1 || x > LANE.left + plotW + 1) continue;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y + 4);
    ctx.stroke();
    ctx.fillText(first ? `${t.toFixed(decimals)} m` : t.toFixed(decimals), x, y + 7);
    first = false;
  }
  // The axis's name as the lanes' labels are written, level with the ticks' values.
  ctx.font = CANVAS_FONT;
  ctx.fillStyle = colors.tick;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText("Position", LANE.left - 12, y + 12.5);
}

/** The elevation lane, under the windows' one: its gaps above and below (to the position
 * axis), its height, and its margins inside, where the highest and lowest points' values are
 * written. */
export const ELEVATION = { gap: 14, after: LANE.axisGap, h: 80, top: 18, bottom: 18 } as const;

/** The ground along the line, and the elevations its lane spans. */
export interface Relief {
  ground: [number, number][]; // (x, z), in x order
  range: [number, number]; // the ground's, widened to a metre at least
  flat: boolean; // to the millimetre: drawn straight
}

/** The ground from the receivers' elevations, extended by the shots beyond them; null without
 * a receiver. */
export function reliefOf(receivers: [number, number][], shots: [number, number][]): Relief | null {
  if (receivers.length === 0) return null;
  const zs = [...receivers, ...shots].map(([, z]) => z);
  const lo = Math.min(...zs);
  const hi = Math.max(...zs);
  const xs = receivers.map(([x]) => x);
  const first = Math.min(...xs);
  const last = Math.max(...xs);
  const ground = [...receivers, ...shots.filter(([x]) => x < first || x > last)].sort((a, b) => a[0] - b[0]);
  // A metre at least, so that a few centimetres of relief look flat.
  const half = Math.max(hi - lo, 1) / 2;
  const mid = (lo + hi) / 2;
  return { ground, range: [mid - half, mid + half], flat: !(hi - lo > 1e-3) };
}

/** The y of elevation `z` in the elevation lane at `top`. */
export function elevationY(relief: Relief, top: number): (z: number) => number {
  const [lo, hi] = relief.range;
  const y0 = top + ELEVATION.top;
  const y1 = top + ELEVATION.h - ELEVATION.bottom;
  return (z: number) => y1 - ((z - lo) / (hi - lo)) * (y1 - y0);
}

/** The ground's elevation at `x`, between its points (the nearest end's beyond them). */
export function groundAt(relief: Relief, x: number): number {
  const { ground } = relief;
  if (x <= ground[0][0]) return ground[0][1];
  for (let i = 1; i < ground.length; i++) {
    const [xa, za] = ground[i - 1];
    const [xb, zb] = ground[i];
    if (x <= xb) return xb > xa ? za + ((x - xa) / (xb - xa)) * (zb - za) : zb;
  }
  return ground[ground.length - 1][1];
}

/** The elevation lane at `top`: its label left of the plot, as the lanes' are written, the
 * ground's line, shaded below, and the elevations of its highest and lowest points in view,
 * written over and under them. */
export function drawElevation(
  ctx: CanvasRenderingContext2D,
  relief: Relief,
  top: number,
  view: readonly [number, number],
  plotW: number,
  colors: { tick: string; ground: string },
) {
  const Y = elevationY(relief, top);
  ctx.font = CANVAS_FONT;
  ctx.fillStyle = colors.tick;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText("Elevation", LANE.left - 12, top + ELEVATION.h / 2);

  const [x0, x1] = view;
  const X = (x: number) => LANE.left + ((x - x0) / (x1 - x0)) * plotW;
  const bottom = top + ELEVATION.h;
  ctx.save();
  ctx.beginPath();
  ctx.rect(LANE.left, top, plotW, ELEVATION.h);
  ctx.clip();
  const points = relief.ground.map(([x, z]) => [X(x), Y(z)] as const);
  const highest = Math.min(...points.map(([, y]) => y));
  // The ground: shaded from its line down, fading.
  const shade = ctx.createLinearGradient(0, highest, 0, bottom);
  shade.addColorStop(0, withAlpha(colors.ground, 0.22));
  shade.addColorStop(1, withAlpha(colors.ground, 0.03));
  ctx.beginPath();
  ctx.moveTo(points[0][0], bottom);
  for (const [x, y] of points) ctx.lineTo(x, y);
  ctx.lineTo(points[points.length - 1][0], bottom);
  ctx.closePath();
  ctx.fillStyle = shade;
  ctx.fill();
  ctx.beginPath();
  points.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  ctx.strokeStyle = colors.ground;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.stroke();

  // The highest and lowest points in view, their elevations by them, kept inside the plot.
  const shown = relief.ground.filter(([x]) => x >= x0 && x <= x1);
  if (shown.length > 0) {
    const high = shown.reduce((a, b) => (b[1] > a[1] ? b : a));
    const low = shown.reduce((a, b) => (b[1] < a[1] ? b : a));
    const decimals = high[1] - low[1] < 1 ? 2 : 1;
    ctx.font = CANVAS_FONT;
    ctx.fillStyle = colors.tick;
    ctx.textAlign = "center";
    const write = ([x, z]: [number, number], y: number, baseline: CanvasTextBaseline) => {
      const text = `${z.toFixed(decimals)} m`;
      const half = ctx.measureText(text).width / 2;
      ctx.textBaseline = baseline;
      ctx.fillText(text, Math.min(Math.max(X(x), LANE.left + half + 2), LANE.left + plotW - half - 2), y);
    };
    if (relief.flat) {
      // One elevation: said once, in the middle.
      write([(x0 + x1) / 2, high[1]], Y(high[1]) - 5, "bottom");
    } else {
      write(high, Y(high[1]) - 5, "bottom");
      write(low, Y(low[1]) + 5, "top");
    }
  }
  ctx.restore();
}

/** A hex colour ("#5a5a60") at opacity `alpha`. */
function withAlpha(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

/** The slope in degrees of the line best through `points` (x, z): a window's ground. */
export function slopeDegrees(points: [number, number][]): number | null {
  if (points.length < 2) return null;
  const mx = points.reduce((sum, [x]) => sum + x, 0) / points.length;
  const mz = points.reduce((sum, [, z]) => sum + z, 0) / points.length;
  let sxz = 0;
  let sxx = 0;
  for (const [x, z] of points) {
    sxz += (x - mx) * (z - mz);
    sxx += (x - mx) ** 2;
  }
  return sxx > 0 ? (Math.atan(Math.abs(sxz / sxx)) * 180) / Math.PI : null;
}
