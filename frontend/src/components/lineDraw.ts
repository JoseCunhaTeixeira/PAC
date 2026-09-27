import { canvasFont } from "../theme";
import { tickDecimals } from "./useZoom";

// What every plot of a line from above shares, so that the computing pages' geometry and
// Visualization's line look alike: the lanes, the symbols' sizes and shapes (a star a shot, an
// inverted triangle a receiver), the position axis under them, and for a line that is not flat,
// its elevation.

export const LANE = {
  left: 84, // the lanes' labels
  right: 16,
  shotY: 28,
  receiverY: 62,
  windowY: 86,
  windowH: 30,
  axisH: 30, // under the windows: ticks, their values
  axisGap: 12, // between the last lane and the position axis
} as const;

/** The windows' lane's bottom, where the position axis starts. */
export const LANE_BOTTOM = LANE.windowY + LANE.windowH;

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
  ctx.font = canvasFont(12);
  ctx.fillStyle = color;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  if (shots) ctx.fillText("Shots", LANE.left - 12, LANE.shotY);
  ctx.fillText("Receivers", LANE.left - 12, LANE.receiverY);
  ctx.fillText("Windows", LANE.left - 12, LANE.windowY + LANE.windowH / 2);
}

/** The position axis under the lanes (from `top`, the last lane's bottom): ticks at round
 * distances with their values, the first with its unit, and "Position" left of them as the
 * lanes' labels are written; no baseline (the user's choice). */
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
  ctx.font = canvasFont(11);
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
  ctx.font = canvasFont(12);
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
  ctx.font = canvasFont(12);
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
    ctx.font = canvasFont(11);
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
