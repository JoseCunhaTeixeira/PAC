import { useEffect, useState } from "react";

// Zoom, shared by every plot and done as Plotly does it: a drag on a plot
// area draws a rectangle, and releasing it zooms both axes to it; a drag
// that is nearly horizontal or vertical, or one along an axis's tick labels,
// zooms that axis only. A double-click (or the "Reset zoom" button, see
// ZoomOverlay) goes back to the full view. The zoom is held in data units, so
// it survives a resize, and new data keeps it where it still fits.

/** [low, high], low < high, in data units. */
export type Range = readonly [number, number];

export interface View {
  x: Range;
  y: Range;
}

/** A plot area, in the drawing's own logical pixels (before any CSS scaling). */
export interface PlotRect {
  left: number;
  top: number;
  width: number;
  height: number;
  /** Height of the x-axis strip under the plot (its tick labels), where a
   * drag zooms x only. None when left out. */
  xAxis?: number;
  /** Width of the y-axis strip left of the plot, where a drag zooms y only. */
  yAxis?: number;
}

/** The rectangle or band being dragged, in CSS pixels from the drawing's
 * top-left corner. */
export interface SelectionBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** One zoom for several plots of the same grid (a Vs section and its std):
 * zooming or resetting one does all. Made by useZoomLink. */
export interface ZoomLink {
  zoom: View | null;
  setZoom: (zoom: View | null) => void;
}

/** A drag under this many CSS pixels is a click, not a zoom. */
export const CLICK_PX = 5;

// In a plot, a drag whose short side is under this many CSS pixels is taken
// as nearly horizontal or vertical: it zooms along its long side only.
const THIN_PX = 10;

// Past a millionth of the full range there is nothing more to see.
const MIN_ZOOM_FRACTION = 1e-6;

/** Where a drag started: in a plot, or on one of its axis strips. */
type Region = "plot" | "x" | "y";
/** What a drag zooms. */
type Axes = "xy" | "x" | "y";

interface Point {
  x: number;
  y: number;
}

interface Drag {
  el: Element;
  plot: PlotRect;
  region: Region;
  start: Point;
  // What the drag is drawn over: the view when it started.
  view: View;
  extent: View;
  yDown: boolean;
  width: number;
  height: number;
  setZoom: (zoom: View) => void;
  onZoom?: () => void;
}

// Object.is, so a NaN range (no data) compares equal to itself.
function sameRange(a: Range, b: Range) {
  return Object.is(a[0], b[0]) && Object.is(a[1], b[1]);
}

function sameView(a: View, b: View) {
  return sameRange(a.x, b.x) && sameRange(a.y, b.y);
}

function clampRange(range: Range, full: Range): Range | null {
  const lo = Math.max(range[0], full[0]);
  const hi = Math.min(range[1], full[1]);
  return hi > lo ? [lo, hi] : null;
}

/** `zoom` within `full`: kept when inside, clamped when partly outside, null
 * (the full view) when they no longer overlap or it covers all of `full`. */
function clampView(zoom: View, full: View): View | null {
  const x = clampRange(zoom.x, full.x);
  const y = clampRange(zoom.y, full.y);
  if (!x || !y || (sameRange(x, full.x) && sameRange(y, full.y))) return null;
  return { x, y };
}

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

function regionAt(plots: PlotRect[], x: number, y: number): { plot: PlotRect; region: Region } | null {
  for (const plot of plots) {
    const { left, top, width, height } = plot;
    const inX = x >= left && x <= left + width;
    const inY = y >= top && y <= top + height;
    if (inX && inY) return { plot, region: "plot" };
    if (inX && y > top + height && y <= top + height + (plot.xAxis ?? 0)) return { plot, region: "x" };
    if (inY && x < left && x >= left - (plot.yAxis ?? 0)) return { plot, region: "y" };
  }
  return null;
}

/** What a drag `sx` by `sy` CSS pixels zooms, or null while it is a click. */
function dragAxes(region: Region, sx: number, sy: number): Axes | null {
  if (region !== "plot") return (region === "x" ? sx : sy) < CLICK_PX ? null : region;
  if (sx < CLICK_PX && sy < CLICK_PX) return null;
  if (sx < THIN_PX || sy < THIN_PX) return sx >= sy ? "x" : "y";
  return "xy";
}

/** The rectangle, or for one axis a band across the whole plot, in CSS pixels. */
function selectionBox(plot: PlotRect, start: Point, end: Point, axes: Axes, cssPerPx: number): SelectionBox {
  const [x0, x1] =
    axes === "y" ? [plot.left, plot.left + plot.width] : [Math.min(start.x, end.x), Math.max(start.x, end.x)];
  const [y0, y1] =
    axes === "x" ? [plot.top, plot.top + plot.height] : [Math.min(start.y, end.y), Math.max(start.y, end.y)];
  return { left: x0 * cssPerPx, top: y0 * cssPerPx, width: (x1 - x0) * cssPerPx, height: (y1 - y0) * cssPerPx };
}

/** The view a released drag zooms to; null when there is nothing to zoom. */
function zoomTo(drag: Drag, end: Point, axes: Axes): View | null {
  const { plot, start, view, extent, yDown } = drag;
  const [vx0, vx1] = view.x;
  const [vy0, vy1] = view.y;
  const xAt = (px: number) => vx0 + ((px - plot.left) / plot.width) * (vx1 - vx0);
  const yAt = (py: number) => {
    const t = (py - plot.top) / plot.height;
    return yDown ? vy0 + t * (vy1 - vy0) : vy1 - t * (vy1 - vy0);
  };
  const sorted = (a: number, b: number): Range => (a < b ? [a, b] : [b, a]);
  const next: View = {
    x: axes === "y" ? view.x : sorted(xAt(start.x), xAt(end.x)),
    y: axes === "x" ? view.y : sorted(yAt(start.y), yAt(end.y)),
  };
  const tooSmall = (r: Range, full: Range) => r[1] - r[0] < (full[1] - full[0]) * MIN_ZOOM_FRACTION;
  if (tooSmall(next.x, extent.x) || tooSmall(next.y, extent.y)) return null;
  return clampView(next, extent);
}

/** A zoom several plots share (pass it to each as `link`), back to the full
 * view when `resetKey` changes (another folder). */
export function useZoomLink(resetKey?: unknown): ZoomLink {
  const [zoom, setZoom] = useState<View | null>(null);
  const [seenKey, setSeenKey] = useState(resetKey);
  if (!Object.is(seenKey, resetKey)) {
    setSeenKey(resetKey);
    setZoom(null);
  }
  return { zoom, setZoom };
}

export function useZoom({
  extent,
  plots,
  width,
  height,
  yDown = false,
  resetKey,
  link,
  onZoom,
}: {
  /** The full view: the data's x and y ranges, as the plot maps them. */
  extent: View;
  /** Where a drag can start, in logical pixels, with the plot's axis strips.
   * Stacked panels sharing their axes pass one rect each; a drag stays in the
   * panel it started in. */
  plots: PlotRect[];
  /** The drawing's logical size, to turn mouse positions into logical pixels. */
  width: number;
  height: number;
  /** True when y grows downward (time, wavelength as a depth). */
  yDown?: boolean;
  /** A change resets the zoom (another quantity on an axis). */
  resetKey?: unknown;
  /** A zoom shared with other plots, instead of this plot's own. */
  link?: ZoomLink;
  /** Called after a drag zooms. */
  onZoom?: () => void;
}) {
  const [ownZoom, setOwnZoom] = useState<View | null>(null);
  const zoom = link ? link.zoom : ownZoom;
  const setZoom = link ? link.setZoom : setOwnZoom;
  const [drag, setDrag] = useState<Drag | null>(null);
  const [selection, setSelection] = useState<SelectionBox | null>(null);

  // New data (the next position, a new model...) keeps the zoom where it
  // still lies within the data, clamps it where it doesn't, and resets it
  // once they no longer overlap. Adjusted during render, React's pattern for
  // state that follows a prop. A linked zoom belongs to several plots: each
  // shows it clamped to its own data, below.
  const [seen, setSeen] = useState({ extent, resetKey });
  if (!sameView(seen.extent, extent) || !Object.is(seen.resetKey, resetKey)) {
    setSeen({ extent, resetKey });
    if (!link) {
      setOwnZoom(ownZoom && Object.is(seen.resetKey, resetKey) ? clampView(ownZoom, extent) : null);
    }
  }

  const view = (zoom && clampView(zoom, extent)) ?? extent;

  // While dragging, follow the mouse on the whole window: the rectangle
  // stops at the plot's edges, and a release anywhere ends the drag.
  useEffect(() => {
    if (!drag) return;
    const { el, plot, region, start, width: w, height: h } = drag;

    const at = (ev: MouseEvent) => {
      const box = el.getBoundingClientRect();
      return {
        x: clamp(((ev.clientX - box.left) / box.width) * w, plot.left, plot.left + plot.width),
        y: clamp(((ev.clientY - box.top) / box.height) * h, plot.top, plot.top + plot.height),
        cssPerPx: box.width / w,
      };
    };
    const axesTo = (end: { x: number; y: number; cssPerPx: number }) =>
      dragAxes(
        region,
        Math.abs(end.x - start.x) * end.cssPerPx,
        Math.abs(end.y - start.y) * end.cssPerPx,
      );
    const stop = () => {
      setDrag(null);
      setSelection(null);
    };
    const onMove = (ev: MouseEvent) => {
      // Released where no mouseup reached the page (another window).
      if (ev.buttons === 0) {
        stop();
        return;
      }
      const end = at(ev);
      const axes = axesTo(end);
      setSelection(axes && selectionBox(plot, start, end, axes, end.cssPerPx));
    };
    const onUp = (ev: MouseEvent) => {
      const end = at(ev);
      stop();
      const axes = axesTo(end);
      const next = axes && zoomTo(drag, end, axes);
      if (next) {
        drag.setZoom(next);
        drag.onZoom?.();
      }
    };
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") stop();
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("keydown", onKey);
    };
  }, [drag]);

  function onMouseDown(e: React.MouseEvent<Element>) {
    if (e.button !== 0) return;
    const el = e.currentTarget;
    const box = el.getBoundingClientRect();
    const start = {
      x: ((e.clientX - box.left) / box.width) * width,
      y: ((e.clientY - box.top) / box.height) * height,
    };
    const hit = regionAt(plots, start.x, start.y);
    if (!hit) return;
    // No text selection while dragging across the page.
    e.preventDefault();
    setDrag({ el, ...hit, start, view, extent, yDown, width, height, setZoom, onZoom });
  }

  /** The cursor at a hover position (logical pixels): a crosshair on a plot,
   * a resize arrow on an axis strip, and the drag's all along a drag. */
  function cursorAt(pos: Point | null): string | undefined {
    const region = drag?.region ?? (pos ? regionAt(plots, pos.x, pos.y)?.region : undefined);
    if (region === "x") return "ew-resize";
    if (region === "y") return "ns-resize";
    return region === "plot" ? "crosshair" : undefined;
  }

  const reset = () => setZoom(null);

  return {
    /** What the plot shows: the zoom, or the full extent. */
    view,
    zoomed: view !== extent,
    /** The rectangle or band being dragged, for ZoomSelection. */
    selection,
    reset,
    onMouseDown,
    onDoubleClick: reset,
    cursorAt,
  };
}

/** The cells [first, end) of a regular grid that a view [lo, hi] shows: `n`
 * cells whose edges run linearly from `a` (edge 0) to `b` (edge n), in either
 * direction. Clamped to the grid. */
export function visibleCells(n: number, a: number, b: number, lo: number, hi: number): [number, number] {
  const ka = ((lo - a) / (b - a)) * n;
  const kb = ((hi - a) / (b - a)) * n;
  return [Math.max(0, Math.floor(Math.min(ka, kb))), Math.min(n, Math.ceil(Math.max(ka, kb)))];
}

/** The columns [first, end) that a view [lo, hi] shows, of a grid whose
 * ascending cell edges (one more than its columns) are `edges`. */
export function visibleColumns(edges: number[], lo: number, hi: number): [number, number] {
  let first = 0;
  while (first < edges.length - 1 && edges[first + 1] < lo) first++;
  let end = edges.length - 1;
  while (end > first && edges[end - 1] > hi) end--;
  return [first, end];
}

/** Min and max of `grid[i][j]` over columns [i0, i1) and entries [j0, j1),
 * nulls skipped; null when there is no value. A colour scale fitted to what
 * a zoom shows, or with the full ranges, to all the data. */
export function valueRange(
  grid: (number | null)[][],
  i0: number,
  i1: number,
  j0: number,
  j1: number,
): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = i0; i < i1; i++) {
    const column = grid[i];
    if (!column) continue;
    for (let j = j0; j < j1; j++) {
      const v = column[j];
      if (v == null) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo <= hi ? [lo, hi] : null;
}

/** `n` + 1 evenly spaced ticks from `lo` to `hi`, both ends included. */
export function evenTicks(lo: number, hi: number, n: number): number[] {
  return Array.from({ length: n + 1 }, (_, i) => lo + (i / n) * (hi - lo));
}

/** Decimals that keep ticks `step` apart distinct in their labels, and at least `min`. */
export function tickDecimals(step: number, min = 0): number {
  if (!(step > 0) || !Number.isFinite(step)) return min;
  return Math.min(8, Math.max(min, Math.ceil(-Math.log10(step) - 1e-9)));
}

/** Indices of up to `max` + 1 values of an ascending grid inside [lo, hi],
 * evenly spread, both ends of that stretch included. Empty when none is inside. */
export function gridTickIndices(grid: number[], lo: number, hi: number, max: number): number[] {
  // The full view's ends are the grid's own ends, give or take rounding.
  const eps = (hi - lo) * 1e-9;
  let first = -1;
  let last = -1;
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] >= lo - eps && grid[i] <= hi + eps) {
      if (first < 0) first = i;
      last = i;
    }
  }
  if (first < 0) return [];
  const count = Math.min(max, last - first);
  if (count === 0) return [first];
  return Array.from({ length: count + 1 }, (_, i) => first + Math.round((i / count) * (last - first)));
}

/** A section's position-axis ticks: at the grid positions on show, as in the
 * full view (up to 9, one decimal), or at even steps once a zoom shows fewer
 * than two of them. */
export function positionTicks(positions: number[], x0: number, x1: number): { p: number; label: string }[] {
  const idx = gridTickIndices(positions, x0, x1, 8);
  if (idx.length >= 2 || idx.length === positions.length) {
    return idx.map((k) => ({ p: positions[k], label: positions[k].toFixed(1) }));
  }
  const decimals = tickDecimals((x1 - x0) / 6, 1);
  return evenTicks(x0, x1, 6).map((p) => ({ p, label: p.toFixed(decimals) }));
}
