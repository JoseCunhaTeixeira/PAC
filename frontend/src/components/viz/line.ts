import type { Range, View, ZoomLink } from "../useZoom";
import type { RunCard } from "./types";

// What the plots along the line share: their extent (every receiver, shot and window of the
// line, a margin around) and one zoom along it, so that they stay aligned.

/** The line's extent along x: every receiver, shot and window, a margin around. */
export function lineExtent(card: RunCard): Range {
  const xs = [
    ...card.receivers,
    ...Object.values(card.sources),
    ...card.windows.flatMap((window) => [window.first, window.last]),
  ];
  if (xs.length === 0) return [0, 1];
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  const pad = (hi - lo) * 0.015 || 1;
  return [lo - pad, hi + pad];
}

/** A zoom link sharing only x (`x`, null for the whole line) with the other plots along the
 * line; y, a plot's own (`y`, null for all of it). */
export function alongLine(
  x: Range | null,
  setX: (x: Range | null) => void,
  y: Range | null,
  setY: (y: Range | null) => void,
  extent: View,
): ZoomLink {
  return {
    zoom: x || y ? { x: x ?? extent.x, y: y ?? extent.y } : null,
    setZoom: (view) => {
      setX(view && !same(view.x, extent.x) ? view.x : null);
      setY(view && !same(view.y, extent.y) ? view.y : null);
    },
  };
}

function same(a: Range, b: Range): boolean {
  return Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9;
}
