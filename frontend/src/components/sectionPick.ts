import { useRef } from "react";
import { CLICK_PX } from "./useZoom";

// What the line's plots share with Visualization's selected window: a dashed line down a
// section at its position, and a click (not a zoom's drag) on a column to select another.

/** A dashed line from `top` to `bottom` at `x`, readable on any colormap. */
export function drawMarker(ctx: CanvasRenderingContext2D, x: number, top: number, bottom: number) {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x, top);
  ctx.lineTo(x, bottom);
  ctx.lineWidth = 3.5;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
  ctx.stroke();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "rgba(15, 15, 18, 0.95)";
  ctx.setLineDash([5, 4]);
  ctx.stroke();
  ctx.restore();
}

// A section clicked stays where it is on screen while the page above it changes (the selected
// window's card reloading above the sections): for HOLD_MS, each change of the page's height
// scrolls it back, the browser's own anchoring not always keeping it. Scrolling lets it go.
const HOLD_MS = 2500;
let held: { element: Element; top: number; until: number } | null = null;
let watcher: ResizeObserver | null = null;

function release() {
  held = null;
}

function holdInPlace(element: Element) {
  held = { element, top: element.getBoundingClientRect().top, until: performance.now() + HOLD_MS };
  if (watcher) return;
  watcher = new ResizeObserver(() => {
    if (!held) return;
    if (performance.now() > held.until || !held.element.isConnected) {
      release();
      return;
    }
    const drift = held.element.getBoundingClientRect().top - held.top;
    if (Math.abs(drift) >= 1) window.scrollBy(0, drift);
  });
  watcher.observe(document.body);
  for (const input of ["wheel", "touchstart", "keydown"]) {
    window.addEventListener(input, release, { passive: true });
  }
}

/** The canvas's mouse handlers, `onMouseDown` (the zoom's) kept: a click that did not drag
 * calls `onClick` with the logical position under the mouse (the drawing's own coordinates,
 * `width` by `height`), the section held where it is on screen. */
export function useClick(
  width: number,
  height: number,
  onClick: ((x: number, y: number) => void) | undefined,
  onMouseDown: (e: React.MouseEvent<HTMLCanvasElement>) => void,
) {
  const down = useRef<{ x: number; y: number } | null>(null);
  return {
    onMouseDown: (e: React.MouseEvent<HTMLCanvasElement>) => {
      down.current = { x: e.clientX, y: e.clientY };
      onMouseDown(e);
    },
    onClick: (e: React.MouseEvent<HTMLCanvasElement>) => {
      const at = down.current;
      down.current = null;
      if (!onClick || !at || Math.hypot(e.clientX - at.x, e.clientY - at.y) >= CLICK_PX) return;
      const box = e.currentTarget.getBoundingClientRect();
      holdInPlace(e.currentTarget);
      onClick(((e.clientX - box.left) / box.width) * width, ((e.clientY - box.top) / box.height) * height);
    },
  };
}
