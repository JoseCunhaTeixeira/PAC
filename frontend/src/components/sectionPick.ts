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

/** The canvas's mouse handlers, `onMouseDown` (the zoom's) kept: a click that did not drag
 * calls `onClick` with the logical position under the mouse (the drawing's own coordinates,
 * `width` by `height`). */
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
      onClick(((e.clientX - box.left) / box.width) * width, ((e.clientY - box.top) / box.height) * height);
    },
  };
}
