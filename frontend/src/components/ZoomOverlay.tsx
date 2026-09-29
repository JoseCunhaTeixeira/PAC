import type { SelectionBox } from "./useZoom";
import "./zoom.css";

// What useZoom shows over a plot: the rectangle being dragged, absolutely positioned in a
// `position: relative` wrapper whose top-left corner is the drawing's. The plot's tools are its
// box's (kit's BoxTools), off the drawing.

/** The rectangle being dragged: a translucent accent fill, an accent border. */
export function ZoomSelection({ box }: { box: SelectionBox | null }) {
  if (!box) return null;
  return (
    <div
      className="zoom-selection"
      style={{ left: box.left, top: box.top, width: box.width, height: box.height }}
    />
  );
}
