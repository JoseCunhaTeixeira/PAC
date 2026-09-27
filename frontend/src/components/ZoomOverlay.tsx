import type { CSSProperties } from "react";
import type { SelectionBox } from "./useZoom";
import "./zoom.css";

// What useZoom shows over a plot, both absolutely positioned in a
// `position: relative` wrapper whose top-left corner is the drawing's.

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

/** Back to the full view, shown only while zoomed; `style` puts it in a
 * corner of the image that holds no data. */
export function ZoomReset({
  zoomed,
  onReset,
  style,
}: {
  zoomed: boolean;
  onReset: () => void;
  style: CSSProperties;
}) {
  if (!zoomed) return null;
  return (
    <button type="button" className="zoom-reset" onClick={onReset} style={style}>
      Reset zoom
    </button>
  );
}
