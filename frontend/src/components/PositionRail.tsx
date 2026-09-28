import { useEffect, useRef, type ReactNode } from "react";
import { cellMetres, cellWidth, COUNT_MIN_W } from "./lineDraw";
import { useContainerWidth } from "./useContainerWidth";

// The windows along the line as a rail of cells across the card, each at its window's middle (a
// window missing, a gap), a receiver's spacing wide whatever the windows' step (a larger step,
// larger gaps): its state a colour, the selected ones outlined, in its curve's band how many modes
// were picked in it when more than one; under them, the first, middle and last windows'
// positions. The picking page selects one (a click); the inversion pages paint several: press on
// a position and drag across the others, all set as the first one becomes.

/** pass / warn / fail: the checks' verdict (passed, flagged, rejected); hand: picked by hand;
 * auto: picked automatically, not judged; none: not picked, or not inverted; off: cannot be
 * chosen. */
export type RailTone = "pass" | "warn" | "fail" | "hand" | "auto" | "none" | "off";

export interface RailCell {
  xmid: number;
  tone: RailTone;
  title: string;
  /** Its checks said apart, top down (a window's image, its curve): a band each. */
  parts?: RailTone[];
  /** The modes picked in it: how many, written in its curve's band when more than one. */
  modes?: string[];
}

/** What the rail's colours mean: groups of swatches, a titled group outlined ("Automatic":
 * passed, flagged, rejected), the others standing alone. */
export function RailLegend({
  groups,
  modes = false,
  children,
}: {
  groups: { title?: string; items: [RailTone, string][] }[];
  /** The count of modes a cell writes, when some cell picked more than one. */
  modes?: boolean;
  children?: ReactNode;
}) {
  return (
    <div className="rail-legend">
      {children}
      {groups.map((group, i) => (
        <span key={i} className={`rail-legend-group${group.title ? " titled" : ""}`}>
          {group.title && <b>{group.title}</b>}
          {group.items.map(([tone, label]) => (
            <span key={tone}>
              <i className={tone} />
              {label}
            </span>
          ))}
        </span>
      ))}
      {modes && (
        <span className="rail-legend-group">
          <span data-tip="How many modes were picked in a window, written in its curve's band when more than one">
            <i className="count">2</i>
            more than one mode picked
          </span>
        </span>
      )}
    </div>
  );
}

/** The windows' span: the first's middle to the last's, half a cell's share around. */
function windowsSpan(xmids: readonly number[], metres: number): [number, number] {
  const half = Number.isFinite(metres) && metres > 0 ? metres / 2 : 1;
  return [Math.min(...xmids) - half, Math.max(...xmids) + half];
}

/** The first window's position under its start, the last's under its end, the middle's under
 * its middle; `left`, a cell's left, `width` its width. */
function RailAxis({
  cells,
  left,
  width,
}: {
  cells: RailCell[];
  left: (xmid: number) => number;
  width: number;
}) {
  const first = cells[0].xmid;
  const last = cells[cells.length - 1].xmid;
  const middle = cells[Math.floor(cells.length / 2)].xmid;
  return (
    <div className="rail-axis">
      <span style={{ left: left(first) }}>{first.toFixed(1)} m</span>
      {cells.length > 2 && (
        <span className="middle" style={{ left: left(middle) + width / 2 }}>
          {middle.toFixed(1)} m
        </span>
      )}
      <span className="last" style={{ left: left(last) + width }}>
        {last.toFixed(1)} m
      </span>
    </div>
  );
}

export function PositionRail({
  cells,
  receivers,
  isActive,
  onClick,
  onPaint,
}: {
  cells: RailCell[];
  /** The line's receivers: a cell is their spacing wide (the windows' tightest without them). */
  receivers: readonly number[];
  isActive: (xmid: number) => boolean;
  onClick?: (xmid: number) => void;
  /** Several positions at once: each cell the drag crosses set on or off. */
  onPaint?: (xmid: number, on: boolean) => void;
}) {
  // Its width, measured: the rail is there from the start, its cells as they come.
  const [railRef, width] = useContainerWidth<HTMLDivElement>();
  // What the drag sets the cells it crosses to, while the button is held.
  const painting = useRef<boolean | null>(null);
  useEffect(() => {
    const stop = () => {
      painting.current = null;
    };
    window.addEventListener("mouseup", stop);
    return () => window.removeEventListener("mouseup", stop);
  }, []);

  const xmids = cells.map((cell) => cell.xmid);
  const [x0, x1] = windowsSpan(xmids, cellMetres(xmids, receivers));
  const w = cellWidth(xmids, receivers, x1 - x0, width);
  const left = (xmid: number) => ((xmid - x0) / (x1 - x0)) * width - w / 2;
  return (
    <div className="rail" ref={railRef}>
      {cells.length > 0 && width > 0 && (
        <>
          <div className="rail-lane">
            <div
              className={`rail-cells${onPaint ? " paint" : ""}`}
              role="listbox"
              aria-label="Windows along the line"
              aria-multiselectable={onPaint ? true : undefined}
            >
              {cells.map((cell) => {
                // How many modes were picked in it, in its curve's band, when more than one (a
                // cell wide enough).
                const n = cell.modes?.length ?? 0;
                const count = n > 1 && w >= COUNT_MIN_W ? n : null;
                return (
                  <button
                    key={cell.xmid}
                    type="button"
                    role="option"
                    aria-selected={isActive(cell.xmid)}
                    disabled={cell.tone === "off"}
                    data-tip={cell.title}
                    data-tip-place="above"
                    className={`rail-cell ${cell.parts ? "split" : cell.tone}${isActive(cell.xmid) ? " active" : ""}`}
                    style={{ left: left(cell.xmid), width: w }}
                    onClick={onPaint ? undefined : () => onClick?.(cell.xmid)}
                    onMouseDown={
                      onPaint
                        ? (e) => {
                            e.preventDefault(); // no text selection while painting
                            const on = !isActive(cell.xmid);
                            painting.current = on;
                            onPaint(cell.xmid, on);
                          }
                        : undefined
                    }
                    onMouseEnter={
                      onPaint
                        ? () => {
                            if (painting.current !== null && cell.tone !== "off") onPaint(cell.xmid, painting.current);
                          }
                        : undefined
                    }
                  >
                    {cell.parts
                      ? cell.parts.map((part, i, parts) => (
                          <i key={i} className={part}>
                            {i === parts.length - 1 && count}
                          </i>
                        ))
                      : count}
                  </button>
                );
              })}
            </div>
          </div>
          <RailAxis cells={cells} left={left} width={w} />
        </>
      )}
    </div>
  );
}
