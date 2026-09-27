import { useEffect, useRef, type ReactNode } from "react";

// The windows along the line as a rail of cells, one per position: its state a colour, the
// selected ones outlined. The picking page selects one (a click); the inversion pages paint
// several: press on a position and drag across the others, all set as the first one becomes.

/** pass / warn / fail: the checks' verdict (passed, flagged, rejected); hand: picked by hand;
 * auto: picked automatically, not judged; none: not picked, or not inverted; off: cannot be
 * chosen. */
export type RailTone = "pass" | "warn" | "fail" | "hand" | "auto" | "none" | "off";

export interface RailCell {
  xmid: number;
  tone: RailTone;
  title: string;
}

/** What the rail's colours mean: groups of swatches, a titled group outlined ("Automatic":
 * passed, flagged, rejected), the others standing alone. */
export function RailLegend({
  groups,
  children,
}: {
  groups: { title?: string; items: [RailTone, string][] }[];
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
    </div>
  );
}

export function PositionRail({
  cells,
  isActive,
  onClick,
  onPaint,
}: {
  cells: RailCell[];
  isActive: (xmid: number) => boolean;
  onClick?: (xmid: number) => void;
  /** Several positions at once: each cell the drag crosses set on or off. */
  onPaint?: (xmid: number, on: boolean) => void;
}) {
  // What the drag sets the cells it crosses to, while the button is held.
  const painting = useRef<boolean | null>(null);
  useEffect(() => {
    const stop = () => {
      painting.current = null;
    };
    window.addEventListener("mouseup", stop);
    return () => window.removeEventListener("mouseup", stop);
  }, []);

  if (cells.length === 0) return null;
  const first = cells[0].xmid;
  const last = cells[cells.length - 1].xmid;
  const middle = cells[Math.floor(cells.length / 2)].xmid;
  return (
    <div className="rail">
      <div
        className={`rail-cells${onPaint ? " paint" : ""}`}
        role="listbox"
        aria-label="Positions along the line"
        aria-multiselectable={onPaint ? true : undefined}
      >
        {cells.map((cell) => (
          <button
            key={cell.xmid}
            type="button"
            role="option"
            aria-selected={isActive(cell.xmid)}
            disabled={cell.tone === "off"}
            data-tip={cell.title}
            data-tip-place="above"
            className={`rail-cell ${cell.tone}${isActive(cell.xmid) ? " active" : ""}`}
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
          />
        ))}
      </div>
      <div className="rail-axis">
        <span>{first.toFixed(1)} m</span>
        {cells.length > 2 && <span>{middle.toFixed(1)} m</span>}
        <span>{last.toFixed(1)} m</span>
      </div>
    </div>
  );
}
