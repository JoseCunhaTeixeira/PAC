export function HoverTooltip({
  x,
  y,
  lines,
}: {
  x: number;
  y: number;
  lines: string[];
}) {
  return (
    <div className="viz-tooltip" style={{ left: x + 10, top: y + 10 }}>
      <TooltipLines lines={lines} />
    </div>
  );
}

/** A tooltip's lines, as every plot says them: what is under the pointer first, in bold, then
 * one bullet a line; a field's bound ("≥ 0 s", "< 1,000 Hz") in yellow, without one. */
const BOUND = /^[<>≤≥] /;

export function TooltipLines({ lines }: { lines: readonly string[] }) {
  return (
    <>
      {lines.map((line, i) =>
        BOUND.test(line) ? (
          <div key={i} className="tip-bound">
            {line}
          </div>
        ) : i === 0 ? (
          <b key={i} style={{ display: "block" }}>
            {line}
          </b>
        ) : (
          <div key={i}>• {line}</div>
        ),
      )}
    </>
  );
}
