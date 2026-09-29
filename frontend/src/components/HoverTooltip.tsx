import type { Tip } from "./tips";

export function HoverTooltip({ x, y, tip }: { x: number; y: number; tip: Tip }) {
  return (
    <div className="viz-tooltip" style={{ left: x + 10, top: y + 10 }}>
      <TipLines tip={tip} />
    </div>
  );
}

// A field's bound ("≥ 0 s", "< 1,000 Hz"): in yellow, without a bullet.
const BOUND = /^[<>≤≥] /;

export function TipLines({ tip }: { tip: Tip }) {
  return (
    <>
      {tip.title && <b style={{ display: "block" }}>{tip.title}</b>}
      {tip.values && <div>{tip.values}</div>}
      {tip.notes?.map((note, i) =>
        BOUND.test(note) ? (
          <div key={i} className="tip-bound">
            {note}
          </div>
        ) : (
          <div key={i}>• {note}</div>
        ),
      )}
    </>
  );
}

/** A hint's lines (an element's `data-tip`, a cell's hover): its name, then a bullet a line. */
export function TooltipLines({ lines }: { lines: readonly string[] }) {
  return <TipLines tip={{ title: lines[0], notes: lines.slice(1) }} />;
}
