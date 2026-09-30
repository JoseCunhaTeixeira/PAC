import type { Tip } from "./tips";

export function HoverTooltip({ x, y, tip }: { x: number; y: number; tip: Tip }) {
  return (
    <div className="viz-tooltip" style={{ left: x + 10, top: y + 10 }}>
      <TipLines tip={tip} />
    </div>
  );
}

// A field's bound on a line of its own ("≥ 0 s", "< 1,000 Hz"): the whole line in yellow, without
// a bullet.
const BOUND = /^[<>≤≥] /;
// A bound within a sentence ("when > 1 mode", "≥ 6 dB", "≤ 20 %"), its unit with it: in yellow too.
const UNITS = "%|ms|s|kHz|Hz|m/s|m|dB|°|windows?|receivers?|samples?|records?|traces?|points?|segments?|layers?|curves?|modes?";
const BOUNDS = new RegExp(`([<>≤≥] ?[−-]?\\d(?:[\\d.,]*\\d)?(?: ?(?:${UNITS})(?![A-Za-z]))?)`);

function Bounded({ text }: { text: string }) {
  const parts = text.split(BOUNDS);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 ? (
          <span key={i} className="tip-bound">
            {part}
          </span>
        ) : (
          part
        ),
      )}
    </>
  );
}

export function TipLines({ tip }: { tip: Tip }) {
  return (
    <>
      {tip.title && (
        <b style={{ display: "block" }}>
          <Bounded text={tip.title} />
        </b>
      )}
      {tip.values && (
        <div>
          <Bounded text={tip.values} />
        </div>
      )}
      {tip.notes?.map((note, i) =>
        BOUND.test(note) ? (
          <div key={i} className="tip-bound">
            {note}
          </div>
        ) : (
          <div key={i}>
            • <Bounded text={note} />
          </div>
        ),
      )}
    </>
  );
}

/** A hint's lines (an element's `data-tip`, a cell's hover): its name, then a bullet a line. */
export function TooltipLines({ lines }: { lines: readonly string[] }) {
  return <TipLines tip={{ title: lines[0], notes: lines.slice(1) }} />;
}
