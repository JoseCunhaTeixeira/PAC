import type { ReactNode } from "react";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  DepthIcon,
  ImageIcon,
  OutcropIcon,
  PulseIcon,
} from "../icons";
import { neighbours } from "./cells";
import { metricLabel, STATE_MEANINGS } from "./format";
import type { Card, Cell, GateView, Overview, Sentence, StageKey } from "./types";
import {
  Empty,
  ErrorBox,
  Fold,
  GateTables,
  PartBadge,
  Sentences,
  SettingsList,
  Skeleton,
  StatusBadge,
} from "./ui";

// What every stage's panel is made of: its summary, and the selected unit's card, with its
// neighbours a click away. A card's badges say its verdict; it shows the unit, then, folded, what
// it says (why it was rejected or flagged first) and everything its display was made with (its
// settings; how the line's other units differ in each).

/** A unit no gate judged (a run made by hand), its measures beyond the assistant's limits
 * named in one warning, as a verdict would say them; none beyond, none said. */
function beyondLimits(gates: GateView[]): Sentence | null {
  const beyond = gates
    .filter((gate) => gate.verdict === null && !gate.by_hand)
    .flatMap((gate) => gate.metrics)
    .filter((metric) => metric.threshold != null && metric.bound != null && !metric.passed);
  if (beyond.length === 0) return null;
  // Said within the sentence: lower case, but an acronym (SNR).
  const names = [...new Set(beyond.map((metric) => metricLabel(metric.name)))].map((name) =>
    /^.[a-z]/.test(name) ? name.charAt(0).toLowerCase() + name.slice(1) : name,
  );
  return { mark: "warn", text: `Beyond the agent's limits: ${names.join(", ")}.` };
}

/** The stage along the line in a line, and its line check's measures, in the same menu as a
 * unit's. */
export function StageHead({ overview, error }: { overview: Overview | null; error: string | null }) {
  if (error) return <ErrorBox message={error} />;
  if (!overview) return <Skeleton height={54} />;
  return (
    <div className="viz-stage-head">
      <p
        className="viz-summary"
        data-tip={
          !overview.paco && overview.cells.length > 0
            ? "By hand\nMeasured against the agent's limits, not judged"
            : undefined
        }
      >
        {overview.summary}
      </p>
      <GateTables gates={overview.gates ?? []} />
    </div>
  );
}

// A window's card's icon, as its stage's pages show it.
const STAGE_ICONS: Record<StageKey, ReactNode> = {
  records: <PulseIcon size={17} />,
  dispersion: <ImageIcon size={17} />,
  inversion: <DepthIcon size={17} />,
  petro: <OutcropIcon size={17} />,
};

/** A unit before and after the one shown, a click away: its key, and its name on the arrow's
 * hover. */
export interface Step {
  key: string;
  name: string;
}

/** The arrows to the units before and after (← and → too, as the page steps them). */
export function StepArrows({
  before,
  after,
  onSelect,
}: {
  before: Step | null;
  after: Step | null;
  onSelect: (key: string) => void;
}) {
  return (
    <span className="viz-nav">
      <button
        type="button"
        className="secondary icon"
        aria-label="Previous"
        disabled={!before}
        data-tip={before ? `Previous (←)\n${before.name}` : undefined}
        onClick={() => before && onSelect(before.key)}
      >
        <ArrowLeftIcon size={16} />
      </button>
      <button
        type="button"
        className="secondary icon"
        aria-label="Next"
        disabled={!after}
        data-tip={after ? `Next (→)\n${after.name}` : undefined}
        onClick={() => after && onSelect(after.key)}
      >
        <ArrowRightIcon size={16} />
      </button>
    </span>
  );
}

const step = (cell: Cell | null): Step | null => (cell ? { key: cell.key, name: cell.hover[0] } : null);

export function UnitCard({
  card,
  error,
  selected,
  loading = false,
  lead = [],
  cells,
  onSelect,
  children,
  details,
  stage,
  bare = false,
}: {
  card: Card | null;
  error: string | null;
  /** The unit selected, which the arrows step from: while its card loads, the last one shown
   * stays in its place, faded (`loading`), so the page does not move. */
  selected: string | null;
  loading?: boolean;
  /** Sentences said before the card's own (the window's shots). */
  lead?: Sentence[];
  cells: Cell[];
  onSelect: (key: string) => void;
  children?: ReactNode;
  details?: ReactNode;
  /** The stage the card is of: what its state means there. */
  stage?: StageKey;
  /** Its name and its neighbours alone above its displays, and its details (a passive record's
   * signal and spectrum, and its measures). */
  bare?: boolean;
}) {
  const { before, after } = neighbours(cells, selected ?? card?.key ?? null);
  const arrows = <StepArrows before={step(before)} after={step(after)} onSelect={onSelect} />;
  if (error) {
    return (
      <section className="viz-card">
        <div className="viz-unit-head">{arrows}</div>
        <ErrorBox message={error} />
      </section>
    );
  }
  if (!card) {
    return (
      <section className="viz-card">
        <Skeleton height={22} style={{ width: "40%", marginBottom: 14 }} />
        <Skeleton height={16} style={{ marginBottom: 8 }} />
        <Skeleton height={16} style={{ width: "80%", marginBottom: 8 }} />
        <Skeleton height={16} style={{ width: "65%" }} />
      </section>
    );
  }
  // The verdict is the badges': folded under the unit, why it was rejected or flagged first,
  // then the rest the card says. The measures in their tables, the sentences say what they do not.
  const verdict = card.verdict ?? beyondLimits(card.gates);
  const said = [...(verdict && verdict.mark !== "pass" ? [verdict] : []), ...lead, ...card.sentences];
  return (
    <section className={`viz-card${loading ? " viz-stale" : ""}`}>
      <div className="viz-unit-head">
        {stage && <span className="card-icon">{STAGE_ICONS[stage]}</span>}
        <strong>{card.title}</strong>
        {bare
          ? null
          : card.parts?.length
          ? // Its checks said apart (a window's image, its curve): a badge each.
            card.parts.map((part) => <PartBadge key={part.label} part={part} />)
          : card.status !== "none" && (
              <StatusBadge status={card.status} meaning={stage ? STATE_MEANINGS[stage][card.status].full : undefined} />
            )}
        {arrows}
      </div>
      {children}
      {!bare && said.length > 0 && (
        <Fold title="Details">
          <Sentences sentences={said} />
        </Fold>
      )}
      {!bare && card.settings && card.settings.length > 0 && (
        <Fold title="Settings">
          <SettingsList settings={card.settings} />
        </Fold>
      )}
      {details}
    </section>
  );
}

export function NoRun({ children }: { children: ReactNode }) {
  return (
    <section className="viz-card">
      <Empty>{children}</Empty>
    </section>
  );
}
