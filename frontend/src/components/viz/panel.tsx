import type { ReactNode } from "react";
import { neighbours } from "./cells";
import { STATE_MEANINGS } from "./format";
import type { Card, Cell, Overview, Sentence, StageKey } from "./types";
import { Empty, ErrorBox, Fold, GateBadge, Sentences, SettingsList, Skeleton, StatusBadge } from "./ui";

// What every stage's panel is made of: its summary and the settings it ran with, and the
// selected unit's card, with its neighbours a click away. A card says its verdict and its
// warnings; the rest of what it says is folded under them.

// The marks said at once: what went wrong or needs a look.
const LOUD = new Set<Sentence["mark"]>(["warn", "fail"]);

export function StageHead({
  overview,
  error,
  aside,
}: {
  overview: Overview | null;
  error: string | null;
  /** A choice for the whole stage (the model shown), on the summary's right. */
  aside?: ReactNode;
}) {
  if (error) return <ErrorBox message={error} />;
  if (!overview) return <Skeleton height={54} />;
  return (
    <section className="viz-card">
      <div className="viz-row">
      <p
        className="viz-summary"
        data-tip={
          !overview.paco && overview.cells.length > 0
            ? "Made by hand\nNo check judged it\nEach measure is shown against the assistant's limits"
            : undefined
        }
      >
        {overview.summary}
      </p>
      {aside}
      </div>
      {overview.settings.length > 0 && (
        <Fold title="Settings, and why">
          <SettingsList settings={overview.settings} />
        </Fold>
      )}
    </section>
  );
}

export function UnitCard({
  card,
  error,
  lead = [],
  cells,
  onSelect,
  children,
  details,
  stage,
}: {
  card: Card | null;
  error: string | null;
  /** Sentences said before the card's own (the window's shots). */
  lead?: Sentence[];
  cells: Cell[];
  onSelect: (key: string) => void;
  children?: ReactNode;
  details?: ReactNode;
  /** The stage the card is of: what its state means there. */
  stage?: StageKey;
}) {
  if (error) return <ErrorBox message={error} />;
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
  const { before, after } = neighbours(cells, card.key);
  const said = [...lead, ...card.sentences];
  // The verdict and the warnings at once, the rest folded; with neither, the first line shown.
  const loud = said.filter((one) => LOUD.has(one.mark));
  const quiet = said.filter((one) => !LOUD.has(one.mark));
  const opener = card.verdict || loud.length ? [] : quiet.slice(0, 1);
  const shown = [...(card.verdict ? [card.verdict] : []), ...opener, ...loud];
  const folded = quiet.slice(opener.length);
  const gates = card.gates.filter((gate) => gate.verdict !== null || gate.by_hand);
  return (
    <section className="viz-card">
      <div className="viz-unit-head">
        <strong>{card.title}</strong>
        {card.status !== "none" && (
          <StatusBadge status={card.status} meaning={stage ? STATE_MEANINGS[stage][card.status].full : undefined} />
        )}
        {gates.length > 0 && (
          <span className="viz-gate-list">
            {gates.map((gate) => (
              <GateBadge key={gate.gate} gate={gate} />
            ))}
          </span>
        )}
        <span className="viz-nav">
          <button
            type="button"
            className="viz-icon"
            disabled={!before}
            data-tip={before ? `Previous (←)\n${before.hover[0]}` : undefined}
            onClick={() => before && onSelect(before.key)}
          >
            ←
          </button>
          <button
            type="button"
            className="viz-icon"
            disabled={!after}
            data-tip={after ? `Next (→)\n${after.hover[0]}` : undefined}
            onClick={() => after && onSelect(after.key)}
          >
            →
          </button>
        </span>
      </div>
      <Sentences sentences={shown} />
      {folded.length > 0 && (
        <div className="viz-more">
          <Fold title="More details">
            <Sentences sentences={folded} />
          </Fold>
        </div>
      )}
      {card.status === "none" && (before || after) && (
        <p className="viz-small viz-muted" style={{ margin: "10px 0 0" }}>
          Nearest with a result:{" "}
          {[before, after]
            .filter((cell): cell is Cell => cell !== null && cell.status !== "none")
            .map((cell, i) => (
              <span key={cell.key}>
                {i > 0 && " · "}
                <a href="#" onClick={(e) => (e.preventDefault(), onSelect(cell.key))}>
                  {cell.hover[0]}
                </a>
              </span>
            ))}
        </p>
      )}
      {children}
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
