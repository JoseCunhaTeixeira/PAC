import { useState } from "react";
import { BY_WORDS, MARK_ICONS, runDate } from "./format";
import { Fold, GateTables, OriginTag } from "./ui";
import type {
  HistoryElement,
  HistoryKind,
  HistoryVersion,
  Mark,
  Status,
  UnitHistory,
  VersionState,
} from "./types";

// A unit's history, the same on every page that shows the unit: its steps in the pipeline's
// order, each its version now and, folded under it, those it replaced; under each version the
// checks that judged it, or its measures against the agent's limits. The kinds it holds (its
// steps, each check, the measures) can each be hidden; the choice is kept for every history.

const HIDDEN_KEY = "pac.history.hidden";
// A version's state, as its step's badge says it.
const STATE_LABELS: Record<VersionState, string> = {
  current: "now",
  replaced: "replaced",
  "set aside": "set aside",
  deleted: "deleted",
  failed: "failed",
};
// What ended a version, before its date.
const ENDED_LABELS: Record<VersionState, string> = {
  current: "",
  replaced: "Replaced",
  "set aside": "Set aside",
  deleted: "Deleted",
  failed: "Failed",
};
// A check's state, as its mark says it: a check's flag as the cards' flags, a result beyond the
// agent's limits as a warning.
const TONE_MARKS: Record<Status, Mark> = { pass: "pass", warn: "flag", fail: "fail", note: "warn", none: "info" };

function storedHidden(): Set<string> {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(HIDDEN_KEY) ?? "[]");
    return new Set(Array.isArray(stored) ? stored.filter((one): one is string => typeof one === "string") : []);
  } catch {
    return new Set();
  }
}

function storeHidden(hidden: Set<string>) {
  try {
    localStorage.setItem(HIDDEN_KEY, JSON.stringify([...hidden]));
  } catch {
    // Not kept: the page shows every kind next time.
  }
}

function elementKind(element: HistoryElement): string {
  return element.kind === "check" ? `check:${element.key}` : "measure";
}

/** A card's history, folded under its checks' measures as each card's details are. */
export function HistoryFold({ history }: { history: UnitHistory | null | undefined }) {
  if (!history || history.steps.length === 0) return null;
  return (
    <Fold title="History">
      <History history={history} />
    </Fold>
  );
}

export function History({ history }: { history: UnitHistory }) {
  const [hidden, setHidden] = useState<Set<string>>(storedHidden);
  if (history.steps.length === 0) return null;
  const toggle = (kind: HistoryKind) => {
    const next = new Set(hidden);
    if (next.has(kind.key)) next.delete(kind.key);
    else next.add(kind.key);
    storeHidden(next);
    setHidden(next);
  };
  const steps = history.steps.filter((step) => !hidden.has(step.kind));
  return (
    <div className="viz-history">
      {history.kinds.length > 1 && (
        <div className="viz-history-kinds" role="group" aria-label="Shown in the history">
          {history.kinds.map((kind) => (
            <button
              key={kind.key}
              type="button"
              className={`viz-history-kind ${kind.group}${hidden.has(kind.key) ? " off" : ""}`}
              aria-pressed={!hidden.has(kind.key)}
              onClick={() => toggle(kind)}
            >
              {kind.label}
            </button>
          ))}
        </div>
      )}
      <ol className="viz-history-steps">
        {steps.map((step) => {
          const shown = step.current ?? step.replaced[0] ?? null;
          const older = step.current ? step.replaced : step.replaced.slice(1);
          return (
            <li key={step.key} className="viz-history-step">
              <div className="viz-history-head">
                <span className="viz-history-title">{step.title}</span>
                <span className={`viz-history-state ${step.state.replace(" ", "-")}`}>{STATE_LABELS[step.state]}</span>
              </div>
              {shown && <VersionView version={shown} hidden={hidden} />}
              {older.length > 0 && (
                <Fold title={`${older.length} earlier version${older.length > 1 ? "s" : ""}`}>
                  {older.map((version, index) => (
                    <VersionView key={index} version={version} hidden={hidden} />
                  ))}
                </Fold>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function VersionView({ version, hidden }: { version: HistoryVersion; hidden: Set<string> }) {
  const elements = version.elements.filter((element) => !hidden.has(elementKind(element)));
  return (
    <div className={`viz-version ${version.state.replace(" ", "-")}`}>
      <div className="viz-version-head">
        <span className="viz-version-title">{version.title}</span>
        <OriginTag origin={version.by} />
        {version.at && <span className="viz-muted">{runDate(version.at)}</span>}
      </div>
      {version.modules.map((module) => (
        <div key={module.key} className="viz-version-module">
          <span className="viz-version-label">{module.label}</span>
          <span>
            {module.lines.map((line) => (
              <span key={line} className="viz-version-line">
                {line}
              </span>
            ))}
          </span>
        </div>
      ))}
      {version.ended && version.state !== "current" && (
        <div className="viz-version-ended">
          {ENDED_LABELS[version.state]} {runDate(version.ended)}
          {version.ended_by ? ` by ${BY_WORDS[version.ended_by]}` : ""}
          {version.ended_why ? `: ${version.ended_why}` : ""}
        </div>
      )}
      {elements.length > 0 && (
        <ul className="viz-version-elements">
          {elements.map((element, index) => (
            <ElementView key={index} element={element} />
          ))}
        </ul>
      )}
    </div>
  );
}

function ElementView({ element }: { element: HistoryElement }) {
  const mark = TONE_MARKS[element.tone];
  const lines = element.modules.flatMap((module) => module.lines);
  return (
    <li className={`viz-element ${element.kind}`}>
      <span className={`viz-mark ${mark}`}>{MARK_ICONS[mark]}</span>
      <div>
        <div className="viz-element-title">{element.title}</div>
        {lines.map((line) => (
          <div key={line} className="viz-element-line">
            {line}
          </div>
        ))}
        {element.gate && element.gate.metrics.length > 0 && <GateTables gates={[element.gate]} />}
      </div>
    </li>
  );
}
