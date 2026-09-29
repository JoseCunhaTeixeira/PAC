import type { ReactNode } from "react";
import { BoxTools, Callout, Segmented } from "../kit";
import {
  capitalized,
  changes,
  metricLabel,
  metricLimit,
  metricValue,
  ORIGIN_LABELS,
  STATUS_LABELS,
  triggerLabel,
} from "./format";
import type {
  AttemptSummary,
  GateView,
  Mark,
  Metric,
  PartState,
  Sentence,
  Setting,
  Status,
  Verdict,
} from "./types";
import "./viz.css";

// The pieces every panel of Visualization says things with: a state's badge, a card's
// sentences, the settings a stage ran with and why, and folded under a card, each gate's
// metrics and the attempts.

const MARK_ICONS: Record<Mark, string> = { pass: "✓", warn: "!", fail: "✕", info: "i" };
const VERDICT_STATUS: Record<Verdict, Status> = { pass: "pass", retry: "warn", reject: "fail" };

// What a badge means, on hover.
const STATUS_MEANINGS: Record<Status, string> = {
  pass: "Passed every check.",
  warn: "Kept, but flagged: worth a look.",
  fail: "Rejected by a check.",
  none: "No check judged it.",
};
const STATUS_TITLES: Record<Status, string> = { pass: "Pass", warn: "Flagged", fail: "Rejected", none: "Not judged" };
const GATE_NAMES: Record<string, string> = {
  G1: "the records' signal check",
  G2: "the dispersion image's check",
  G3: "the curve's check",
  G4: "the check of the curves along the line",
  G5: "the model's check",
  G6: "the check of the models along the line",
  G7: "the soil column's check",
  G8: "the check of the soil columns along the line",
};
const VERDICT_MEANINGS: Record<Verdict, string> = {
  pass: "passed",
  retry: "still flagged after its retries",
  reject: "rejected",
};

export function StatusBadge({ status, label, meaning }: { status: Status; label?: string; meaning?: string }) {
  return (
    <span className={`viz-badge ${status}`} data-tip={`${STATUS_TITLES[status]}\n${meaning ?? STATUS_MEANINGS[status]}`}>
      {label ?? STATUS_LABELS[status]}
    </span>
  );
}

// A part's state in words, and what each part checks (its badge's hover).
const PART_WORDS: Record<PartState, string> = {
  pass: "passed",
  warn: "flagged",
  fail: "rejected",
  hand: "by hand",
  none: "not checked",
};
const PART_GATES: Record<string, string> = {
  image: "G2: the dispersion image's check",
  curve: "G3 and G4: the curve's checks, alone and along the line",
};

/** One of a unit's checks said apart (a window's image, its curve): what it checks, its state,
 * in its colour (indigo by hand). */
export function PartBadge({ part }: { part: { label: string; state: PartState } }) {
  const said =
    part.state === "none" && part.label === "curve" ? "no curve" : `${part.label} ${PART_WORDS[part.state]}`;
  const why = part.state === "hand" ? "The user's curve, passed as it is" : (PART_GATES[part.label] ?? "");
  return (
    <span className={`viz-badge ${part.state}`} data-tip={`${capitalized(said)}\n${why}`}>
      {said}
    </span>
  );
}

/** A gate's verdict, or for a run PAC made, that it measured without judging. */
export function GateBadge({ gate }: { gate: GateView }) {
  const name = GATE_NAMES[gate.gate] ?? gate.gate;
  if (gate.by_hand) {
    const passed = gate.verdict === "pass";
    return (
      <span
        className={`viz-badge ${passed ? "hand" : "none"}`}
        data-tip={`${gate.gate}: ${name}\n${passed ? "Picked by hand: passed as it is" : "Not checked: its curve was picked by hand"}`}
      >
        {gate.gate} {passed ? "by hand" : "not checked"}
      </span>
    );
  }
  if (gate.verdict === null) {
    return (
      <span className="viz-badge none" data-tip={`${gate.gate}: ${name}\nMeasured, not judged (made by hand)`}>
        {gate.gate}: measured
      </span>
    );
  }
  return (
    <span
      className={`viz-badge ${VERDICT_STATUS[gate.verdict]}`}
      data-tip={`${gate.gate}: ${name}\n${capitalized(VERDICT_MEANINGS[gate.verdict])}`}
    >
      {gate.gate} {gate.verdict}
    </span>
  );
}

/** A sentence's whole, as a tooltip: what it found in bold, then why it matters and what to do,
 * a bullet each ("finding: consequence. Advice." -> finding / consequence / advice). */
function detailTip(detail: string): string {
  const [found, ...rest] = detail.split(": ");
  const after = rest.join(": ");
  if (!after) return detail;
  return [found, ...after.split(/(?<=\.) (?=[A-Z])|; /).map((part) => capitalized(part.trim()))].join("\n");
}

export function Sentences({ sentences }: { sentences: Sentence[] }) {
  return (
    <ul className="viz-sentences">
      {sentences.map((sentence, i) => (
        <li key={i} data-tip={sentence.detail ? detailTip(sentence.detail) : undefined}>
          <span className={`viz-mark ${sentence.mark}`}>{MARK_ICONS[sentence.mark]}</span>
          <span>{sentence.text}</span>
        </li>
      ))}
    </ul>
  );
}

// Where a setting came from, on hover.
const ORIGIN_MEANINGS: Record<Setting["origin"], string> = {
  rule: "Set by the assistant's rules, from the data.",
  given: "Given in the request: by you, or chosen by the assistant.",
  default: "The preset's default.",
  pac: "Set by hand.",
};

export function OriginTag({ origin }: { origin: Setting["origin"] }) {
  return (
    <span className={`viz-origin ${origin}`} data-tip={`${capitalized(ORIGIN_LABELS[origin])}\n${ORIGIN_MEANINGS[origin]}`}>
      {ORIGIN_LABELS[origin]}
    </span>
  );
}

/** Settings as a definition list: each its value, then why. */
export function SettingsList({ settings }: { settings: Setting[] }) {
  return (
    <dl className="viz-settings">
      {settings.map((setting) => (
        <div key={setting.key} style={{ display: "contents" }}>
          <dt>{setting.label}</dt>
          <dd>
            <b>{setting.value}</b>
            {setting.detail && <span className="viz-muted">{setting.detail} </span>}
            <OriginTag origin={setting.origin} />
            <span className="viz-why">{capitalized(setting.why)}.</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function Fold({
  title,
  children,
  open,
  onToggle,
}: {
  title: ReactNode;
  children: ReactNode;
  open?: boolean;
  onToggle?: (open: boolean) => void;
}) {
  return (
    <details
      className="viz-fold"
      open={open}
      onToggle={onToggle ? (e) => onToggle((e.currentTarget as HTMLDetailsElement).open) : undefined}
    >
      <summary>{title}</summary>
      {children}
    </details>
  );
}

interface MetricRow {
  label: string;
  value: string;
  limit: string;
  judged: boolean;
  passed: boolean;
}

/** A gate's metrics as rows, a measure held between a floor and a ceiling (the acceptance
 * between 20 and 30 %) in one row, its band as its limit. */
function metricRows(metrics: Metric[]): MetricRow[] {
  const rows: MetricRow[] = [];
  const merged = new Set<number>();
  metrics.forEach((metric, i) => {
    if (merged.has(i)) return;
    const j = metrics.findIndex(
      (other, k) =>
        k > i &&
        other.name === metric.name &&
        other.value === metric.value &&
        other.threshold != null &&
        metric.threshold != null &&
        other.bound != null &&
        metric.bound != null &&
        other.bound !== metric.bound,
    );
    if (j >= 0) {
      merged.add(j);
      const [low, high] = metric.bound === "min" ? [metric, metrics[j]] : [metrics[j], metric];
      rows.push({
        label: metricLabel(metric.name),
        value: metricValue(metric),
        limit: band(low, high),
        judged: true,
        passed: low.passed && high.passed,
      });
      return;
    }
    const judged = metric.threshold != null && metric.bound != null;
    rows.push({
      label: metricLabel(metric.name),
      value: metricValue(metric),
      limit: metricLimit(metric) || "reported",
      judged,
      passed: metric.passed,
    });
  });
  return rows;
}

/** A floor and a ceiling as one band: "≥ 20, ≤ 30 %". */
function band(low: Metric, high: Metric): string {
  const from = metricValue(low, low.threshold);
  const to = metricValue(high, high.threshold);
  const unit = / (\S+)$/.exec(to)?.[1];
  return `≥ ${unit && from.endsWith(` ${unit}`) ? from.slice(0, -unit.length - 1) : from}, ≤ ${to}`;
}

/** Each gate's metrics against their limits. */
export function GateTables({ gates }: { gates: GateView[] }) {
  // A curve picked by hand has nothing measured: its badges say so.
  const measured = gates.filter((gate) => !gate.by_hand);
  if (measured.length === 0) return null;
  return (
    <div className="viz-gates">
      {measured.map((gate) => (
        <div key={gate.gate}>
          <div className="viz-gate-head">
            <GateBadge gate={gate} />
            {gate.verdict === null && (
              <span className="viz-muted viz-small">against the assistant's limits</span>
            )}
          </div>
          {gate.metrics.length === 0 ? (
            <p className="viz-muted viz-small">Nothing measured.</p>
          ) : (
            <div className="viz-table-wrap">
              <table className="viz-table">
                <thead>
                  <tr>
                    <th>Measure</th>
                    <th className="num">Value</th>
                    <th className="num">Limit</th>
                  </tr>
                </thead>
                <tbody>
                  {metricRows(gate.metrics).map((row, i) => (
                    <tr key={i} className={row.judged && !row.passed ? "failed" : undefined}>
                      <td>{row.label}</td>
                      <td className="num">
                        {row.value} {row.judged ? (row.passed ? "✓" : "✕") : ""}
                      </td>
                      <td className="num">{row.limit}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/** A unit's attempts, the first first: why each ran, what the gates said, what it changed. */
export function AttemptTable<T extends AttemptSummary>({
  attempts,
  extra = [],
}: {
  attempts: T[];
  extra?: { label: string; render: (attempt: T) => ReactNode }[];
}) {
  if (attempts.length === 0) return null;
  return (
    <div className="viz-table-wrap">
      <table className="viz-table">
        <thead>
          <tr>
            <th className="num">#</th>
            <th>Stage</th>
            <th>Why</th>
            {extra.map((column) => (
              <th key={column.label}>{column.label}</th>
            ))}
            <th>Verdicts</th>
            <th>Changed</th>
          </tr>
        </thead>
        <tbody>
          {attempts.map((attempt) => (
            <tr key={`${attempt.stage}-${attempt.attempt}`}>
              <td className="num">{attempt.attempt}</td>
              <td>{attempt.stage.replaceAll("_", " ")}</td>
              <td>{triggerLabel(attempt.triggered_by)}</td>
              {extra.map((column) => (
                <td key={column.label}>{column.render(attempt)}</td>
              ))}
              <td>
                {attempt.status === "failed"
                  ? "failed"
                  : Object.entries(attempt.verdicts)
                      .map(([gate, verdict]) => `${gate} ${verdict}`)
                      .join(", ") || "—"}
              </td>
              <td>{changedText(attempt.parameters)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** What an attempt set, short: the long lists (every layer's bounds) counted, not spelt out. */
function changedText(parameters: Record<string, unknown>): string {
  const said = changes(parameters);
  if (said.length > 6) return `${said.slice(0, 5).join(", ")}, … (${said.length})`;
  return said.join(", ") || "—";
}

/** A card's details: each gate's metrics, then the attempts. */
export function Details({ gates, attempts, children }: {
  gates: GateView[];
  attempts: AttemptSummary[];
  children?: ReactNode;
}) {
  if (gates.length === 0 && attempts.length === 0 && !children) return null;
  return (
    <Fold title="Measures and attempts">
      <GateTables gates={gates} />
      {attempts.length > 0 && (
        <div className="viz-section">
          <h3 className="viz-h3">Attempts</h3>
          <AttemptTable attempts={attempts} />
        </div>
      )}
      {children}
    </Fold>
  );
}

/** A section's lateral smoothing, off or on, named above it as a card's other switches. */
export function SmoothingSwitch({ on, onChange }: { on: boolean; onChange: (on: boolean) => void }) {
  return (
    <div className="viz-field">
      Lateral smoothing
      <Segmented
        size="sm"
        label="Lateral smoothing"
        value={on ? "on" : "off"}
        onChange={(one) => onChange(one === "on")}
        options={[
          { value: "off", label: "Off" },
          { value: "on", label: "On" },
        ]}
      />
    </div>
  );
}

/** A plot's head, in its PlotBox: its title, and on the right its switches (`children`) then
 * the box's tools. */
export function PlotHead({ title, children }: { title?: ReactNode; children?: ReactNode }) {
  return (
    <div className="viz-row viz-plot-head boxed">
      {title !== undefined && <p className="viz-plot-title">{title}</p>}
      <span className="viz-plot-tools">
        {children}
        <BoxTools />
      </span>
    </div>
  );
}

export function Skeleton({ height, style }: { height: number; style?: React.CSSProperties }) {
  return <div className="viz-skeleton" style={{ height, ...style }} />;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="viz-empty">{children}</div>;
}

export function ErrorBox({ message }: { message: string }) {
  return <Callout tone="error">{message}</Callout>;
}
