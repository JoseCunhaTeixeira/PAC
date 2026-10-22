import type { ReactNode } from "react";
import { BoxTools, Callout, Segmented } from "../kit";
import {
  capitalized,
  checkName,
  measured,
  metricLabel,
  metricLimit,
  metricTip,
  metricValue,
  MEASURED_LABELS,
  objectTitle,
  ORIGIN_LABELS,
  runDate,
  STATUS_LABELS,
} from "./format";
import type {
  GateView,
  Mark,
  Metric,
  Origin,
  PartState,
  Sentence,
  Setting,
  Status,
  Step,
  Verdict,
} from "./types";
import type { SavedFigure } from "./runFigures";
import "./viz.css";

// The pieces every panel of Visualization says things with: a state's badge, a card's
// sentences, the settings a stage ran with, and folded under a card, each check's measures and
// what was done, a step each.

const MARK_ICONS: Record<Mark, string> = { pass: "✓", warn: "!", flag: "!", fail: "✕", info: "i" };
const VERDICT_STATUS: Record<Verdict, Status> = { pass: "pass", retry: "warn", reject: "fail" };
// A check's verdict, as its badge says it.
const VERDICT_WORDS: Record<Verdict, string> = { pass: "passed", retry: "flagged", reject: "rejected" };
// A setting's parameters apart, as the run card's tiles say them: a line each.
const SEP = " · ";

export function StatusBadge({ status, label, meaning }: { status: Status; label?: string; meaning?: string }) {
  return (
    <span className={`viz-badge ${status}`} data-tip={meaning || undefined}>
      {label ?? STATUS_LABELS[status]}
    </span>
  );
}

// A part's state in words, and what each part checks (its badge's hover).
const PART_WORDS: Record<PartState, string> = {
  pass: "passed",
  note: "beyond limits",
  warn: "flagged",
  fail: "rejected",
  hand: "picked by the user",
  none: "not checked",
};
const NONE_WORDS: Record<string, string> = { image: "not made", curve: "not picked" };
const PART_GATES: Record<string, string> = {
  image: "The dispersion image check",
  curve: "The curve check, and the check against the neighbouring curves",
};

/** One of a unit's checks said apart (a window's image, its curve): what it checks, its state,
 * in its colour (indigo by hand); made by the user and judged by no check, against the agent's
 * limits. */
export function PartBadge({ part }: { part: { label: string; state: PartState; judged?: boolean } }) {
  const word =
    (part.judged === false && part.state !== "hand" && MEASURED_LABELS[part.state]) || PART_WORDS[part.state];
  const said = part.state === "none" ? `${part.label} ${NONE_WORDS[part.label] ?? "not made"}` : `${part.label} ${word}`;
  const why = part.state === "hand" ? "" : (PART_GATES[part.label] ?? "");
  return (
    <span className={`viz-badge ${part.state}`} data-tip={why || undefined}>
      {said}
    </span>
  );
}

/** A check's verdict, in a word; for a run made by hand, that it was measured, not judged. */
export function GateBadge({ gate }: { gate: GateView }) {
  if (gate.by_hand) {
    const passed = gate.verdict === "pass";
    return (
      <span className={`viz-badge ${passed ? "hand" : "none"}`}>{passed ? "picked by the user" : "not checked"}</span>
    );
  }
  if (gate.verdict === null) {
    return (
      <span className="viz-badge none" data-tip="Against the agent's limits, not judged: made by the user">
        measured
      </span>
    );
  }
  return <span className={`viz-badge ${VERDICT_STATUS[gate.verdict]}`}>{VERDICT_WORDS[gate.verdict]}</span>;
}

export function Sentences({ sentences }: { sentences: Sentence[] }) {
  return (
    <ul className="viz-sentences">
      {sentences.map((sentence, i) => (
        <li key={i} data-tip={sentence.detail || undefined}>
          <span className={`viz-mark ${sentence.mark}`}>{MARK_ICONS[sentence.mark]}</span>
          <span>{sentence.text}</span>
        </li>
      ))}
    </ul>
  );
}

/** Who set a setting, a tag. */
export function OriginTag({ origin }: { origin: Origin }) {
  return <span className={`viz-origin ${origin}`}>{ORIGIN_LABELS[origin]}</span>;
}

/** Settings as a definition list: each its value and who set it, its other parameters a line
 * each, why when its unit's own, then how the line's other units differ in it. */
export function SettingsList({ settings }: { settings: Setting[] }) {
  return (
    <dl className="viz-settings">
      {settings.map((setting) => (
        <div key={setting.key} style={{ display: "contents" }}>
          <dt>{setting.label}</dt>
          <dd>
            <b>{setting.value}</b>
            <OriginTag origin={setting.origin} />
            {setting.detail
              .split(SEP)
              .filter(Boolean)
              .map((line) => (
                <span key={line} className="viz-why">
                  {line}
                </span>
              ))}
            {setting.why && <span className="viz-why">{capitalized(setting.why)}.</span>}
            {setting.spread && <span className="viz-why viz-muted">{capitalized(setting.spread)}.</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** What was done: a step after another along a timeline, each its number on the line in the
 * colour of who did it (its date on hover), its title and who did it, then its facts, a line
 * each. */
export function Timeline({ steps }: { steps: Step[] }) {
  return (
    <ol className="viz-timeline">
      {steps.map((step, index) => (
        <li key={index} className="viz-step" data-tip={step.detail || undefined}>
          <span className={`viz-step-node ${step.by}`} data-tip={step.at ? runDate(step.at) : undefined}>
            {index + 1}
          </span>
          <div className="viz-step-title">{step.title}</div>
          <OriginTag origin={step.by} />
          {step.lines.length > 0 && (
            <ul className="viz-step-lines">
              {step.lines.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          )}
        </li>
      ))}
    </ol>
  );
}

/** The figures a run or a window saved, each linking to itself, in a fold; nothing without
 * any. */
export function SavedFigures({ figures }: { figures: SavedFigure[] }) {
  if (figures.length === 0) return null;
  return (
    <Fold title="Saved figures">
      <div className="viz-figures">
        {figures.map(({ name, url }) => (
          <a key={name} href={url} target="_blank" rel="noreferrer" title={name}>
            <img src={url} alt={name} loading="lazy" />
          </a>
        ))}
      </div>
    </Fold>
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
  tip: string;
  value: string;
  limit: string;
  /** Against its limit: within it; beyond it and a warning alone, the reason the check flagged
   * the unit, or the reason it rejected it; null when reported only, or not measured. */
  state: "pass" | "warn" | "flag" | "fail" | null;
}

// A measure against its limit, its mark; one beyond it that did not reject its unit, its colour
// alone.
const STATE_MARKS = { pass: "✓", warn: "", flag: "", fail: "✕" };
const STATE_ROWS = { pass: undefined, warn: "warned", flag: "flagged", fail: "failed" };

/** A check's measures as rows, a measure held between a floor and a ceiling (the acceptance
 * between 20 and 30 %) in one row, its band as its limit; one beyond its limit the reason the
 * check rejected the unit when `rejected` names it, flagged it when `flagged` does, else a
 * warning. */
function metricRows(gate: string, metrics: Metric[], rejected: string[], flagged: string[]): MetricRow[] {
  const state = (judged: boolean, passed: boolean, name: string): MetricRow["state"] =>
    !judged
      ? null
      : passed
        ? "pass"
        : rejected.includes(name)
          ? "fail"
          : flagged.includes(name)
            ? "flag"
            : "warn";
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
    const label = metricLabel(metric.name, gate);
    const tip = metricTip(metric.name, gate);
    if (j >= 0) {
      merged.add(j);
      const [low, high] = metric.bound === "min" ? [metric, metrics[j]] : [metrics[j], metric];
      rows.push({
        label,
        tip,
        value: metricValue(metric),
        limit: band(low, high),
        state: state(measured(metric), low.passed && high.passed, metric.name),
      });
      return;
    }
    rows.push({
      label,
      tip,
      value: metricValue(metric),
      limit: metricLimit(metric),
      state: state(metric.threshold != null && metric.bound != null && measured(metric), metric.passed, metric.name),
    });
  });
  return rows;
}

/** A floor and a ceiling as one band: "20 to 30 %". */
function band(low: Metric, high: Metric): string {
  const from = metricValue(low, low.threshold);
  const to = metricValue(high, high.threshold);
  const unit = / (\S+)$/.exec(to)?.[1];
  return `${unit && from.endsWith(` ${unit}`) ? from.slice(0, -unit.length - 1) : from} to ${to}`;
}

/** A check's measures by the thing they describe (a record's signal, its spectrum), in the order
 * measured. */
function byObject(metrics: Metric[]): [string, Metric[]][] {
  const objects = new Map<string, Metric[]>();
  for (const metric of metrics) {
    const of = metric.of ?? "";
    objects.set(of, [...(objects.get(of) ?? []), metric]);
  }
  return [...objects];
}

/** Measures against their limits: each its name, what it is on the name's hover, its value
 * and its limit. */
function MetricTable({
  gate,
  metrics,
  rejected,
  flagged,
  title,
}: {
  gate: string;
  metrics: Metric[];
  rejected: string[];
  flagged: string[];
  title?: string;
}) {
  const rows = metricRows(gate, metrics, rejected, flagged);
  // The same columns in every table, their values aligned from one to the next.
  return (
    <div className="viz-table-wrap">
      <table className="viz-table viz-measures">
        <colgroup>
          <col />
          <col className="viz-col-num" />
          <col className="viz-col-limit" />
        </colgroup>
        <thead>
          <tr>
            <th>{title ?? "Measure"}</th>
            <th className="num">Value</th>
            <th className="num">Limit</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={row.state ? STATE_ROWS[row.state] : undefined}>
              <td>
                <span className={row.tip ? "viz-term" : undefined} data-tip={row.tip || undefined}>
                  {row.label}
                </span>
              </td>
              <td className="num">
                {row.value} {row.state ? STATE_MARKS[row.state] : ""}
              </td>
              <td className="num">{row.limit}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A card's measures in one menu, like its settings: a submenu for each check, its verdict
 * beside it, its measures in a table for each thing it measures (a record's signal, its
 * spectrum), titled when it measures several. */
export function GateTables({ gates }: { gates: GateView[] }) {
  // A curve picked by hand is measured, not judged (its badge says so); nothing of it is compared
  // along the line.
  const measured = gates.filter((gate) => !gate.by_hand || gate.metrics.length > 0);
  if (measured.length === 0) return null;
  return (
    <Fold title="Measures and limits">
      <div className="viz-measure-menus">
        {measured.map((gate) => {
          const objects = byObject(gate.metrics);
          const line = objects.length > 0 && objects.every(([of]) => of === "line");
          return (
            <Fold
              key={gate.gate}
              title={
                <span className="viz-measure-head">
                  {checkName(gate.gate, line)}
                  <GateBadge gate={gate} />
                </span>
              }
            >
              {objects.length === 0 ? (
                <p className="viz-muted viz-small">Nothing measured.</p>
              ) : (
                objects.map(([of, metrics]) => (
                  <MetricTable
                    key={of}
                    gate={gate.gate}
                    metrics={metrics}
                    rejected={gate.rejected_by ?? []}
                    flagged={gate.flagged_by ?? []}
                    title={objects.length > 1 ? objectTitle(gate.gate, of) : undefined}
                  />
                ))
              )}
            </Fold>
          );
        })}
      </div>
    </Fold>
  );
}

/** A card's details: each check's measures, then what was done to its unit, a step each, when
 * a step ran again, folded. */
export function Details({ gates, steps = [], children }: {
  gates: GateView[];
  steps?: Step[];
  children?: ReactNode;
}) {
  if (gates.length === 0 && steps.length === 0 && !children) return null;
  return (
    <>
      <GateTables gates={gates} />
      {(steps.length > 0 || children) && (
        <Fold title="What was done">
          {steps.length > 0 && <Timeline steps={steps} />}
          {children}
        </Fold>
      )}
    </>
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
