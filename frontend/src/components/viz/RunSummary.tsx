import { capitalized, MAKER_LABELS, ORIGIN_LABELS, runDate } from "./format";
import { FolderIcon } from "../icons";
import { RunMaker } from "./maker";
import type { Maker, RunCard, Setting, Step, TriedTable } from "./types";
import { Fold, OriginTag, SettingsList } from "./ui";

// A run at a glance, alike whoever made it: who made it and when, its windows' length and step
// and the shots they stack, the band imaged; folded under them how each setting was chosen
// (every setting with why, and what was tried), and the run's history step by step.

const TILES = ["length", "step", "windows", "reach", "band"];

// Who made a run, on hover.
const MAKER_MEANINGS: Record<Maker, string> = {
  manual: "Made in PAC's pages, by hand",
  workflow: "Made by the assistant in the workflow mode\nPACo's rules chose the settings; its checks judged each stage",
  autonomous: "Made by the assistant in the autonomous mode\nThe agent chose the settings by eye; its reasoning under History",
};

function Tile({ setting }: { setting: Setting }) {
  return (
    <div className="viz-tile" data-tip={`${setting.label}\n${capitalized(setting.why)}`}>
      <div className="viz-tile-label">{setting.label}</div>
      <div className="viz-tile-value">{setting.value}</div>
      {setting.detail && <div className="viz-tile-sub">{setting.detail}</div>}
      <OriginTag origin={setting.origin} />
    </div>
  );
}

/** What was tried for one choice, by whom: each row what, how many trial windows passed with
 * it, its measures, the one kept. */
function Tried({ table }: { table: TriedTable }) {
  return (
    <div className="viz-tried">
      <div className="viz-tried-title">
        <span>{table.title}</span>
        <OriginTag origin={table.by} />
      </div>
      {table.about && <div className="viz-tried-about">{table.about}</div>}
      <div className="viz-table-wrap">
        <table className="viz-table">
          <thead>
            <tr>
              <th>Tried</th>
              <th className="num">Passed</th>
              <th>Measures</th>
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, index) => (
              <tr key={index} className={row.kept ? "kept" : undefined} data-tip={row.note || undefined}>
                <td>
                  {row.label}
                  {row.kept && <span className="viz-kept">kept</span>}
                </td>
                <td className="num">{row.passed || "—"}</td>
                <td>{row.measures || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// What each kind of step is: in the margin when it has no id (a stage's), and on hover.
const STEP_KINDS: Record<string, string> = {
  objective: "Objective",
  settings: "Your settings",
  search: "Search",
  experiment: "Experiment",
  commit: "Commit",
  freeze: "Freeze",
  report: "Report",
  reopen: "Reopened",
  hypothesis: "Hypothesis",
  conclusion: "Conclusion",
  decision: "Decision",
  given: "Given",
  records: "Records",
  length: "Length",
  mute: "Mute",
  segments: "Segments",
  images: "Images",
  line: "Line",
  picks: "Picks",
  inversion: "Inversion",
  petro: "Soils",
};
// The agent's own reasoning, apart from what code wrote.
const REASONING = new Set(["hypothesis", "conclusion", "decision"]);

/** The run's history: a step a line, its id or its kind in the margin in the colour of who made
 * it, the entries it rests on after it. */
function History({ steps }: { steps: Step[] }) {
  return (
    <ol className="viz-history">
      {steps.map((step, index) => {
        const kind = STEP_KINDS[step.kind] ?? capitalized(step.kind);
        const when = step.at ? `\n${runDate(step.at)}` : "";
        return (
          <li key={step.id || index} className={REASONING.has(step.kind) ? "reasoning" : undefined}>
            <span className={`viz-history-id ${step.by}`} data-tip={`${kind} · ${ORIGIN_LABELS[step.by]}${when}`}>
              {step.id || kind}
            </span>
            <span data-tip={step.detail || undefined}>
              {step.text}
              {step.of.length > 0 && <span className="viz-history-of"> (from {step.of.join(", ")})</span>}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

export function RunSummary({ card }: { card: RunCard }) {
  const tiles = TILES.map((key) => card.settings.find((one) => one.key === key)).filter(
    (one): one is Setting => one !== undefined,
  );
  // A backend before these lists sends none of them.
  const tried = card.tried ?? [];
  const history = card.history ?? [];
  return (
    <RunMaker.Provider value={card.made_by ?? null}>
      <section className="viz-card">
        <div className="viz-run-title">
          <span className="card-icon">
            <FolderIcon size={17} />
          </span>
          <strong>{card.profile}</strong>
          {card.mode && (
            <span className="viz-chip" data-tip={"Processing mode"}>
              {card.mode}
            </span>
          )}
          {card.made_by && (
            <span className={`viz-chip ${card.made_by}`} data-tip={MAKER_MEANINGS[card.made_by]}>
              {MAKER_LABELS[card.made_by]}
            </span>
          )}
          {card.started_at && <span className="viz-chip">{runDate(card.started_at)}</span>}
          {card.run_id === null && <span className="viz-chip">older layout: no run.json</span>}
        </div>
        {tiles.length > 0 && <div className="viz-tiles">{tiles.map((one) => <Tile key={one.key} setting={one} />)}</div>}
        {card.settings.length > 0 && (
          <Fold title="How it was chosen">
            <SettingsList settings={card.settings} />
            {tried.map((table, index) => (
              <Tried key={index} table={table} />
            ))}
          </Fold>
        )}
        {history.length > 0 && (
          <Fold title={`History: ${history.length} steps`}>
            <History steps={history} />
          </Fold>
        )}
      </section>
    </RunMaker.Provider>
  );
}
