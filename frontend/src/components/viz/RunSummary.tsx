import { capitalized, runDate } from "./format";
import type { LengthTrial, RunCard, Setting } from "./types";
import { Fold, OriginTag, SettingsList } from "./ui";

// A run at a glance: who made it and when, its windows' length and step and the shots they
// stack, the band imaged, and folded under them every setting with why.

const TILES = ["length", "step", "windows", "reach", "band"];

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

function Trials({ trials, kept }: { trials: LengthTrial[]; kept: number | null }) {
  return (
    <div className="viz-table-wrap viz-trials">
      <table className="viz-table">
        <thead>
          <tr>
            <th>Length tried</th>
            <th>Passed</th>
            <th data-tip={"Picks ±\nThe passed curves' median velocity uncertainty"}>Picks ±</th>
            <th>Wavelengths</th>
            <th>Windows</th>
          </tr>
        </thead>
        <tbody>
          {trials.map((trial) => (
            <tr key={trial.length}>
              <td>
                {trial.length} receivers ({trial.metres} m)
                {trial.length === kept ? " · kept" : trial.compared ? " · to compare" : ""}
              </td>
              <td className="num">
                {trial.passed} of {trial.xmids.length}
              </td>
              <td className="num">{trial.uncertainty != null ? `${Math.round(trial.uncertainty * 100)} %` : "—"}</td>
              <td className="num">
                {trial.wavelengths_m ? `${trial.wavelengths_m[0]}–${trial.wavelengths_m[1]} m` : "—"}
              </td>
              <td className="num">{trial.windows}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RunSummary({ card }: { card: RunCard }) {
  const tiles = TILES.map((key) => card.settings.find((one) => one.key === key)).filter(
    (one): one is Setting => one !== undefined,
  );
  const length = card.settings.find((one) => one.key === "length");
  const kept = length ? Number(length.detail.split(" ")[0]) : null;
  return (
    <section className="viz-card">
      <div className="viz-run-title">
        <strong>{card.profile}</strong>
        {card.mode && (
          <span className="viz-chip" data-tip={"Processing mode"}>
            {card.mode}
          </span>
        )}
        {card.by && (
          <span
            className="viz-chip blue"
            data-tip={card.by === "assistant" ? "Run by the assistant, checked stage by stage" : "Run by hand, from the computing pages"}
          >
            {card.by === "assistant" ? "by the assistant" : "by hand"}
          </span>
        )}
        {card.started_at && <span className="viz-chip">{runDate(card.started_at)}</span>}
        {card.run_id === null && <span className="viz-chip">older layout: no run.json</span>}
      </div>
      {tiles.length > 0 && <div className="viz-tiles">{tiles.map((one) => <Tile key={one.key} setting={one} />)}</div>}
      {card.settings.length > 0 && (
        <Fold title="All settings, and why">
          <SettingsList settings={card.settings} />
          {card.trials.length > 0 && <Trials trials={card.trials} kept={kept} />}
        </Fold>
      )}
    </section>
  );
}
