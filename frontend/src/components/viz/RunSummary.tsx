import { capitalized, num, range, runDate, triggerLabel } from "./format";
import { FolderIcon } from "../icons";
import type { LengthTrial, LineTried, MuteTried, RunCard, SegmentsTried, Setting } from "./types";
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
            <th className="num">Passed</th>
            <th className="num" data-tip={"Picks ±\nThe passed curves' median velocity uncertainty"}>Picks ±</th>
            <th className="num">Wavelengths</th>
            <th className="num">Windows</th>
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

const MUTE_NAMES: Record<string, string> = {
  none: "none",
  standard: "standard",
  wider: "wider cone",
  cone: "cone",
  tight: "tight cone",
};

function Mutes({ mutes }: { mutes: MuteTried[] }) {
  return (
    <div className="viz-table-wrap viz-trials">
      <table className="viz-table">
        <thead>
          <tr>
            <th data-tip={"Mute tried\nEach on the same trial windows, at the line's length"}>Mute tried</th>
            <th className="num">Passed</th>
            <th className="num">Wavelengths</th>
          </tr>
        </thead>
        <tbody>
          {mutes.map((mute) => {
            const muting = mute.muting as { vmin?: number; vmax?: number } | null;
            const bounds = muting?.vmin != null && muting.vmax != null ? `, ${num(muting.vmin)}–${num(muting.vmax)} m/s` : "";
            return (
              <tr key={mute.candidate}>
                <td>
                  {MUTE_NAMES[mute.candidate] ?? mute.candidate}
                  {bounds}
                  {mute.kept ? " · kept" : ""}
                </td>
                <td className="num">
                  {mute.passed} of {mute.verdicts.length}
                </td>
                <td className="num">{range(mute.wavelengths_m, "m")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Segments({ segments }: { segments: SegmentsTried[] }) {
  return (
    <div className="viz-table-wrap viz-trials">
      <table className="viz-table">
        <thead>
          <tr>
            <th>Segments tried</th>
            <th>Selection</th>
            <th className="num" data-tip={"Span\nThe picks' longest wavelength over their shortest"}>Span</th>
            <th className="num" data-tip={"Coherence\nThe images' along the picks"}>Coherence</th>
          </tr>
        </thead>
        <tbody>
          {segments.map((one, index) => (
            <tr key={index}>
              <td>
                {num(one.segment_s)} s{one.own ? " · the line's" : ""}
                {one.kept ? " · kept" : ""}
              </td>
              <td>{one.threshold == null ? "none" : `FK ${num(one.threshold)}`}</td>
              <td className="num">{num(one.wavelength_span)}</td>
              <td className="num">{num(one.coherence)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LineTries({ tries }: { tries: LineTried[] }) {
  return (
    <div className="viz-table-wrap viz-trials">
      <table className="viz-table">
        <thead>
          <tr>
            <th data-tip={"Change tried\nOn the whole line, asked by the checks of windows whose curve failed"}>
              Change tried
            </th>
            <th className="num">Asked by</th>
            <th className="num" data-tip={"Passed\nTrial windows passing G3: the line's settings, then with the change"}>
              Passed
            </th>
            <th>Line</th>
          </tr>
        </thead>
        <tbody>
          {tries.map((one, index) => (
            <tr key={index} data-tip={one.note}>
              <td>{one.change}</td>
              <td className="num" data-tip={triggerLabel(one.flag)}>
                {one.asked_by} windows
              </td>
              <td className="num">
                {one.before} → {one.after} of {one.xmids.length}
              </td>
              <td>{one.kept ? "kept" : "not kept"}</td>
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
  // A backend before these lists sends none of them.
  const mutes = card.mutes ?? [];
  const segments = card.segments ?? [];
  const lineTries = card.line_tries ?? [];
  return (
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
        {card.by && (
          <span
            className="viz-chip blue"
            data-tip={card.by === "assistant" ? "By the assistant\nChecked stage by stage" : "By hand"}
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
          {mutes.length > 0 && <Mutes mutes={mutes} />}
          {segments.length > 0 && <Segments segments={segments} />}
          {lineTries.length > 0 && <LineTries tries={lineTries} />}
        </Fold>
      )}
    </section>
  );
}
