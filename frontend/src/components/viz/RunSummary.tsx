import { useState, type CSSProperties } from "react";
import { MAKER_LABELS, runDate } from "./format";
import { FolderIcon } from "../icons";
import { DispersionThumbs } from "./DispersionThumbs";
import { RunMaker } from "./maker";
import { OptionImages } from "./OptionImages";
import type { Maker, Reason, RunCard, Setting, TriedTable } from "./types";
import { Fold, OriginTag, Timeline } from "./ui";

// A run at a glance, for someone who does not know how it was made: its settings in a strip, in
// the order its records went through them; folded under them what was done, a step after another
// along a timeline, and why these settings: what was tried, every setting of each option and its
// dispersion images, the assistant's own words.

// Who made a run, on hover.
const MAKER_MEANINGS: Record<Maker, string> = {
  manual: "Made by the user in PAC",
  workflow: "Made by the workflow agent, step by step with fixed rules",
  autonomous: "Made by the autonomous agent, comparing the dispersion images with its vision",
};

// The agent, by the run's mode.
const AGENTS = { agent: "autonomous agent", workflow: "workflow agent" };

// A setting's parameters, apart: a line each.
const SEP = " · ";

// What a comparison's settings mean, on their name's hover, where their name alone does not say.
const ROW_TIPS: Record<string, string> = {
  "Shots used": "Their distance from each window's middle",
  "Picks' span": "The picked curve's longest wavelength over its shortest\nMedian over the sample windows",
  Coherence: "The dispersion image's coherence along the picked curve\nMedian over the sample windows",
  "Asked by": "The windows whose curve failed and asked for this change",
};

/** A setting's parameters, a line each, the first as its value. */
function Lines({ text }: { text: string }) {
  const [value, ...more] = text.split(SEP);
  return (
    <>
      <div>{value}</div>
      {more.map((line) => (
        <div key={line} className="viz-sub">
          {line}
        </div>
      ))}
    </>
  );
}

function Tile({ setting }: { setting: Setting }) {
  return (
    <div className="viz-tile">
      <div className="viz-tile-label">{setting.label}</div>
      <div className="viz-tile-value">{setting.value}</div>
      {setting.detail
        .split(SEP)
        .filter(Boolean)
        .map((line) => (
          <div key={line} className="viz-tile-sub">
            {line}
          </div>
        ))}
      {setting.origin && <OriginTag origin={setting.origin} />}
    </div>
  );
}

/** A comparison: what was done, then its options side by side, a column each, a setting a row,
 * every parameter of each; how many sample windows gave a curve the checks passed and half the
 * curves' longest wavelength, last. An option clicked, its dispersion images under the table. */
function Comparison({ folder, table }: { folder: string; table: TriedTable }) {
  const clear = table.rows.some((row) => row.clear);
  const depth = table.rows.some((row) => row.depth);
  const imaged = table.rows.some((row) => row.trial);
  const [shown, setShown] = useState(() => Math.max(0, table.rows.findIndex((row) => row.kept)));
  const [hovered, setHovered] = useState<number | null>(null);
  // An option's cells: the chosen one's tinted, the one shown marked; a click on any of them
  // shows its dispersion images.
  const cell = (index: number) => ({
    className:
      [
        table.rows[index].kept && "kept",
        imaged && index === shown && "shown",
        imaged && "pickable",
        imaged && index === hovered && "hovered",
      ]
        .filter(Boolean)
        .join(" ") || undefined,
    onClick: imaged ? () => setShown(index) : undefined,
    onMouseEnter: imaged ? () => setHovered(index) : undefined,
    onMouseLeave: imaged ? () => setHovered(null) : undefined,
  });
  return (
    <div className="viz-tried">
      <div className="viz-tried-title">{table.title}</div>
      {table.intro.length > 0 && (
        <ul className="viz-tried-intro">
          {table.intro.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      )}
      <div className="viz-table-wrap">
        <table
          className="viz-table viz-options"
          style={{ "--options": table.rows.length } as CSSProperties}
        >
          <thead>
            <tr>
              <th />
              {table.rows.map((row, index) => (
                <th key={index} {...cell(index)}>
                  {imaged ? (
                    <button type="button" className="viz-option">
                      {row.name}
                    </button>
                  ) : (
                    row.name
                  )}
                  {row.kept && <span className="viz-chosen">Chosen</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.columns.map((column, at) => (
              <tr key={column}>
                <th scope="row" data-tip={ROW_TIPS[column]}>
                  {column}
                </th>
                {table.rows.map((row, index) => (
                  <td key={index} {...cell(index)}>
                    <Lines text={row.values[at] ?? "-"} />
                  </td>
                ))}
              </tr>
            ))}
            {clear && (
              <tr className="viz-measure">
                <th
                  scope="row"
                  data-tip={"Sample windows whose curve passed the checks\nA change: under it, the windows asking it"}
                >
                  Curves passed
                </th>
                {table.rows.map((row, index) => (
                  <td key={index} {...cell(index)}>
                    <Lines text={row.clear || "—"} />
                  </td>
                ))}
              </tr>
            )}
            {depth && (
              <tr className="viz-measure">
                <th
                  scope="row"
                  className="viz-symbol"
                  data-tip={"Half the longest wavelength of the curves\nMedian over the sample windows, not a depth measured"}
                >
                  λmax / 2
                </th>
                {table.rows.map((row, index) => (
                  <td key={index} {...cell(index)}>
                    {row.depth || "—"}
                  </td>
                ))}
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {imaged && table.rows[shown] && (
        <div className="viz-option-images">
          <div className="viz-option-images-title">
            Dispersion images of {table.rows[shown].name}
            <span className="viz-muted"> · click an option to see its own</span>
          </div>
          <OptionImages key={shown} folder={folder} option={table.rows[shown]} />
        </div>
      )}
    </div>
  );
}

/** Reasons a setting each, its name then its reasons, a line each. */
function Reasons({ title, reasons }: { title: string; reasons: Reason[] }) {
  return (
    <div className="viz-reasons">
      <div className="viz-reasons-title">{title}</div>
      <dl>
        {reasons.map((one) => (
          <div key={one.setting} style={{ display: "contents" }}>
            <dt>{one.setting}</dt>
            <dd>
              {one.text.split(SEP).map((line) => (
                <div key={line}>{line}</div>
              ))}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** Why these settings: the comparisons made to choose them, with their options' dispersion
 * images (or the run's own along its line), the assistant's own words, then what you set. */
function Why({ card, tables, reasons }: { card: RunCard; tables: TriedTable[]; reasons: Reason[] }) {
  const own = reasons.filter((one) => one.by !== "you");
  const yours = reasons.filter((one) => one.by === "you");
  // The run's own dispersion images along its line, unless the option it was made with shows
  // them (the autonomous search's: its settings are the run's); none for the workflow's, whose
  // comparisons are their tables.
  const shown = tables.some((table) => table.by === "agent" && table.rows.some((row) => row.kept && row.images));
  const imaged =
    card.made_by !== "workflow" && !shown && card.stages.some((stage) => stage.key === "dispersion" && stage.done > 0);
  return (
    <>
      {tables.map((table, index) => (
        <Comparison key={index} folder={card.folder} table={table} />
      ))}
      {imaged && (
        <div className="viz-tried">
          <div className="viz-tried-title">Dispersion images with the settings kept</div>
          <DispersionThumbs folder={card.folder} receivers={card.receivers} windows={card.windows} />
        </div>
      )}
      {own.length > 0 && (
        <Reasons
          title={`In the ${card.made_by === "workflow" ? AGENTS.workflow : AGENTS.agent}'s own words`}
          reasons={own}
        />
      )}
      {yours.length > 0 && <Reasons title="Set by the user" reasons={yours} />}
    </>
  );
}

export function RunSummary({ card }: { card: RunCard }) {
  // A backend before these lists sends none of them.
  const tried = card.tried ?? [];
  const reasons = card.reasons ?? [];
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
        {card.settings.length > 0 && (
          <div className="viz-tiles">
            {card.settings.map((one) => (
              <Tile key={one.key} setting={one} />
            ))}
          </div>
        )}
        {history.length > 0 && (
          <Fold title="What was done">
            <Timeline steps={history} />
          </Fold>
        )}
        {(tried.length > 0 || reasons.length > 0) && (
          <Fold title="Why these settings">
            <Why card={card} tables={tried} reasons={reasons} />
          </Fold>
        )}
      </section>
    </RunMaker.Provider>
  );
}
