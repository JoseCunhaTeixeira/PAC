import { useState } from "react";
import { API } from "../../api";
import { canvasPalette, useTheme } from "../../theme";
import { afmhotR, purples, terrain } from "../colormaps";
import { LayersIcon, StrataIcon } from "../icons";
import { BoxTools, Card, PlotBox, Segmented } from "../kit";
import { ModeHead } from "../PseudoSectionCanvas";
import { PseudoSectionComparisonCanvas, type PseudoSectionComparisonData } from "../PseudoSectionComparisonCanvas";
import { useZoomLink } from "../useZoom";
import {
  VelocitySectionCanvas,
  type InformedOverlay,
  type InformedWindow,
} from "../VelocitySectionCanvas";
import { ChainLegend, ChainTracesCanvas, MarginalsGrid } from "./ChainPlots";
import { DEPTH_INFORMED_TIP, MODEL_LABELS, num, parameterLabel, xmidOf } from "./format";
import { StageHead, UnitCard } from "./panel";
import { CurveFitPlot, VsProfilePlot, type CurveAxis } from "./plots";
import type { Chains, InversionCard, ModelName, Overview } from "./types";
import { AttemptTable, Empty, ErrorBox, Fold, GateTables, PlotHead, Skeleton, SmoothingSwitch } from "./ui";
import { nearestCell } from "./cells";
import { useJson } from "./useJson";

// The seismic inversion stage: the display settings, the selected window's card (how deep its
// data inform its model, its fit, its chains, what it ran with and each attempt's change), its
// model against depth beside its curve against the model's, and along the line, the Vs section
// and its spread, and the picked curves against the models'.

const at = (folder: string) => encodeURIComponent(folder);

// The interfaces' shares, mostly under a third where the models agree only roughly: coloured by
// their square root, so that those show too (the colour bar keeps the shares).
const interfaceColours = (t: number) => purples(Math.sqrt(t));

// The model shown, for now the only one (the user, 2026-09-29): at each depth, the kept models'
// median Vs. The inversion saves the others still (the assistant's checks read the layered
// median).
const MODEL: ModelName = "ensemble";

interface VelocitySection {
  positions: number[];
  elevations: number[];
  vs_grid: (number | null)[][];
  // The kept models' relative uncertainty of Vs, U = (P90 - P10) / (2 P50), %: the depth
  // informed read from it.
  vs_uncertainty_grid: (number | null)[][];
  // The share of the kept models with an interface, % (null: not known).
  interface_grid: (number | null)[][];
  floors?: number[]; // per column, the elevation its models end at
  // Each window's column, how deep its data inform it as its inversion measured.
  windows: InformedWindow[];
  // Per column: the elevation down to which the data inform it, smoothed as the section.
  informed_levels: (number | null)[];
}

/** The legend's swatch of the veil: the Vs colours seen through it. */
function veiled(veil: string): string {
  const stops = [0.15, 0.5, 0.85].map((t) => `rgb(${terrain(t).join(", ")})`);
  return `linear-gradient(${veil}, ${veil}), linear-gradient(90deg, ${stops.join(", ")})`;
}

function ModelTable({ card }: { card: InversionCard }) {
  // The half-space's own row, when the layers were given.
  const n = card.parameters?.layering === "free" ? 0 : (card.parameters?.vs_layers.length ?? 0);
  return (
    <div className="viz-section">
      <h3 className="viz-h3">Model, priors and sampling</h3>
      <p className="viz-small viz-muted" style={{ margin: "0 0 6px" }}>
        {card.parameters?.n_chains} chains, {card.samples_per_chain.toLocaleString("en-US")} samples each kept;
        acceptance by chain: {card.acceptance.map((rate) => `${num(rate)} %`).join(", ")}
        {card.tuning.length > 0 &&
          `; steps tuned in ${card.tuning.length} trial run${card.tuning.length > 1 ? "s" : ""} (${card.tuning
            .map(([factor, rate]) => `×${num(factor, 2)}: ${num(rate)} %`)
            .join(", ")})`}
        {card.exchanges != null && `; exchanges between tempered copies ${num(card.exchanges)} %`}
        .
      </p>
      <div className="viz-table-wrap">
        <table className="viz-table">
          <thead>
            <tr>
              <th>Parameter</th>
              <th className="num">Prior</th>
              <th className="num" data-tip="Median of all chains' models">Result</th>
              <th className="num" data-tip="Where 80 % of the models lie">10–90 %</th>
              <th className="num">R-hat</th>
              <th className="num">Effective samples</th>
              <th className="num">Lag-1 autocorr.</th>
              <th className="num">Step</th>
            </tr>
          </thead>
          <tbody>
            {card.convergence.map((row) => (
              <tr key={row.parameter}>
                <td>
                  {parameterLabel(row.parameter)}
                  {row.parameter === `vs${n}` ? " (half-space)" : ""}
                </td>
                <td className="num">
                  {row.fixed != null ? (
                    <span className="faint" data-tip={"Fixed\nNot sampled"}>
                      fixed
                    </span>
                  ) : row.prior ? (
                    `${num(row.prior[0])}–${num(row.prior[1])}`
                  ) : (
                    "—"
                  )}
                </td>
                <td className="num">
                  <b>{num(row.median)}</b>
                </td>
                <td className="num">{row.low !== null && row.high !== null ? `${num(row.low)}–${num(row.high)}` : "—"}</td>
                <td className="num">{num(row.rhat)}</td>
                <td className="num">{num(row.ess)}</td>
                <td className="num">{num(row.autocorrelation)}</td>
                <td className="num">
                  {num(row.step)}
                  {row.step != null && row.step_unit ? ` ${row.step_unit}` : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <MovesTable card={card} />
      <BoundsTable card={card} />
      <FitsTable card={card} />
    </div>
  );
}

// The moves of the chains when the data chose the layers, as the sampler names them.
const MOVES: Record<string, [string, string]> = {
  birth: ["Birth", "A layer added"],
  death: ["Death", "A layer removed"],
  interface: ["Interface", "An interface's own move"],
  relocate: ["Relocate", "An interface drawn anew between its neighbours"],
  vs: ["Vs", "A layer's own Vs"],
  noise: ["Noise factor", "The factor on the picks' uncertainties"],
  shift: ["Shift", "Every Vs together"],
  stretch: ["Stretch", "Every depth together"],
};

function MovesTable({ card }: { card: InversionCard }) {
  const moves = Object.entries(card.moves ?? {});
  if (!moves.length) return null;
  return (
    <div className="viz-table-wrap">
      <table className="viz-table">
        <thead>
          <tr>
            <th data-tip={"How the chains moved\nMedian of the chains"}>Move</th>
            <th className="num">Accepted</th>
            <th className="num" data-tip="Relative to the value moved">Step</th>
          </tr>
        </thead>
        <tbody>
          {moves.map(([move, rate]) => {
            const [label, tip] = MOVES[move] ?? [move, ""];
            const step = card.move_steps?.[move];
            return (
              <tr key={move}>
                <td data-tip={tip || undefined}>{label}</td>
                <td className="num">{num(rate)} %</td>
                <td className="num">{step != null ? `${num(step)} %` : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// The parameters the bounds are watched on when the data chose the layers.
const BOUND_LABELS: Record<string, string> = {
  top_vs: "Top Vs [m/s]",
  half_space_vs: "Half-space Vs [m/s]",
  deepest_interface: "Deepest interface [m]",
  layers: "Layers",
};

function BoundsTable({ card }: { card: InversionCard }) {
  const piled = card.at_bounds.filter((share) => share.share > 0);
  if (!piled.length) return null;
  return (
    <div className="viz-table-wrap">
      <table className="viz-table">
        <thead>
          <tr>
            <th data-tip={"Samples at a prior's bound\nThe data would go further"}>At a bound</th>
            <th className="num">Bound</th>
            <th className="num">Samples</th>
          </tr>
        </thead>
        <tbody>
          {piled.map((share) => (
            <tr key={`${share.parameter}-${share.bound}`}>
              <td>{BOUND_LABELS[share.parameter] ?? parameterLabel(share.parameter)}</td>
              <td className="num">
                {share.bound} {num(share.value)}
              </td>
              <td className="num">{num(100 * share.share)} %</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const BAND_LABELS = ["Short", "Middle", "Long"];

function FitsTable({ card }: { card: InversionCard }) {
  if (!card.fits.length) return null;
  const bands = card.fits[0].bands;
  return (
    <div className="viz-table-wrap">
      <table className="viz-table">
        <thead>
          <tr>
            <th data-tip={"Each model's curve against the picks\nIn uncertainties"}>Fit</th>
            <th className="num">Misfit</th>
            {bands.map((band, i) => (
              <th key={i} className="num" data-tip={`Wavelengths ${num(band.wavelength_m[0])}–${num(band.wavelength_m[1])} m`}>
                {bands.length === 3 ? BAND_LABELS[i] : `Band ${i + 1}`}
              </th>
            ))}
            <th className="num" data-tip="Picks the model has no fundamental mode at">Missing</th>
          </tr>
        </thead>
        <tbody>
          {card.fits.filter((fit) => fit.model === MODEL).map((fit) => (
            <tr key={fit.model}>
              <td>{MODEL_LABELS[fit.model] ?? fit.model}</td>
              <td className="num">{num(fit.misfit)}</td>
              {fit.bands.map((band, i) => (
                <td key={i} className="num">
                  {num(band.misfit)}
                </td>
              ))}
              <td className="num">{fit.n_missing}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChainsView({ folder, xmid }: { folder: string; xmid: number }) {
  const chains = useJson<Chains>(`${API}/quality/inversion/chains/${at(folder)}/${xmid}`);
  const [parameter, setParameter] = useState<string | null>(null);
  if (chains.error) return <Empty>{chains.error}</Empty>;
  if (!chains.data) return <Skeleton height={200} />;
  const traces = chains.data.traces.find((one) => one.parameter === parameter) ?? chains.data.traces[0];
  const marginal = chains.data.marginals.find((one) => one.parameter === traces?.parameter);
  return (
    <div>
      <PlotBox>
        <div className="viz-toolbar boxed" style={{ marginTop: 8 }}>
          <label className="viz-field">
            Parameter
            <select value={traces?.parameter ?? ""} onChange={(e) => setParameter(e.target.value)}>
              {chains.data.traces.map((one) => (
                <option key={one.parameter} value={one.parameter}>
                  {parameterLabel(one.parameter)}
                </option>
              ))}
            </select>
          </label>
          <span className="viz-small viz-muted">
            <ChainLegend n={traces?.chains.length ?? 0} />
          </span>
          <BoxTools />
        </div>
        {traces && <ChainTracesCanvas traces={traces} prior={marginal ? [marginal.low, marginal.high] : undefined} />}
      </PlotBox>
      <PlotBox>
        <div style={{ marginTop: 12 }}>
          <PlotHead title="Marginals within the priors (dashed: a flat posterior)" />
        </div>
        <MarginalsGrid marginals={chains.data.marginals} />
      </PlotBox>
    </div>
  );
}

export function InversionPanel({
  folder,
  selected,
  overview,
  overviewError,
  onSelect,
}: {
  folder: string;
  selected: string | null;
  overview: Overview | null;
  overviewError: string | null;
  onSelect: (key: string) => void;
}) {
  const model = MODEL;
  const [smoothing, setSmoothing] = useState(false);
  // The depth each window's data inform, over the sections.
  const [informed, setInformed] = useState(true);
  const colours = canvasPalette(useTheme());
  const [vsMin, setVsMin] = useState("");
  const [vsMax, setVsMax] = useState("");
  const [curveAxis, setCurveAxis] = useState<CurveAxis>("frequency");
  const [chainsOpen, setChainsOpen] = useState(false);
  // The picked and modelled pseudo-sections' vertical axis.
  const [comparisonAxis, setComparisonAxis] = useState<"frequency" | "wavelength">("frequency");
  const zoomLink = useZoomLink(folder);
  const xmid = selected ? xmidOf(selected) : null;
  const card = useJson<InversionCard>(
    xmid !== null ? `${API}/quality/inversion/card/${at(folder)}/${xmid}?model=${model}` : null,
  );
  const section = useJson<VelocitySection>(
    `${API}/inversion/velocity_section/${at(folder)}?model=${model}&lateral_smoothing=${smoothing}`,
  );
  const overlay: InformedOverlay | undefined =
    informed && section.data
      ? { levels: section.data.informed_levels, windows: section.data.windows, floors: section.data.floors }
      : undefined;
  const labels = useJson<Record<string, number>>(`${API}/dispersion_image_labels/${at(folder)}`);
  const modes = Object.keys(labels.data ?? {});
  const modelLabel = MODEL_LABELS[model] ?? model;
  const inverted = (overview?.cells ?? []).some((cell) => cell.status !== "none");
  const pick = (position: number) => {
    const cell = nearestCell(overview?.cells ?? [], position);
    if (cell) onSelect(cell.key);
  };
  const range = {
    min: vsMin === "" ? undefined : Number(vsMin),
    max: vsMax === "" ? undefined : Number(vsMax),
  };

  const inversionCard = card.data;
  return (
    <>
      <StageHead overview={overview} error={overviewError} />
      {overview && !inverted ? (
        <section className="viz-section viz-card">
          <Empty>No window inverted yet: invert them in Seismic inversion, or ask the assistant.</Empty>
        </section>
      ) : (
        <>
          {selected && (
            <div className="viz-section">
              <UnitCard
                stage="inversion"
                card={inversionCard}
                error={card.error}
                cells={overview?.cells ?? []}
                onSelect={onSelect}
                details={
                  inversionCard?.inverted && (
                    <>
                      <Fold title="Model, checks and attempts">
                        <ModelTable card={inversionCard} />
                        <div className="viz-section">
                          <GateTables gates={inversionCard.gates} />
                        </div>
                        {inversionCard.attempts.length > 0 && (
                          <div className="viz-section">
                            <h3 className="viz-h3">Attempts</h3>
                            <AttemptTable
                              attempts={inversionCard.attempts}
                              extra={[
                                { label: "Layers", render: (one) => (one.n_layers ? one.n_layers - 1 : "—") },
                                { label: "Half-space top", render: (one) => (one.depth_m ? `≤ ${num(one.depth_m)} m` : "—") },
                              ]}
                            />
                          </div>
                        )}
                      </Fold>
                      <Fold title="Chains and marginals" open={chainsOpen} onToggle={setChainsOpen}>
                        {chainsOpen && xmid !== null && <ChainsView folder={folder} xmid={xmid} />}
                      </Fold>
                      {inversionCard.figures.length > 0 && (
                        <Fold title="Saved figures">
                          <div className="viz-figures">
                            {inversionCard.figures.map((name) => {
                              const url = `${API}/quality/inversion/figure/${at(folder)}/${xmid}/${name}`;
                              return (
                                <a key={name} href={url} target="_blank" rel="noreferrer" title={name.replaceAll("_", " ")}>
                                  <img src={url} alt={name.replaceAll("_", " ")} loading="lazy" />
                                </a>
                              );
                            })}
                          </div>
                        </Fold>
                      )}
                    </>
                  )
                }
              >
                {inversionCard?.inverted && (
                  <div className="viz-plots-2">
                    {inversionCard.profile ? <VsProfilePlot profile={inversionCard.profile} /> : <Empty>No model saved.</Empty>}
                    {inversionCard.curve ? (
                      <div>
                        <CurveFitPlot
                          curve={inversionCard.curve}
                          axis={curveAxis}
                          modelled={`modelled (${modelLabel})`}
                          aside={
                            <Segmented
                              size="sm"
                              label="Along"
                              value={curveAxis}
                              onChange={setCurveAxis}
                              options={[
                                { value: "frequency", label: "Frequency" },
                                { value: "wavelength", label: "Wavelength" },
                              ]}
                            />
                          }
                        />
                      </div>
                    ) : (
                      <Empty>No picked curve.</Empty>
                    )}
                  </div>
                )}
              </UnitCard>
            </div>
          )}
          <Card
            className="viz-section"
            icon={<LayersIcon size={17} />}
            title="Vs, uncertainty and interface sections"
            plots
            hint={
              "Of the kept models\nVs: their median at each depth, P50\n" +
              "Vs uncertainty: (P90 − P10) / (2 P50), %\n" +
              "Interfaces: the share placing a layer boundary there"
            }
            aside={
              <div className="viz-toolbar" style={{ margin: 0 }}>
                <div
                  className="viz-field"
                  data-tip={DEPTH_INFORMED_TIP}
                >
                  Depth informed
                  <Segmented
                    size="sm"
                    label="Depth informed"
                    value={informed ? "on" : "off"}
                    onChange={(one) => setInformed(one === "on")}
                    options={[
                      { value: "off", label: "Off" },
                      { value: "on", label: "On" },
                    ]}
                  />
                </div>
                <SmoothingSwitch on={smoothing} onChange={setSmoothing} />
                <div className="viz-field center">
                  Vs range (m/s)
                  <div className="viz-range">
                    <input type="number" placeholder="Min" value={vsMin} onChange={(e) => setVsMin(e.target.value)} />
                    <span className="viz-muted">–</span>
                    <input type="number" placeholder="Max" value={vsMax} onChange={(e) => setVsMax(e.target.value)} />
                  </div>
                </div>
              </div>
            }
          >
            {section.data ? (
              <>
                <VelocitySectionCanvas
                  positions={section.data.positions}
                  elevations={section.data.elevations}
                  values={section.data.vs_grid}
                  colorLabel="Vs (m/s)"
                  colormap={terrain}
                  height={200}
                  link={zoomLink}
                  colorRange={range}
                  marker={xmid ?? undefined}
                  onPick={pick}
                  informed={overlay}
                />
                <VelocitySectionCanvas
                  positions={section.data.positions}
                  elevations={section.data.elevations}
                  values={section.data.vs_uncertainty_grid}
                  colorLabel="Vs uncertainty (%)"
                  colormap={afmhotR}
                  height={200}
                  link={zoomLink}
                  marker={xmid ?? undefined}
                  onPick={pick}
                  informed={overlay}
                />
                {section.data.interface_grid?.some((row) => row.some((value) => value !== null)) ? (
                  <VelocitySectionCanvas
                    positions={section.data.positions}
                    elevations={section.data.elevations}
                    values={section.data.interface_grid}
                    colorLabel="Interfaces (%)"
                    colormap={interfaceColours}
                    colorRange={{ min: 0 }}
                    height={200}
                    link={zoomLink}
                    marker={xmid ?? undefined}
                    onPick={pick}
                    informed={overlay}
                  />
                ) : (
                  <Empty>Interfaces: invert again to see them.</Empty>
                )}
                {overlay?.windows.some((one) => one.informed !== null) && (
                  <div className="viz-legend-inline">
                    <span style={{ color: colours.informed }} data-tip={DEPTH_INFORMED_TIP}>
                      <i className="dashed" style={{ borderTopWidth: 1 }} />
                      depth informed
                    </span>
                    <span data-tip={DEPTH_INFORMED_TIP}>
                      <i className="area" style={{ background: veiled(colours.veil) }} />
                      not informed by the data
                    </span>
                  </div>
                )}
              </>
            ) : section.missing ? (
              <Empty>Needs 2 inverted windows.</Empty>
            ) : section.error ? (
              <ErrorBox message={`Not loaded: ${section.error}`} />
            ) : (
              <Skeleton height={640} />
            )}
          </Card>
          {modes.length > 0 && (
            <Card
              className="viz-section"
              icon={<StrataIcon size={17} />}
              title="Picked and modelled pseudo-sections"
              plots
              aside={
                <Segmented
                  size="sm"
                  label="Vertical axis"
                  value={comparisonAxis}
                  onChange={setComparisonAxis}
                  options={[
                    { value: "frequency", label: "Frequency" },
                    { value: "wavelength", label: "Wavelength" },
                  ]}
                />
              }
            >
              <div className="stack">
                {modes.map((label) => (
                  <ComparedSection
                    key={label}
                    folder={folder}
                    label={label}
                    model={model}
                    axis={comparisonAxis}
                    windows={overview?.cells.length ?? 0}
                    marker={xmid ?? undefined}
                    onPick={pick}
                  />
                ))}
              </div>
            </Card>
          )}
        </>
      )}
    </>
  );
}

/** One mode's picked curves against the model's along the line (M0, M1…), as the picking page
 * heads them: the windows holding both. */
function ComparedSection({
  folder,
  label,
  model,
  axis,
  windows,
  marker,
  onPick,
}: {
  folder: string;
  label: string;
  model: ModelName;
  axis: "frequency" | "wavelength";
  windows: number;
  marker?: number;
  onPick: (position: number) => void;
}) {
  const comparison = useJson<PseudoSectionComparisonData>(
    `${API}/inversion/pseudo_section_comparison/${at(folder)}/${encodeURIComponent(label)}?model=${model}`,
  );
  return (
    <div>
      <ModeHead label={label} count={comparison.data?.positions.length} total={windows} unit="windows" />
      {comparison.data ? (
        <PseudoSectionComparisonCanvas
          comparison={comparison.data}
          velocityLabel="Phase velocity (m/s)"
          mode={axis}
          marker={marker}
          onPick={onPick}
        />
      ) : comparison.missing ? (
        <p className="faint">Needs 2 windows with a pick and a model.</p>
      ) : comparison.error ? (
        <ErrorBox message={`Not loaded: ${comparison.error}`} />
      ) : (
        <Skeleton height={420} />
      )}
    </div>
  );
}
