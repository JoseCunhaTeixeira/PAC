import { useState } from "react";
import { API } from "../../api";
import { afmhotR, terrain } from "../colormaps";
import { PseudoSectionComparisonCanvas, type PseudoSectionComparisonData } from "../PseudoSectionComparisonCanvas";
import { useZoomLink } from "../useZoom";
import { VelocitySectionCanvas } from "../VelocitySectionCanvas";
import { ChainLegend, ChainTracesCanvas, MarginalsGrid } from "./ChainPlots";
import { num, parameterLabel, xmidOf } from "./format";
import { StageHead, UnitCard } from "./panel";
import { CurveFitPlot, VsProfilePlot, type CurveAxis } from "./plots";
import type { Chains, InversionCard, ModelName, Overview } from "./types";
import { AttemptTable, Empty, Fold, GateTables, Skeleton } from "./ui";
import { useJson } from "./useJson";

// The seismic inversion stage: the display settings, the selected window's card (how deep its
// data inform its model, its fit, its chains, what it ran with and each attempt's change), its
// model against depth beside its curve against the model's, and along the line, the Vs section
// and its spread, and the picked curves against the models'.

const at = (folder: string) => encodeURIComponent(folder);

const MODELS: { value: ModelName; label: string }[] = [
  { value: "smooth_median", label: "Smooth median" },
  { value: "median", label: "Median, layered" },
  { value: "smooth_best", label: "Smooth best" },
  { value: "best", label: "Best, layered" },
  { value: "ensemble", label: "Median of the ensemble" },
];

interface VelocitySection {
  positions: number[];
  elevations: number[];
  vs_grid: (number | null)[][];
  vs_std_grid: (number | null)[][];
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
        .
      </p>
      <div className="viz-table-wrap">
        <table className="viz-table">
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Prior</th>
              <th data-tip={"Result\nThe median of every chain's samples"}>Result</th>
              <th data-tip={"10–90 %\nThe range holding the samples' middle 80 %"}>10–90 %</th>
              <th>R-hat</th>
              <th>Effective samples</th>
              <th>Lag-1 autocorr.</th>
              <th>Step</th>
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
                <td className="num">{num(row.step)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
      <div className="viz-toolbar" style={{ marginTop: 8 }}>
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
      </div>
      {traces && <ChainTracesCanvas traces={traces} prior={marginal ? [marginal.low, marginal.high] : undefined} />}
      <p className="viz-plot-title" style={{ marginTop: 12 }}>
        Marginals within the priors (dashed: a flat posterior)
      </p>
      <MarginalsGrid marginals={chains.data.marginals} />
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
  const [model, setModel] = useState<ModelName>("smooth_median");
  const [smoothing, setSmoothing] = useState(false);
  const [vsMin, setVsMin] = useState("");
  const [vsMax, setVsMax] = useState("");
  const [curveAxis, setCurveAxis] = useState<CurveAxis>("frequency");
  const [chainsOpen, setChainsOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const zoomLink = useZoomLink(folder);
  const xmid = selected ? xmidOf(selected) : null;
  const card = useJson<InversionCard>(
    xmid !== null ? `${API}/quality/inversion/card/${at(folder)}/${xmid}?model=${model}` : null,
  );
  const section = useJson<VelocitySection>(
    `${API}/inversion/velocity_section/${at(folder)}?model=${model}&lateral_smoothing=${smoothing}`,
  );
  const labels = useJson<Record<string, number>>(`${API}/dispersion_image_labels/${at(folder)}`);
  const label = Object.keys(labels.data ?? {})[0] ?? null;
  const comparison = useJson<PseudoSectionComparisonData>(
    label ? `${API}/inversion/pseudo_section_comparison/${at(folder)}/${encodeURIComponent(label)}?model=${model}` : null,
  );
  const modelLabel = MODELS.find((one) => one.value === model)?.label.toLowerCase() ?? model;
  const inverted = (overview?.cells ?? []).some((cell) => cell.status !== "none");
  const pick = (position: number) => {
    const cell = overview?.cells.find((one) => one.x !== null && Math.abs(one.x - position) < 1e-6);
    if (cell) onSelect(cell.key);
  };
  const range = {
    min: vsMin === "" ? undefined : Number(vsMin),
    max: vsMax === "" ? undefined : Number(vsMax),
  };

  function save() {
    setSaving(true);
    setSaved(null);
    fetch(`${API}/inversion/save_images/${at(folder)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels: Object.keys(labels.data ?? {}), model, lateral_smoothing: smoothing }),
    })
      .then(async (res) => {
        const body = await res.json().catch(() => null);
        if (!res.ok) throw new Error(body?.detail ?? `HTTP ${res.status}`);
        return body as { saved_paths: string[]; errors: string[] };
      })
      .then((data) =>
        setSaved(
          `Saved ${data.saved_paths.length} image${data.saved_paths.length === 1 ? "" : "s"} in the run's folder` +
            (data.errors.length ? `; ${data.errors.length} skipped: ${data.errors.join("; ")}` : "."),
        ),
      )
      .catch((err) => setSaved(`Not saved: ${err instanceof Error ? err.message : String(err)}`))
      .finally(() => setSaving(false));
  }

  const inversionCard = card.data;
  return (
    <>
      <StageHead
        overview={overview}
        error={overviewError}
        aside={
          <label className="viz-field viz-inline" data-tip={"Model\nThe one the plots below show"}>
            Model
            <select value={model} onChange={(e) => setModel(e.target.value as ModelName)}>
              {MODELS.map((one) => (
                <option key={one.value} value={one.value}>
                  {one.label}
                </option>
              ))}
            </select>
          </label>
        }
      />
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
                        <CurveFitPlot curve={inversionCard.curve} axis={curveAxis} modelled={`the ${modelLabel} model's`} />
                        <div className="viz-segment" style={{ marginTop: 6 }} role="group" aria-label="Along">
                          {(["frequency", "wavelength"] as const).map((one) => (
                            <button key={one} type="button" className={curveAxis === one ? "active" : ""} onClick={() => setCurveAxis(one)}>
                              {one === "frequency" ? "Frequency" : "Wavelength"}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <Empty>No picked curve.</Empty>
                    )}
                  </div>
                )}
              </UnitCard>
            </div>
          )}
          <section className="viz-section viz-card">
            <div className="viz-row" style={{ alignItems: "flex-end", marginBottom: 10 }}>
              <h2 className="viz-h2" style={{ margin: 0 }}>
                Vs section
              </h2>
              <div className="viz-toolbar" style={{ margin: 0 }}>
                <div className="viz-field">
                  Lateral smoothing
                  <div className="viz-segment" role="group" aria-label="Lateral smoothing">
                    {[false, true].map((on) => (
                      <button key={String(on)} type="button" className={smoothing === on ? "active" : ""} onClick={() => setSmoothing(on)}>
                        {on ? "On" : "Off"}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="viz-field">
                  Vs colours (m/s)
                  <div className="viz-range">
                    <input type="number" placeholder="Min" value={vsMin} onChange={(e) => setVsMin(e.target.value)} />
                    <span className="viz-muted">–</span>
                    <input type="number" placeholder="Max" value={vsMax} onChange={(e) => setVsMax(e.target.value)} />
                  </div>
                </div>
                <button type="button" className="viz-button" onClick={save} disabled={saving}>
                  {saving ? "Saving…" : "Save images"}
                </button>
              </div>
            </div>
            {saved && <p className="viz-small viz-muted" style={{ margin: "0 0 8px" }}>{saved}</p>}
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
                />
                <VelocitySectionCanvas
                  positions={section.data.positions}
                  elevations={section.data.elevations}
                  values={section.data.vs_std_grid}
                  colorLabel="Vs std (m/s)"
                  colormap={afmhotR}
                  height={200}
                  link={zoomLink}
                  marker={xmid ?? undefined}
                  onPick={pick}
                />
              </>
            ) : section.error ? (
              <Empty>Needs 2 inverted windows.</Empty>
            ) : (
              <Skeleton height={420} />
            )}
          </section>
          {label && (
            <section className="viz-section viz-card">
              <h2 className="viz-h2">
                Picked and modelled {label}
              </h2>
              {comparison.data ? (
                <PseudoSectionComparisonCanvas
                  comparison={comparison.data}
                  velocityLabel="Phase velocity (m/s)"
                  marker={xmid ?? undefined}
                  onPick={pick}
                />
              ) : comparison.error ? (
                <Empty>Needs 2 windows with a pick and a model.</Empty>
              ) : (
                <Skeleton height={420} />
              )}
            </section>
          )}
        </>
      )}
    </>
  );
}
