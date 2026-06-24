import { useEffect, useState } from "react";
import { API } from "./api";
import { VelocitySectionCanvas } from "./components/VelocitySectionCanvas";
import { DispersionCurveCanvas } from "./components/DispersionCurveCanvas";
import { RunPanel, type Job } from "./components/RunPanel";
import { terrain, afmhotR } from "./components/colormaps";

const MODE_COLORS = ["#3b82f6", "#f97316", "#10b981", "#a855f7", "#ec4899", "#14b8a6"];

const VELOCITY_TYPE_LABELS: Record<string, string> = {
  phase: "Phase velocity [m/s]",
  group: "Group velocity [m/s]",
};

interface PositionCurves {
  xmid: number;
  observed_fs: number[] | null;
  observed_vs: number[] | null;
  observed_vs_err: number[] | null;
  predicted_fs: number[] | null;
  predicted_vs: number[] | null;
  velocity_type: string;
}

interface VsLayer {
  vs_min: number;
  vs_max: number;
  vs_perturb_std: number;
}

interface ThicknessLayer {
  thickness_min: number;
  thickness_max: number;
  thickness_perturb_std: number;
}

interface VelocitySection {
  positions: number[];
  elevations: number[];
  vs_grid: (number | null)[][];
  vs_std_grid: (number | null)[][];
}

function defaultVsLayer(): VsLayer {
  return { vs_min: 100, vs_max: 1000, vs_perturb_std: 20 };
}

function defaultThicknessLayer(): ThicknessLayer {
  return { thickness_min: 1, thickness_max: 10, thickness_perturb_std: 1 };
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label style={{ display: "block", margin: "4px 0" }}>
      {label}:{" "}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export default function InversionPage() {
  const [folders, setFolders] = useState<string[]>([]);
  const [folder, setFolder] = useState("");

  const [xmids, setXmids] = useState<number[]>([]);
  const [labelCounts, setLabelCounts] = useState<Record<string, number>>({});
  const [positionPicks, setPositionPicks] = useState<{ xmid: number; labels: string[] }[]>([]);

  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);
  const [selectedPositions, setSelectedPositions] = useState<Record<number, boolean>>({});

  const [nLayers, setNLayers] = useState(2);
  const [vsLayers, setVsLayers] = useState<VsLayer[]>([defaultVsLayer(), defaultVsLayer()]);
  const [thicknessLayers, setThicknessLayers] = useState<ThicknessLayer[]>([
    defaultThicknessLayer(),
  ]);

  const [nIterations, setNIterations] = useState(100_000);
  const [nBurninIterations, setNBurninIterations] = useState(10_000);
  const [nChains, setNChains] = useState(5);
  const [nWorkers, setNWorkers] = useState(1);

  const [velocitySection, setVelocitySection] = useState<VelocitySection | null>(null);
  const [positionCurves, setPositionCurves] = useState<Record<string, PositionCurves[]>>({});
  const [resultLabels, setResultLabels] = useState<string[]>([]);

  const [error, setError] = useState<string | null>(null);

  const nCpus = navigator.hardwareConcurrency || 1;

  useEffect(() => {
    fetch(`${API}/output_folders`)
      .then((res) => res.json())
      .then((data: string[]) => setFolders(data))
      .catch((err) => setError(String(err)));
  }, []);

  function refreshLabels(folderName: string) {
    fetch(`${API}/dispersion_image_labels/${encodeURIComponent(folderName)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: Record<string, number>) => setLabelCounts(data))
      .catch((err) => setError(String(err)));
  }

  function refreshPositionPicks(folderName: string) {
    fetch(`${API}/dispersion_picks_by_position/${encodeURIComponent(folderName)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: { xmid: number; labels: string[] }[]) => setPositionPicks(data))
      .catch((err) => setError(String(err)));
  }

  useEffect(() => {
    setXmids([]);
    setLabelCounts({});
    setPositionPicks([]);
    setSelectedLabels([]);
    setSelectedPositions({});
    setVelocitySection(null);
    setPositionCurves({});
    setResultLabels([]);
    if (!folder) return;

    fetch(`${API}/xmids/${encodeURIComponent(folder)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: number[]) => setXmids(data))
      .catch((err) => setError(String(err)));

    refreshLabels(folder);
    refreshPositionPicks(folder);
  }, [folder]);

  useEffect(() => {
    setSelectedPositions({});
  }, [selectedLabels]);

  useEffect(() => {
    setVsLayers((prev) => {
      const next = prev.slice(0, nLayers);
      while (next.length < nLayers) next.push(defaultVsLayer());
      return next;
    });
    setThicknessLayers((prev) => {
      const n = Math.max(0, nLayers - 1);
      const next = prev.slice(0, n);
      while (next.length < n) next.push(defaultThicknessLayer());
      return next;
    });
  }, [nLayers]);

  function toggleLabel(labelValue: string) {
    setSelectedLabels((prev) =>
      prev.includes(labelValue) ? prev.filter((l) => l !== labelValue) : [...prev, labelValue],
    );
  }

  const allLabels = Object.keys(labelCounts);
  const allLabelsSelected =
    allLabels.length > 0 && allLabels.every((lbl) => selectedLabels.includes(lbl));

  function toggleAllLabels() {
    setSelectedLabels(allLabelsSelected ? [] : allLabels);
  }

  const eligibleXmids = xmids.filter((xmid) => {
    const picks = positionPicks.find((p) => p.xmid === xmid);
    return !!picks && picks.labels.some((l) => selectedLabels.includes(l));
  });
  const allEligibleSelected =
    eligibleXmids.length > 0 && eligibleXmids.every((xmid) => selectedPositions[xmid]);

  function toggleAllPositions() {
    const next = !allEligibleSelected;
    setSelectedPositions(Object.fromEntries(eligibleXmids.map((xmid) => [xmid, next])));
  }

  function togglePosition(xmid: number) {
    setSelectedPositions((prev) => ({ ...prev, [xmid]: !prev[xmid] }));
  }

  const selectedXmids = eligibleXmids.filter((xmid) => selectedPositions[xmid]);
  const maxWorkers = selectedXmids.length > 0 ? Math.min(nCpus, selectedXmids.length) : nCpus;

  const config = {
    folder,
    positions: selectedXmids,
    labels: selectedLabels,
    parameters: {
      n_layers: nLayers,
      vs_layers: vsLayers,
      thickness_layers: thicknessLayers,
      n_iterations: nIterations,
      n_burnin_iterations: nBurninIterations,
      n_chains: nChains,
    },
    n_workers: nWorkers,
  };

  const missing: string[] = [];
  if (!folder) missing.push("a data folder");
  if (selectedLabels.length === 0) missing.push("at least one mode to invert");
  if (selectedXmids.length === 0) missing.push("at least one position to invert");

  function loadResults(invertedLabels: string[]) {
    setResultLabels(invertedLabels);
    fetch(`${API}/inversion/velocity_section/${encodeURIComponent(folder)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: VelocitySection) => setVelocitySection(data))
      .catch((err) => setError(String(err)));

    invertedLabels.forEach((labelValue) => {
      fetch(
        `${API}/inversion/curves/${encodeURIComponent(folder)}/${encodeURIComponent(labelValue)}`,
      )
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data: PositionCurves[]) =>
          setPositionCurves((prev) => ({ ...prev, [labelValue]: data })),
        )
        .catch((err) => setError(String(err)));
    });
  }

  function handleJobDone(job: Job) {
    if (job.state === "succeeded") loadResults(selectedLabels);
  }

  return (
    <div style={{ padding: 24 }}>
      <h1>Inversion</h1>
      {/* <p>🛈 Surface wave dispersion inversion.</p> */}

      <div style={{ marginBottom: 32 }}>
        <label>
          <h2>Loading</h2>
          Data folder:{" "}
          <select value={folder} onChange={(e) => setFolder(e.target.value)}>
            <option value="">— choose —</option>
            {folders.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error && <p style={{ color: "var(--accent)" }}>Error: {error}</p>}

      {folder && Object.keys(labelCounts).length === 0 && (
        <p>No picked dispersion data found. Select another folder.</p>
      )}

      {folder && Object.keys(labelCounts).length > 0 && (
        <>

          <h2>Modes to invert</h2>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <input type="checkbox" checked={allLabelsSelected} onChange={toggleAllLabels} />
              All
            </label>
            {allLabels.map((lbl) => {
              const checked = selectedLabels.includes(lbl);
              return (
                <label
                  key={lbl}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    padding: "6px 10px",
                    borderRadius: 6,
                    background: checked ? "var(--success-bg)" : "var(--surface-hover)",
                  }}
                >
                  <input type="checkbox" checked={checked} onChange={() => toggleLabel(lbl)} />
                  {lbl}
                </label>
              );
            })}
          </div>
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            🛈 The maximum number of selected modes will be inverted where they were picked.
          </p>

          <>
            <h2>Positions to invert</h2>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  opacity: eligibleXmids.length > 0 ? 1 : 0.4,
                }}
              >
                <input
                  type="checkbox"
                  disabled={eligibleXmids.length === 0}
                  checked={allEligibleSelected}
                  onChange={toggleAllPositions}
                />
                All
              </label>
              {xmids.map((xmid) => {
                const eligible = eligibleXmids.includes(xmid);
                const checked = !!selectedPositions[xmid];
                return (
                  <label
                    key={xmid}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 4,
                      opacity: eligible ? 1 : 0.4,
                      minWidth: 64,
                      padding: "6px 10px",
                      borderRadius: 6,
                      background: checked ? "var(--success-bg)" : "var(--surface-hover)",
                    }}
                  >
                    <input
                      type="checkbox"
                      disabled={!eligible}
                      checked={checked}
                      onChange={() => togglePosition(xmid)}
                    />
                    {xmid.toFixed(2)} m
                  </label>
                );
              })}
            </div>

            <h2>Parameter space</h2>

            <NumberField
              label="Number of layers (including half-space)"
              value={nLayers}
              onChange={(v) => setNLayers(Math.max(2, Math.round(v)))}
              min={2}
              step={1}
            />

            {vsLayers.map((layer, i) => {
              const isHalfSpace = i === nLayers - 1;

              return (
                <div
                  key={i}
                  style={{
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    padding: 12,
                    marginTop: 12,
                  }}
                >
                  <strong style={{ fontSize: "1.1rem" }}>
                    {isHalfSpace ? "Half-space:" : `Layer ${i + 1}:`}
                  </strong>

                  {!isHalfSpace && (
                    <>
                      <h4 style={{ marginTop: 30, marginBottom: 8 }}>Thickness</h4>
                      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                        <NumberField
                          label="Min [m]"
                          value={thicknessLayers[i]?.thickness_min ?? 1}
                          step={0.1}
                          min={0.1}
                          onChange={(v) =>
                            setThicknessLayers((prev) =>
                              prev.map((l, j) =>
                                j === i ? { ...l, thickness_min: v } : l,
                              ),
                            )
                          }
                        />

                        <NumberField
                          label="Max [m]"
                          value={thicknessLayers[i]?.thickness_max ?? 10}
                          step={0.1}
                          min={0.1}
                          onChange={(v) =>
                            setThicknessLayers((prev) =>
                              prev.map((l, j) =>
                                j === i ? { ...l, thickness_max: v } : l,
                              ),
                            )
                          }
                        />

                        <NumberField
                          label="Std [m]"
                          value={thicknessLayers[i]?.thickness_perturb_std ?? 1}
                          step={0.01}
                          min={0.01}
                          onChange={(v) =>
                            setThicknessLayers((prev) =>
                              prev.map((l, j) =>
                                j === i
                                  ? { ...l, thickness_perturb_std: v }
                                  : l,
                              ),
                            )
                          }
                        />
                      </div>
                    </>
                  )}

                  <h4 style={{ marginTop: 30, marginBottom: 8 }}>Vs</h4>
                  <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                    <NumberField
                      label="Min [m/s]"
                      value={layer.vs_min}
                      step={10}
                      min={10}
                      onChange={(v) =>
                        setVsLayers((prev) =>
                          prev.map((l, j) =>
                            j === i ? { ...l, vs_min: v } : l,
                          ),
                        )
                      }
                    />

                    <NumberField
                      label="Max [m/s]"
                      value={layer.vs_max}
                      step={10}
                      min={10}
                      onChange={(v) =>
                        setVsLayers((prev) =>
                          prev.map((l, j) =>
                            j === i ? { ...l, vs_max: v } : l,
                          ),
                        )
                      }
                    />

                    <NumberField
                      label="Std [m/s]"
                      value={layer.vs_perturb_std}
                      step={1}
                      min={1}
                      onChange={(v) =>
                        setVsLayers((prev) =>
                          prev.map((l, j) =>
                            j === i
                              ? { ...l, vs_perturb_std: v }
                              : l,
                          ),
                        )
                      }
                    />
                  </div>
                </div>
              );
            })}
            <h2>Running parameters</h2>
            <NumberField label="Iterations [#]" value={nIterations} min={1} step={100} onChange={setNIterations} />
            <NumberField
              label="Burn-in iterations [#]"
              value={nBurninIterations}
              min={1}
              step={100}
              onChange={setNBurninIterations}
            />
            <NumberField label="Chains [#]" value={nChains} min={1} step={1} onChange={setNChains} />

            <h2>Execution</h2>
            <NumberField
              label="Number of workers"
              value={nWorkers}
              min={1}
              step={1}
              onChange={(v) => setNWorkers(Math.min(v, maxWorkers))}
            />
            {missing.length > 0 ? (
              <>
                <button disabled>Compute</button>
                <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  Missing: {missing.join(", ")}.
                </p>
              </>
            ) : (
              <RunPanel
                config={config}
                runUrl="/inversion/run"
                itemLabel="positions"
                itemLabelSingular="position"
                onDone={handleJobDone}
              />
            )}
          </>
        </>
      )}

      {velocitySection && (
        <>
          <h2>Inverted Vs section</h2>
          <VelocitySectionCanvas
            positions={velocitySection.positions}
            elevations={velocitySection.elevations}
            values={velocitySection.vs_grid}
            colorLabel="Vs [m/s]"
            colormap={terrain}
            height={200}
          />
          <VelocitySectionCanvas
            positions={velocitySection.positions}
            elevations={velocitySection.elevations}
            values={velocitySection.vs_std_grid}
            colorLabel="Vs std [m/s]"
            colormap={afmhotR}
            height={200}
          />
        </>
      )}

      {resultLabels.length > 0 && (
        <>
          <h2>Observed vs predicted dispersion</h2>
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            🛈 Solid: observed. Dashed: predicted.{" "}
            {resultLabels.map((lbl, i) => (
              <span key={lbl} style={{ color: MODE_COLORS[i % MODE_COLORS.length], marginRight: 8 }}>
                {lbl}
              </span>
            ))}
          </p>
          {(() => {
            const items = positionCurves[resultLabels[0]] ?? [];
            const numCols = 4;
            const velocityType = resultLabels
              .flatMap((lbl) => positionCurves[lbl] ?? [])
              .map((c) => c.velocity_type)
              .find((t) => t);
            const yAxisLabel = VELOCITY_TYPE_LABELS[velocityType ?? ""] ?? "Velocity [m/s]";
            return (
              <div style={{ display: "grid", gridTemplateColumns: `repeat(${numCols}, max-content)`, gap: 8 }}>
                {items.map(({ xmid }, i) => {
                  const col = i % numCols;
                  const isLastInColumn = i + numCols >= items.length;
                  return (
                    <DispersionCurveCanvas
                      key={xmid}
                      title={`${xmid.toFixed(2)} m`}
                      xLabel={isLastInColumn ? "Frequency [Hz]" : undefined}
                      yLabel={col === 0 ? yAxisLabel : undefined}
                      series={resultLabels.map((lbl, j) => {
                        const c = (positionCurves[lbl] ?? []).find((p) => p.xmid === xmid);
                        return {
                          label: lbl,
                          color: MODE_COLORS[j % MODE_COLORS.length],
                          observedFs: c?.observed_fs ?? null,
                          observedVs: c?.observed_vs ?? null,
                          observedVsErr: c?.observed_vs_err ?? null,
                          predictedFs: c?.predicted_fs ?? null,
                          predictedVs: c?.predicted_vs ?? null,
                        };
                      })}
                    />
                  );
                })}
              </div>
            );
          })()}
        </>
      )}
    </div>
  );
}
