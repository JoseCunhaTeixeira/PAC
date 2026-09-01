import { useCallback, useEffect, useState } from "react";
import { API } from "./api";
import { terrain, viridis } from "./components/colormaps";
import { DispersionCurveCanvas } from "./components/DispersionCurveCanvas";
import { PetroSectionCanvas, type PetroSectionData } from "./components/PetroSectionCanvas";
import {
  PseudoSectionComparisonCanvas,
  type PseudoSectionComparisonData,
} from "./components/PseudoSectionComparisonCanvas";
import { RunPanel, type Job } from "./components/RunPanel";
import { VelocitySectionCanvas } from "./components/VelocitySectionCanvas";

const MODE_COLOR = "#3b82f6";
// GPa shear modulus values are typically 0.05-0.5 -- VelocitySectionCanvas's
// default 1-decimal formatting would round all of them to "0.0"/"0.1".
const formatGPa = (v: number) => v.toFixed(2);

// Picking labels aren't tied to a wave-type letter in this app (e.g. "M0",
// "L0", "R0" are all used) -- a pick counts as the fundamental mode when its
// trailing digits parse to mode number 0, regardless of the letter prefix.
function isFundamentalModeLabel(label: string): boolean {
  const match = /(\d+)$/.exec(label);
  return match !== null && Number(match[1]) === 0;
}

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

interface ContinuousSection {
  positions: number[];
  elevations: number[];
  values: (number | null)[][];
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

export default function PetroInversionPage() {
  const [folders, setFolders] = useState<string[]>([]);
  const [folder, setFolder] = useState("");

  const [xmids, setXmids] = useState<number[]>([]);
  const [positionPicks, setPositionPicks] = useState<{ xmid: number; labels: string[] }[]>([]);
  const [selectedPositions, setSelectedPositions] = useState<Record<number, boolean>>({});

  const [models, setModels] = useState<string[]>([]);
  const [modelName, setModelName] = useState("");
  const [nWorkers, setNWorkers] = useState(1);

  const [petroSection, setPetroSection] = useState<PetroSectionData | null>(null);
  const [shearModulusSection, setShearModulusSection] = useState<ContinuousSection | null>(null);
  const [vsSection, setVsSection] = useState<ContinuousSection | null>(null);
  const [positionCurves, setPositionCurves] = useState<PositionCurves[]>([]);
  const [pseudoComparison, setPseudoComparison] = useState<PseudoSectionComparisonData | null>(
    null,
  );

  const [error, setError] = useState<string | null>(null);
  const [loadingFolders, setLoadingFolders] = useState(true);
  // Starts true so the first render after picking a folder doesn't flash
  // "No picked dispersion data found" before the effect below runs.
  const [loadingPicks, setLoadingPicks] = useState(true);

  const nCpus = navigator.hardwareConcurrency || 1;

  useEffect(() => {
    fetch(`${API}/output_folders`)
      .then((res) => res.json())
      .then((data: string[]) => setFolders(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoadingFolders(false));
  }, []);

  useEffect(() => {
    fetch(`${API}/petro_inversion/models`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: string[]) => setModels(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const refreshPositionPicks = useCallback((folderName: string) => {
    Promise.resolve().then(() => setLoadingPicks(true));
    fetch(`${API}/dispersion_picks_by_position/${encodeURIComponent(folderName)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: { xmid: number; labels: string[] }[]) => setPositionPicks(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoadingPicks(false));
  }, []);

  useEffect(() => {
    Promise.resolve().then(() => {
      setXmids([]);
      setPositionPicks([]);
      setSelectedPositions({});
      setPetroSection(null);
      setShearModulusSection(null);
      setVsSection(null);
      setPositionCurves([]);
      setPseudoComparison(null);
    });
    if (!folder) return;

    fetch(`${API}/xmids/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: number[]) => setXmids(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));

    refreshPositionPicks(folder);
  }, [folder, refreshPositionPicks]);

  const eligibleXmids = xmids.filter((xmid) => {
    const picks = positionPicks.find((p) => p.xmid === xmid);
    return !!picks && picks.labels.some(isFundamentalModeLabel);
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
    model_name: modelName,
    n_workers: nWorkers,
  };

  const missing: string[] = [];
  if (!folder) missing.push("a data folder");
  if (!modelName) missing.push("a Silex model");
  if (selectedXmids.length === 0) missing.push("at least one position to invert");

  function loadResults(forFolder: string) {
    fetch(`${API}/petro_inversion/section/${encodeURIComponent(forFolder)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: PetroSectionData) => setPetroSection(data))
      // Best-effort: a folder with too few inverted positions just skips the
      // section instead of blocking the rest of the page.
      .catch(() => setPetroSection(null));

    fetch(`${API}/petro_inversion/shear_modulus_section/${encodeURIComponent(forFolder)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: ContinuousSection) => setShearModulusSection(data))
      .catch(() => setShearModulusSection(null));

    fetch(`${API}/petro_inversion/vs_section/${encodeURIComponent(forFolder)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: ContinuousSection) => setVsSection(data))
      .catch(() => setVsSection(null));

    fetch(`${API}/petro_inversion/curves/${encodeURIComponent(forFolder)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: PositionCurves[]) => setPositionCurves(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));

    fetch(`${API}/petro_inversion/pseudo_section_comparison/${encodeURIComponent(forFolder)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: PseudoSectionComparisonData) => setPseudoComparison(data))
      .catch(() => setPseudoComparison(null));
  }

  function handleJobDone(job: Job) {
    if (job.state === "succeeded") loadResults(folder);
  }

  const velocityType = positionCurves.map((c) => c.velocity_type).find((t) => t);
  const yAxisLabel = VELOCITY_TYPE_LABELS[velocityType ?? ""] ?? "Velocity [m/s]";
  const modeLabel = positionPicks.flatMap((p) => p.labels).find(isFundamentalModeLabel) ?? "M0";

  return (
    <div style={{ padding: 24 }}>
      <h1>Petrophysical Inversion</h1>
      <p>🛈 Only inverts the fundamental mode (mode number 0, e.g. "M0", "R0") dispersion curve.</p>

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
        {!loadingFolders && folders.length === 0 && <p>❌ No folders found.</p>}
      </div>

      {error && <p style={{ color: "var(--accent)" }}>Error: {error}</p>}

      {folder && !loadingPicks && eligibleXmids.length === 0 && (
        <p>
          ❌ No positions with a picked fundamental-mode (e.g. "M0", "R0" — mode number 0)
          dispersion curve found.
        </p>
      )}

      {folder && eligibleXmids.length > 0 && (
        <>
          <h2>Positions to invert</h2>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
            <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <input
                type="checkbox"
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

          <h2>Model</h2>
          <label>
            Silex model:{" "}
            <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
              <option value="">— choose —</option>
              {models.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          {models.length === 0 && (
            <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
              ❌ No Silex models found.
            </p>
          )}

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
              runUrl="/petro_inversion/run"
              itemLabel="positions"
              itemLabelSingular="position"
              onDone={handleJobDone}
            />
          )}
        </>
      )}

      {petroSection ? (
        <>
          <h2>Petrophysical section</h2>
          <PetroSectionCanvas section={petroSection} />
        </>
      ) : positionCurves.length > 0 ? (
        <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
          🛈 At least 2 inverted positions are required to build a section.
        </p>
      ) : null}

      {shearModulusSection && (
        <>
          <h2>Shear modulus section</h2>
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            🛈 Hertz-Mindlin effective shear modulus, from santiludo's rock-physics forward model.
          </p>
          <VelocitySectionCanvas
            positions={shearModulusSection.positions}
            elevations={shearModulusSection.elevations}
            values={shearModulusSection.values}
            colorLabel="Shear modulus [GPa]"
            colormap={viridis}
            height={200}
            formatValue={formatGPa}
          />
        </>
      )}

      {vsSection && (
        <>
          <h2>Vs section</h2>
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            🛈 Shear wave velocity forward-modeled from the petro model via santiludo, before the
            fixed-substratum dispersion curve fit.
          </p>
          <VelocitySectionCanvas
            positions={vsSection.positions}
            elevations={vsSection.elevations}
            values={vsSection.values}
            colorLabel="Vs [m/s]"
            colormap={terrain}
            height={200}
          />
        </>
      )}

      {positionCurves.length > 0 && (
        <>
          <h2>Observed vs predicted dispersion</h2>
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            🛈 Solid: observed. Dashed: predicted.{" "}
            <span style={{ color: MODE_COLOR }}>{modeLabel}</span>
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, max-content)", gap: 8 }}>
            {positionCurves.map((c, i) => {
              const col = i % 4;
              const isLastInColumn = i + 4 >= positionCurves.length;
              return (
                <DispersionCurveCanvas
                  key={c.xmid}
                  title={`${c.xmid.toFixed(2)} m`}
                  xLabel={isLastInColumn ? "Frequency [Hz]" : undefined}
                  yLabel={col === 0 ? yAxisLabel : undefined}
                  series={[
                    {
                      label: modeLabel,
                      color: MODE_COLOR,
                      observedFs: c.observed_fs,
                      observedVs: c.observed_vs,
                      observedVsErr: c.observed_vs_err,
                      predictedFs: c.predicted_fs,
                      predictedVs: c.predicted_vs,
                    },
                  ]}
                />
              );
            })}
          </div>

          <h2>Pseudo-section comparison</h2>
          {pseudoComparison ? (
            <>
              <h4>{modeLabel}</h4>
              <PseudoSectionComparisonCanvas comparison={pseudoComparison} velocityLabel={yAxisLabel} />
            </>
          ) : (
            <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
              🛈 At least 2 positions with both a pick and a petro inversion result are required
              to build a pseudo-section comparison.
            </p>
          )}
        </>
      )}
    </div>
  );
}
