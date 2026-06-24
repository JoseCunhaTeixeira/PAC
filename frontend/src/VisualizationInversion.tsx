import { useEffect, useState } from "react";
import { API } from "./api";
import { VelocitySectionCanvas } from "./components/VelocitySectionCanvas";
import { DispersionCurveCanvas } from "./components/DispersionCurveCanvas";
import { PseudoSectionComparisonCanvas, type PseudoSectionComparisonData } from "./components/PseudoSectionComparisonCanvas";
import { terrain, afmhotR } from "./components/colormaps";

type ModelName = "best" | "smooth_best" | "median" | "smooth_median" | "ensemble";

const MODEL_OPTIONS: { value: ModelName; label: string }[] = [
  { value: "best", label: "Best layered model" },
  { value: "smooth_best", label: "Smooth best layered model" },
  { value: "median", label: "Median layered model" },
  { value: "smooth_median", label: "Smooth median layered model" },
  { value: "ensemble", label: "Median ensemble model" },
];

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

interface VelocitySection {
  positions: number[];
  elevations: number[];
  vs_grid: (number | null)[][];
  vs_std_grid: (number | null)[][];
}

export function VisualizationInversion({ folder }: { folder: string }) {
  const [model, setModel] = useState<ModelName>("smooth_median");
  const [lateralSmoothing, setLateralSmoothing] = useState(false);
  const [vsMin, setVsMin] = useState<number | "">("");
  const [vsMax, setVsMax] = useState<number | "">("");

  const [labels, setLabels] = useState<string[]>([]);
  const [velocitySection, setVelocitySection] = useState<VelocitySection | null>(null);
  const [positionCurves, setPositionCurves] = useState<Record<string, PositionCurves[]>>({});
  const [pseudoComparisons, setPseudoComparisons] = useState<
    Record<string, PseudoSectionComparisonData | null>
  >({});
  const [error, setError] = useState<string | null>(null);
  const [saveResult, setSaveResult] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setLabels([]);
    setVelocitySection(null);
    setPositionCurves({});
    setPseudoComparisons({});
    setError(null);
    setSaveResult(null);

    fetch(`${API}/dispersion_image_labels/${encodeURIComponent(folder)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: Record<string, number>) => setLabels(Object.keys(data)))
      .catch((err) => setError(String(err)));
  }, [folder]);

  useEffect(() => {
    if (!folder) return;
    setError(null);

    fetch(
      `${API}/inversion/velocity_section/${encodeURIComponent(folder)}?model=${model}&lateral_smoothing=${lateralSmoothing}`,
    )
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: VelocitySection) => setVelocitySection(data))
      .catch((err) => setError(String(err)));

    labels.forEach((labelValue) => {
      fetch(
        `${API}/inversion/curves/${encodeURIComponent(folder)}/${encodeURIComponent(labelValue)}?model=${model}`,
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
  }, [folder, model, lateralSmoothing, labels]);

  useEffect(() => {
    if (!folder) return;

    // Independent of lateralSmoothing: the comparison is built from picked
    // curves vs. the model's forward-modeled curves, not the Vs(x,z) grid.
    labels.forEach((labelValue) => {
      fetch(
        `${API}/inversion/pseudo_section_comparison/${encodeURIComponent(folder)}/${encodeURIComponent(labelValue)}?model=${model}`,
      )
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data: PseudoSectionComparisonData) =>
          setPseudoComparisons((prev) => ({ ...prev, [labelValue]: data })),
        )
        // Best-effort per label: a label with too few positions just shows
        // nothing instead of blocking the rest of the page.
        .catch(() => setPseudoComparisons((prev) => ({ ...prev, [labelValue]: null })));
    });
  }, [folder, model, labels]);

  useEffect(() => {
    if (!velocitySection) return;
    if (vsMin !== "" || vsMax !== "") return; // user already set a range; don't override it

    let min = Infinity, max = -Infinity;
    for (const row of velocitySection.vs_grid) {
      for (const v of row) {
        if (v !== null) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
    }
    if (Number.isFinite(min)) {
      setVsMin(Math.floor(min));
      setVsMax(Math.ceil(max));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [velocitySection]);

  function handleSaveImages() {
    setSaving(true);
    setSaveResult(null);
    fetch(`${API}/inversion/save_images/${encodeURIComponent(folder)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels, model, lateral_smoothing: lateralSmoothing }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: { saved_paths: string[]; errors: string[] }) => {
        const parts = [`Saved ${data.saved_paths.length} image(s).`];
        if (data.errors.length > 0) parts.push(`${data.errors.length} skipped: ${data.errors.join("; ")}`);
        setSaveResult(parts.join(" "));
      })
      .catch((err) => setError(String(err)))
      .finally(() => setSaving(false));
  }

  if (error) return <p style={{ color: "var(--accent)" }}>Error: {error}</p>;

  if (!velocitySection) return null;

  const clippedVsGrid =
    vsMin === "" && vsMax === ""
      ? velocitySection.vs_grid
      : velocitySection.vs_grid.map((row) =>
        row.map((v) => {
          if (v === null) return v;
          if (vsMin !== "" && v < vsMin) return vsMin;
          if (vsMax !== "" && v > vsMax) return vsMax;
          return v;
        }),
      );

  return (
    <>
      <h2>Shear wave velocity profile</h2>
      <h4>Display settings</h4>
      <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 12, display: "flex", gap: 140, flexWrap: "wrap" }}>
        <div>
          <strong>Model</strong>
          {MODEL_OPTIONS.map((opt) => (
            <label key={opt.value} style={{ display: "block" }}>
              <input
                type="radio"
                checked={model === opt.value}
                onChange={() => setModel(opt.value)}
              />{" "}
              {opt.label}
            </label>
          ))}
        </div>
        <div>
          <strong>Lateral smoothing</strong>
          {[true, false].map((v) => (
            <label key={String(v)} style={{ display: "block" }}>
              <input
                type="radio"
                checked={lateralSmoothing === v}
                onChange={() => setLateralSmoothing(v)}
              />{" "}
              {v ? "Yes" : "No"}
            </label>
          ))}
        </div>
        <div>
          <strong>Vs range [m/s]</strong>
          <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
            <input
              type="number"
              placeholder="Min"
              value={vsMin}
              onChange={(e) => setVsMin(e.target.value === "" ? "" : Number(e.target.value))}
              style={{ width: 80 }}
            />
            <input
              type="number"
              placeholder="Max"
              value={vsMax}
              onChange={(e) => setVsMax(e.target.value === "" ? "" : Number(e.target.value))}
              style={{ width: 80 }}
            />
          </div>
        </div>
      </div>

      <h2>Inverted Vs section</h2>
      <VelocitySectionCanvas
        positions={velocitySection.positions}
        elevations={velocitySection.elevations}
        values={clippedVsGrid}
        colorLabel="Vs [m/s]"
        colormap={terrain}
        height={200}
      />
      <VelocitySectionCanvas
        positions={velocitySection.positions}
        elevations={velocitySection.elevations}
        values={velocitySection.vs_std_grid}
        height={200}
        colorLabel="Vs std [m/s]"
        colormap={afmhotR}
      />

      <button onClick={handleSaveImages} disabled={saving} style={{ marginTop: 12 }}>
        {saving ? "Saving…" : "Save images"}
      </button>
      <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
        🛈 Will save the above Vs and standard deviation sections, and the corresponding
        pseudo-sections, into the profile's output folder.
      </p>
      {saveResult && <p style={{ fontSize: 12 }}>{saveResult}</p>}

      {labels.length > 0 && (
        <>
          <h2>Observed vs predicted dispersion</h2>
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            🛈 Solid: observed. Dashed: predicted.{" "}
            {labels.map((lbl, i) => (
              <span key={lbl} style={{ color: MODE_COLORS[i % MODE_COLORS.length], marginRight: 8 }}>
                {lbl}
              </span>
            ))}
          </p>
          {(() => {
            const items = positionCurves[labels[0]] ?? [];
            const numCols = 4;
            const velocityType = labels
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
                      series={labels.map((lbl, j) => {
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

          <h2>Pseudo-section comparison</h2>
          {labels.map((lbl) => {
            const comparison = pseudoComparisons[lbl];
            const velocityType = (positionCurves[lbl] ?? [])
              .map((c) => c.velocity_type)
              .find((t) => t);
            const velocityLabel = VELOCITY_TYPE_LABELS[velocityType ?? ""] ?? "Velocity [m/s]";
            return (
              <div key={lbl} style={{ marginBottom: 16 }}>
                <h4>{lbl}</h4>
                {comparison === undefined && <p style={{ fontSize: 12, color: "var(--text-muted)" }}>Loading…</p>}
                {comparison === null && (
                  <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    Not enough positions with both a pick and an inversion result for this label.
                  </p>
                )}
                {comparison && (
                  <PseudoSectionComparisonCanvas comparison={comparison} velocityLabel={velocityLabel} />
                )}
              </div>
            );
          })}
        </>
      )}
    </>
  );
}
