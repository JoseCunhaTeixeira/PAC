import { useEffect, useState } from "react";
import { API } from "./api";
import { terrain, viridis } from "./components/colormaps";
import { DispersionCurveCanvas } from "./components/DispersionCurveCanvas";
import { PetroSectionCanvas, type PetroSectionData } from "./components/PetroSectionCanvas";
import {
  PseudoSectionComparisonCanvas,
  type PseudoSectionComparisonData,
} from "./components/PseudoSectionComparisonCanvas";
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

export function VisualizationPetroInversion({ folder }: { folder: string }) {
  const [xmids, setXmids] = useState<number[]>([]);
  const [hasResults, setHasResults] = useState(false);
  const [labelCounts, setLabelCounts] = useState<Record<string, number>>({});
  const [petroSection, setPetroSection] = useState<PetroSectionData | null>(null);
  const [shearModulusSection, setShearModulusSection] = useState<ContinuousSection | null>(null);
  const [vsSection, setVsSection] = useState<ContinuousSection | null>(null);
  const [positionCurves, setPositionCurves] = useState<PositionCurves[]>([]);
  const [pseudoComparison, setPseudoComparison] = useState<PseudoSectionComparisonData | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  // Starts true (not false) so the first render of a freshly-selected folder
  // shows "Loading…" instead of flashing "No positions found" for one frame
  // before the mount effect below gets a chance to run.
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.resolve().then(() => {
      setXmids([]);
      setHasResults(false);
      setLabelCounts({});
      setPetroSection(null);
      setShearModulusSection(null);
      setVsSection(null);
      setPositionCurves([]);
      setPseudoComparison(null);
      setError(null);
      setLoading(true);
    });

    const xmidsDone = fetch(`${API}/xmids/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: number[]) => setXmids(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));

    const statusDone = fetch(`${API}/petro_inversion/status/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: { xmid: number; has_result: boolean }[]) =>
        setHasResults(data.some((s) => s.has_result)),
      )
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));

    Promise.allSettled([xmidsDone, statusDone]).finally(() => setLoading(false));

    fetch(`${API}/dispersion_image_labels/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: Record<string, number>) => setLabelCounts(data))
      .catch(() => setLabelCounts({}));

    fetch(`${API}/petro_inversion/section/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: PetroSectionData) => setPetroSection(data))
      // Best-effort: a folder with too few inverted positions just skips the
      // section instead of blocking the rest of the page.
      .catch(() => setPetroSection(null));

    fetch(`${API}/petro_inversion/shear_modulus_section/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: ContinuousSection) => setShearModulusSection(data))
      .catch(() => setShearModulusSection(null));

    fetch(`${API}/petro_inversion/vs_section/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: ContinuousSection) => setVsSection(data))
      .catch(() => setVsSection(null));

    fetch(`${API}/petro_inversion/curves/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: PositionCurves[]) => setPositionCurves(data))
      .catch(() => setPositionCurves([]));

    fetch(`${API}/petro_inversion/pseudo_section_comparison/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: PseudoSectionComparisonData) => setPseudoComparison(data))
      .catch(() => setPseudoComparison(null));
  }, [folder]);

  if (error) return <p style={{ color: "var(--accent)" }}>Error: {error}</p>;
  if (loading) return <p>Loading…</p>;
  if (xmids.length === 0) return <p>❌ No positions found.</p>;
  if (!hasResults) return <p>❌ No petrophysical inversion results found.</p>;

  const velocityType = positionCurves.map((c) => c.velocity_type).find((t) => t);
  const yAxisLabel = VELOCITY_TYPE_LABELS[velocityType ?? ""] ?? "Velocity [m/s]";
  const modeLabel = Object.keys(labelCounts).find(isFundamentalModeLabel) ?? "M0";

  return (
    <>
      {petroSection ? (
        <>
          <h2>Petrophysical section</h2>
          <PetroSectionCanvas section={petroSection} />
        </>
      ) : (
        <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
          🛈 At least 2 inverted positions are required to build a section.
        </p>
      )}

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
    </>
  );
}
