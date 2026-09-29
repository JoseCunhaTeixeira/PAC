import { useState } from "react";
import { API } from "../../api";
import { terrain, viridis } from "../colormaps";
import { LayersIcon, OutcropIcon, StrataIcon } from "../icons";
import { Card, Segmented } from "../kit";
import { ModeHead } from "../PseudoSectionCanvas";
import { PetroSectionCanvas, type PetroSectionData } from "../PetroSectionCanvas";
import { PseudoSectionComparisonCanvas, type PseudoSectionComparisonData } from "../PseudoSectionComparisonCanvas";
import { useZoomLink } from "../useZoom";
import { VelocitySectionCanvas } from "../VelocitySectionCanvas";
import { xmidOf } from "./format";
import { StageHead, UnitCard } from "./panel";
import { CurveFitPlot, SoilColumnView, type CurveAxis } from "./plots";
import type { Overview, PetroCard } from "./types";
import { runFigures, useRunFigures } from "./runFigures";
import { Details, Empty, ErrorBox, SavedFigures, Skeleton, SmoothingSwitch } from "./ui";
import { nearestCell } from "./cells";
import { useJson } from "./useJson";

// The petrophysical stage: the selected window's card (its soil column, its fit, whether the
// Silex model covers its curve, the checks), and along the line, the soils and N values, the
// rock physics' shear modulus and Vs, and the picked curves against the soil columns'.

const at = (folder: string) => encodeURIComponent(folder);
// GPa shear moduli run 0.05 to 0.5: one decimal would round them all away.
const formatGPa = (v: number) => v.toFixed(2);

/** A section the server failed to make: not one it has none of (a 404). */
function failed(loaded: { error: string | null; missing: boolean }): boolean {
  return loaded.error !== null && !loaded.missing;
}

interface Continuous {
  positions: number[];
  elevations: number[];
  values: (number | null)[][];
}

export function PetroPanel({
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
  const zoomLink = useZoomLink(folder);
  // The picked and modelled pseudo-sections' vertical axis.
  const [comparisonAxis, setComparisonAxis] = useState<"frequency" | "wavelength">("frequency");
  // The window's curve's axis.
  const [curveAxis, setCurveAxis] = useState<CurveAxis>("frequency");
  // Each section card's own: its sections smoothed along the line, as the Vs section can be.
  const [soilSmoothing, setSoilSmoothing] = useState(false);
  const [rockSmoothing, setRockSmoothing] = useState(false);
  const xmid = selected ? xmidOf(selected) : null;
  const card = useJson<PetroCard>(xmid !== null ? `${API}/quality/petro/card/${at(folder)}/${xmid}` : null);
  const columns = (overview?.cells ?? []).filter((cell) => cell.status !== "none").length;
  const inverted = columns > 0;
  // The line's views take two columns at least.
  const line = columns >= 2;
  const saved = useRunFigures(folder);
  const section = useJson<PetroSectionData>(
    line ? `${API}/petro_inversion/section/${at(folder)}?lateral_smoothing=${soilSmoothing}` : null,
  );
  const modulus = useJson<Continuous>(
    line ? `${API}/petro_inversion/shear_modulus_section/${at(folder)}?lateral_smoothing=${rockSmoothing}` : null,
  );
  const vs = useJson<Continuous>(
    line ? `${API}/petro_inversion/vs_section/${at(folder)}?lateral_smoothing=${rockSmoothing}` : null,
  );
  const comparison = useJson<PseudoSectionComparisonData>(
    line ? `${API}/petro_inversion/pseudo_section_comparison/${at(folder)}` : null,
  );
  const pick = (position: number) => {
    const cell = nearestCell(overview?.cells ?? [], position);
    if (cell) onSelect(cell.key);
  };
  const petroCard = card.data;

  return (
    <>
      <StageHead overview={overview} error={overviewError} />
      {overview && !inverted ? (
        <section className="viz-section viz-card">
          <Empty>
            No window of this run has a soil column yet: invert it in the Petrophysical inversion page, or ask
            the assistant.
          </Empty>
        </section>
      ) : (
        <>
          {selected && (
            <div className="viz-section">
              <UnitCard
                stage="petro"
                card={petroCard}
                error={card.error}
                cells={overview?.cells ?? []}
                onSelect={onSelect}
                details={petroCard && <Details gates={petroCard.gates} attempts={petroCard.attempts} />}
              >
                {petroCard?.column && (
                  <div className="viz-plots-2">
                    <SoilColumnView column={petroCard.column} />
                    {petroCard.curve ? (
                      <CurveFitPlot
                        curve={petroCard.curve}
                        axis={curveAxis}
                        modelled="modelled (soil column)"
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
            icon={<OutcropIcon size={17} />}
            title="Soil type and penetration resistance (N) sections"
            plots
            hint={"N (SPT)\nBlow count: the soil's resistance to a driven sampler"}
            aside={<SmoothingSwitch on={soilSmoothing} onChange={setSoilSmoothing} />}
          >
            {section.data ? (
              <PetroSectionCanvas section={section.data} marker={xmid ?? undefined} onPick={pick} />
            ) : !line || section.missing ? (
              <Empty>Needs 2 windows with a soil column.</Empty>
            ) : section.error ? (
              <ErrorBox message={`Not loaded: ${section.error}`} />
            ) : (
              <Skeleton height={380} />
            )}
            <SavedFigures figures={runFigures(folder, saved, "PetroInversion_Section")} />
          </Card>
          {(modulus.data || vs.data || failed(modulus) || failed(vs)) && (
            <Card
              className="viz-section"
              icon={<LayersIcon size={17} />}
              title="Shear modulus and Vs sections"
              plots
              hint={"From the soil columns (Hertz-Mindlin)\nBefore the fit to the picked curves"}
              aside={<SmoothingSwitch on={rockSmoothing} onChange={setRockSmoothing} />}
            >
              {failed(modulus) && <ErrorBox message={`Shear modulus not loaded: ${modulus.error}`} />}
              {failed(vs) && <ErrorBox message={`Vs not loaded: ${vs.error}`} />}
              {modulus.data && (
                <VelocitySectionCanvas
                  positions={modulus.data.positions}
                  elevations={modulus.data.elevations}
                  values={modulus.data.values}
                  colorLabel="Shear modulus (GPa)"
                  colormap={viridis}
                  height={180}
                  formatValue={formatGPa}
                  link={zoomLink}
                  marker={xmid ?? undefined}
                  onPick={pick}
                />
              )}
              {vs.data && (
                <VelocitySectionCanvas
                  positions={vs.data.positions}
                  elevations={vs.data.elevations}
                  values={vs.data.values}
                  colorLabel="Vs (m/s)"
                  colormap={terrain}
                  height={180}
                  link={zoomLink}
                  marker={xmid ?? undefined}
                  onPick={pick}
                />
              )}
              <SavedFigures figures={runFigures(folder, saved, "PetroInversion_RockPhysicsSection")} />
            </Card>
          )}
          {comparison.data && (
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
              <ModeHead label="M0" count={comparison.data.positions.length} total={overview?.cells.length ?? 0} unit="windows" />
              <PseudoSectionComparisonCanvas
                comparison={comparison.data}
                velocityLabel="Phase velocity (m/s)"
                mode={comparisonAxis}
                marker={xmid ?? undefined}
                onPick={pick}
              />
              <SavedFigures figures={runFigures(folder, saved, "PetroInversion_PseudoSectionComparison")} />
            </Card>
          )}
        </>
      )}
    </>
  );
}
