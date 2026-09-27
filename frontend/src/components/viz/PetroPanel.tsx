import { API } from "../../api";
import { terrain, viridis } from "../colormaps";
import { PetroSectionCanvas, type PetroSectionData } from "../PetroSectionCanvas";
import { PseudoSectionComparisonCanvas, type PseudoSectionComparisonData } from "../PseudoSectionComparisonCanvas";
import { useZoomLink } from "../useZoom";
import { VelocitySectionCanvas } from "../VelocitySectionCanvas";
import { xmidOf } from "./format";
import { StageHead, UnitCard } from "./panel";
import { CurveFitPlot, SoilColumnView } from "./plots";
import type { Overview, PetroCard } from "./types";
import { Details, Empty, Skeleton } from "./ui";
import { useJson } from "./useJson";

// The petrophysical stage: the selected window's card (its soil column, its fit, whether the
// Silex model covers its curve, the checks), and along the line, the soils and N values, the
// rock physics' shear modulus and Vs, and the picked curves against the soil columns'.

const at = (folder: string) => encodeURIComponent(folder);
// GPa shear moduli run 0.05 to 0.5: one decimal would round them all away.
const formatGPa = (v: number) => v.toFixed(2);

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
  const xmid = selected ? xmidOf(selected) : null;
  const card = useJson<PetroCard>(xmid !== null ? `${API}/quality/petro/card/${at(folder)}/${xmid}` : null);
  const columns = (overview?.cells ?? []).filter((cell) => cell.status !== "none").length;
  const inverted = columns > 0;
  // The line's views take two columns at least.
  const line = columns >= 2;
  const section = useJson<PetroSectionData>(line ? `${API}/petro_inversion/section/${at(folder)}` : null);
  const modulus = useJson<Continuous>(line ? `${API}/petro_inversion/shear_modulus_section/${at(folder)}` : null);
  const vs = useJson<Continuous>(line ? `${API}/petro_inversion/vs_section/${at(folder)}` : null);
  const comparison = useJson<PseudoSectionComparisonData>(
    line ? `${API}/petro_inversion/pseudo_section_comparison/${at(folder)}` : null,
  );
  const pick = (position: number) => {
    const cell = overview?.cells.find((one) => one.x !== null && Math.abs(one.x - position) < 1e-6);
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
                      <CurveFitPlot curve={petroCard.curve} axis="frequency" modelled="the soil column's" />
                    ) : (
                      <Empty>No picked curve.</Empty>
                    )}
                  </div>
                )}
              </UnitCard>
            </div>
          )}
          <section className="viz-section viz-card">
            <h2
              className="viz-h2"
              data-tip={"N\nThe blow count of the Standard Penetration Test (SPT)\nThe soil's resistance to a driven sampler"}
            >
              Soil types and penetration resistance (N)
            </h2>
            {section.data ? (
              <PetroSectionCanvas section={section.data} marker={xmid ?? undefined} onPick={pick} />
            ) : !line || section.error ? (
              <Empty>Needs 2 windows with a soil column.</Empty>
            ) : (
              <Skeleton height={380} />
            )}
          </section>
          {(modulus.data || vs.data) && (
            <section className="viz-section viz-card">
              <h2
                className="viz-h2"
                data-tip={"Shear modulus and Vs\nHertz-Mindlin, forward-modelled from the soil columns\nBefore the fit to the picked curves"}
              >
                Shear modulus and Vs
              </h2>
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
            </section>
          )}
          {comparison.data && (
            <section className="viz-section viz-card">
              <h2 className="viz-h2">Picked and modelled curves</h2>
              <PseudoSectionComparisonCanvas
                comparison={comparison.data}
                velocityLabel="Phase velocity (m/s)"
                marker={xmid ?? undefined}
                onPick={pick}
              />
            </section>
          )}
        </>
      )}
    </>
  );
}
