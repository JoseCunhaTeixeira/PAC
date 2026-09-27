import { useState } from "react";
import { API } from "../../api";
import { DispersionImageCanvas, type DispersionImage } from "../DispersionImageCanvas";
import { PseudoSectionCanvas, type PseudoSection } from "../PseudoSectionCanvas";
import { xmidOf } from "./format";
import { StageHead, UnitCard } from "./panel";
import type { DispersionCard, Overview, WindowSources } from "./types";
import { Details, Empty, Skeleton } from "./ui";
import { useJson } from "./useJson";

// The dispersion stage: the selected window's card (the shots its image stacks, its image and
// picks, the checks), its image with the picks on it, and along the line, the pseudo-section
// of the picked curves.

const at = (folder: string) => encodeURIComponent(folder);

export function DispersionPanel({
  folder,
  selected,
  overview,
  overviewError,
  sources,
  onSelect,
}: {
  folder: string;
  selected: string | null;
  overview: Overview | null;
  overviewError: string | null;
  sources: WindowSources | null;
  onSelect: (key: string) => void;
}) {
  const xmid = selected ? xmidOf(selected) : null;
  const card = useJson<DispersionCard>(xmid !== null ? `${API}/quality/dispersion/card/${at(folder)}/${xmid}` : null);
  const image = useJson<DispersionImage>(xmid !== null ? `${API}/dispersion_images/${at(folder)}/${xmid}` : null);
  const labels = useJson<Record<string, number>>(`${API}/dispersion_image_labels/${at(folder)}`);
  const sectioned = Object.entries(labels.data ?? {}).filter(([, count]) => count >= 2).map(([label]) => label);
  const [label, setLabel] = useState<string | null>(null);
  const shown = label && sectioned.includes(label) ? label : sectioned[0] ?? null;
  const [axis, setAxis] = useState<"frequency" | "wavelength">("frequency");
  const section = useJson<PseudoSection>(
    shown ? `${API}/dispersion_pseudo_section/${at(folder)}/${encodeURIComponent(shown)}` : null,
  );
  const pick = (position: number) => {
    const cell = overview?.cells.find((one) => one.x !== null && Math.abs(one.x - position) < 1e-6);
    if (cell) onSelect(cell.key);
  };

  return (
    <>
      <StageHead overview={overview} error={overviewError} />
      {selected && (
        <div className="viz-section">
          <UnitCard
            stage="dispersion"
            card={card.data}
            error={card.error}
            lead={sources && sources.key === selected ? sources.sentences : []}
            cells={overview?.cells ?? []}
            onSelect={onSelect}
            details={card.data && <Details gates={card.data.gates} attempts={card.data.attempts} />}
          >
            <div style={{ marginTop: 16 }}>
              <p className="viz-plot-title">
                Dispersion image{card.data?.picked_by === "auto" ? " · picked automatically" : card.data?.picked_by === "hand" ? " · picked by hand" : ""}
              </p>
              {image.data ? (
                <DispersionImageCanvas image={image.data} />
              ) : image.error ? (
                <Empty>No dispersion image for this window.</Empty>
              ) : (
                <Skeleton height={300} />
              )}
            </div>
          </UnitCard>
        </div>
      )}
      <section className="viz-section viz-card">
        <div className="viz-row" style={{ marginBottom: 10 }}>
          <h2 className="viz-h2" style={{ margin: 0 }}>
            {shown ? `${shown} pseudo-section` : "Pseudo-section"}
          </h2>
          <div className="viz-toolbar" style={{ margin: 0 }}>
            {sectioned.length > 1 && (
              <label className="viz-field">
                Mode
                <select value={shown ?? ""} onChange={(e) => setLabel(e.target.value)}>
                  {sectioned.map((one) => (
                    <option key={one}>{one}</option>
                  ))}
                </select>
              </label>
            )}
            <div className="viz-segment" role="group" aria-label="Vertical axis">
              {(["frequency", "wavelength"] as const).map((one) => (
                <button key={one} type="button" className={axis === one ? "active" : ""} onClick={() => setAxis(one)}>
                  {one === "frequency" ? "Frequency" : "Wavelength"}
                </button>
              ))}
            </div>
          </div>
        </div>
        {section.data ? (
          <PseudoSectionCanvas
            section={section.data}
            mode={axis}
            height={260}
            marker={xmid ?? undefined}
            onPick={pick}
          />
        ) : labels.data && sectioned.length === 0 ? (
          <Empty>Needs 2 picked windows.</Empty>
        ) : (
          <Skeleton height={300} />
        )}
      </section>
    </>
  );
}
