import { useState } from "react";
import { API } from "../../api";
import { DispersionImageCanvas, type DispersionImage } from "../DispersionImageCanvas";
import { StrataIcon } from "../icons";
import { Card, Segmented } from "../kit";
import { ModeHead, PseudoSectionCanvas, type PseudoSection } from "../PseudoSectionCanvas";
import { xmidOf } from "./format";
import { StageHead, UnitCard } from "./panel";
import type { DispersionCard, Overview, WindowSources } from "./types";
import { Details, Empty, Skeleton } from "./ui";
import { useJson } from "./useJson";

// The dispersion stage: the selected window's card (the shots its image stacks, its image and
// picks, the checks), its image with the picks on it, and along the line, the pseudo-section
// of each mode's picked curves (M0, M1…), as the picking page shows them.

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
  const modes = Object.entries(labels.data ?? {});
  const windows = overview?.cells.length ?? 0;
  const [axis, setAxis] = useState<"frequency" | "wavelength">("frequency");
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
              <div className="viz-row viz-plot-head">
                <p className="viz-plot-title">
                  Dispersion image{card.data?.picked_by === "auto" ? " · picked automatically" : card.data?.picked_by === "hand" ? " · picked by hand" : ""}
                </p>
              </div>
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
      <Card
        className="viz-section"
        icon={<StrataIcon size={17} />}
        title="Pseudo-sections"
        aside={
          <Segmented
            size="sm"
            label="Vertical axis"
            value={axis}
            onChange={setAxis}
            options={[
              { value: "frequency", label: "Frequency" },
              { value: "wavelength", label: "Wavelength" },
            ]}
          />
        }
      >
        {modes.length > 0 ? (
          <div className="stack">
            {modes.map(([label, count]) => (
              <PickedSection
                key={label}
                folder={folder}
                label={label}
                count={count}
                windows={windows}
                axis={axis}
                marker={xmid ?? undefined}
                onPick={pick}
              />
            ))}
          </div>
        ) : labels.data || labels.error ? (
          <Empty>No curve picked.</Empty>
        ) : (
          <Skeleton height={300} />
        )}
      </Card>
    </>
  );
}

/** One mode's picked curves along the line: its head, then its pseudo-section. */
function PickedSection({
  folder,
  label,
  count,
  windows,
  axis,
  marker,
  onPick,
}: {
  folder: string;
  label: string;
  count: number;
  windows: number;
  axis: "frequency" | "wavelength";
  marker?: number;
  onPick: (position: number) => void;
}) {
  const section = useJson<PseudoSection>(
    count >= 2 ? `${API}/dispersion_pseudo_section/${at(folder)}/${encodeURIComponent(label)}` : null,
  );
  return (
    <div>
      <ModeHead label={label} count={count} total={windows} unit="windows" />
      {count < 2 ? (
        <p className="faint">Needs 2 picked windows.</p>
      ) : section.data ? (
        <PseudoSectionCanvas section={section.data} mode={axis} height={220} marker={marker} onPick={onPick} />
      ) : section.error ? (
        <Empty>No pseudo-section for {label}.</Empty>
      ) : (
        <Skeleton height={220} />
      )}
    </div>
  );
}
