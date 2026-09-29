import { useMemo, useState } from "react";
import { API } from "../../api";
import { DispersionImageCanvas, type DispersionImage } from "../DispersionImageCanvas";
import { StrataIcon } from "../icons";
import { Card, PlotBox, Segmented } from "../kit";
import { ModeHead, PseudoSectionCanvas, type PseudoSection } from "../PseudoSectionCanvas";
import type { Range } from "../useZoom";
import { xmidOf } from "./format";
import { LineGather, type GatherData } from "./LineGather";
import { StageHead, UnitCard } from "./panel";
import { Normalization, SavedSpectrum } from "./RecordsPanel";
import { runFigures, useRunFigures } from "./runFigures";
import type { DispersionCard, Overview, WindowSources } from "./types";
import { Details, Empty, ErrorBox, PlotHead, SavedFigures, Skeleton } from "./ui";
import { nearestCell } from "./cells";
import { useJson } from "./useJson";

// The dispersion stage: the selected window's card (the shots its image stacks, its image and
// picks, the checks), its image with the picks on it, and along the line, the pseudo-section
// of each mode's picked curves (M0, M1…), as the picking page shows them.

const at = (folder: string) => encodeURIComponent(folder);

/** The stacked correlations a passive or passive-active window's image was made of, each trace
 * at its receiver; nothing in an active window (it stacks images, not correlations). */
function WindowGather({ folder, xmid }: { folder: string; xmid: number }) {
  const [norm, setNorm] = useState<"trace" | "global">("trace");
  const [xZoom, setXZoom] = useState<Range | null>(null);
  const gather = useJson<GatherData>(`${API}/quality/dispersion/gather/${at(folder)}/${xmid}?norm=${norm}`);
  const positions = gather.data?.positions;
  const extent = useMemo((): Range => {
    const xs = positions && positions.length > 0 ? positions : [0, 1];
    const low = Math.min(...xs);
    const high = Math.max(...xs);
    const pad = Math.max((high - low) / Math.max(xs.length - 1, 1), 0.5);
    return [low - pad, high + pad];
  }, [positions]);
  if (!gather.data) return null;
  return (
    <PlotBox>
      <div style={{ marginTop: 16 }}>
        <PlotHead title="Stacked correlations · the image's signal">
          <Normalization value={norm} onChange={setNorm} />
        </PlotHead>
        <LineGather key={xmid} data={gather.data} extent={extent} xZoom={xZoom} onXZoom={setXZoom} height={320} />
        <SavedSpectrum
          url={`${API}/quality/dispersion/spectrum/${at(folder)}/${xmid}`}
          title="Stacked correlations · their spectrum"
          outside="outside the band"
          extent={extent}
          xZoom={xZoom}
          onXZoom={setXZoom}
        />
      </div>
    </PlotBox>
  );
}

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
  const place = xmid !== null ? { xmid } : null;
  const figures = useRunFigures(folder, place);
  const runSaved = useRunFigures(folder);
  const [axis, setAxis] = useState<"frequency" | "wavelength">("frequency");
  const pick = (position: number) => {
    const cell = nearestCell(overview?.cells ?? [], position);
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
            details={
              card.data && (
                <>
                  <Details gates={card.data.gates} attempts={card.data.attempts} />
                  <SavedFigures
                    figures={runFigures(
                      folder,
                      figures,
                      ["Selection_", "Stream_", "Spectrum_", "DispersionImage_"],
                      place ?? {},
                    )}
                  />
                </>
              )
            }
          >
            <PlotBox>
              <div style={{ marginTop: 16 }}>
                <PlotHead
                  title={`Dispersion image${card.data?.picked_by === "auto" ? " · picked automatically" : card.data?.picked_by === "hand" ? " · picked by hand" : ""}`}
                />
                {image.data ? (
                  <DispersionImageCanvas image={image.data} />
                ) : image.error ? (
                  <Empty>No dispersion image for this window.</Empty>
                ) : (
                  <Skeleton height={300} />
                )}
              </div>
            </PlotBox>
            {xmid !== null && <WindowGather folder={folder} xmid={xmid} />}
          </UnitCard>
        </div>
      )}
      <Card
        className="viz-section"
        icon={<StrataIcon size={17} />}
        title="Pseudo-sections"
        plots
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
        <SavedFigures figures={runFigures(folder, runSaved, "DispersionPicking_PseudoSection")} />
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
      ) : section.missing ? (
        <Empty>No pseudo-section for {label}.</Empty>
      ) : section.error ? (
        <ErrorBox message={`Not loaded: ${section.error}`} />
      ) : (
        <Skeleton height={220} />
      )}
    </div>
  );
}
