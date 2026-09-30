import { useMemo, useState } from "react";
import { API } from "../../api";
import { DispersionImageCanvas, type DispersionImage } from "../DispersionImageCanvas";
import { StrataIcon } from "../icons";
import { Card, PlotBox, Segmented } from "../kit";
import { ModeHead, PseudoSectionCanvas, type PseudoSection } from "../PseudoSectionCanvas";
import type { Range } from "../useZoom";
import { num, xmidOf } from "./format";
import { LineGather, type GatherData } from "./LineGather";
import { LinePlot } from "./LinePlot";
import { vizPalette } from "./palette";
import { StageHead, UnitCard } from "./panel";
import { Normalization, SavedSpectrum } from "./RecordsPanel";
import { runFigures, useRunFigures } from "./runFigures";
import type { DispersionCard, Overview, WindowSources } from "./types";
import { Details, Empty, ErrorBox, PlotHead, SavedFigures, Skeleton } from "./ui";
import { nearestCell } from "./cells";
import { useJson, useShownUnit } from "./useJson";
import { useTheme } from "../../theme";

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
  const shown = useShownUnit(xmid, gather.loading);
  const positions = gather.shown?.positions;
  const extent = useMemo((): Range => {
    const xs = positions && positions.length > 0 ? positions : [0, 1];
    const low = Math.min(...xs);
    const high = Math.max(...xs);
    const pad = Math.max((high - low) / Math.max(xs.length - 1, 1), 0.5);
    return [low - pad, high + pad];
  }, [positions]);
  if (!gather.shown) return null;
  return (
    <PlotBox>
      <div style={{ marginTop: 16 }} className={gather.loading ? "viz-stale" : undefined}>
        <PlotHead title="Stacked correlations · the image's signal">
          <Normalization value={norm} onChange={setNorm} />
        </PlotHead>
        <LineGather key={shown} data={gather.shown} extent={extent} xZoom={xZoom} onXZoom={setXZoom} height={320} />
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

interface SelectionScores {
  threshold: number;
  flip: boolean;
  ratios: number[];
  kept: boolean[];
  segments: number;
  kept_count: number;
  flipped_count: number;
}

/** A passive window's fk segment selection, as its job saved it: each segment's f-k ratio (in
 * the order the window met them, record after record), kept, kept and flipped, or left out,
 * around ±its threshold; nothing without one. */
function WindowSelection({ folder, xmid }: { folder: string; xmid: number }) {
  const palette = vizPalette(useTheme());
  const selection = useJson<SelectionScores>(`${API}/quality/dispersion/selection/${at(folder)}/${xmid}`);
  const shown = useShownUnit(xmid, selection.loading);
  const data = selection.shown;
  if (!data) return null;
  const points = (take: (ratio: number, kept: boolean) => boolean) =>
    data.ratios.flatMap((ratio, i): [number, number][] => (take(ratio, data.kept[i]) ? [[i + 1, ratio]] : []));
  const flipped = (ratio: number) => data.flip && ratio > 0;
  const { threshold, segments } = data;
  return (
    <PlotBox>
      <div style={{ marginTop: 16 }} className={selection.loading ? "viz-stale" : undefined}>
        <PlotHead title={`fk selection · ${num(data.kept_count)} of ${num(segments)} segments kept`} />
        <LinePlot
          key={shown}
          series={[
            { label: "left out", color: palette.use.inside, points: points((_, kept) => !kept), line: false, dots: true },
            { label: "kept", color: palette.series, points: points((ratio, kept) => kept && !flipped(ratio)), line: false, dots: true },
            {
              label: "kept, flipped",
              color: palette.chains[1],
              points: points((ratio, kept) => kept && flipped(ratio)),
              line: false,
              dots: true,
            },
          ]}
          areas={[
            {
              color: palette.selected,
              polygon: [
                [0, -threshold],
                [segments + 1, -threshold],
                [segments + 1, threshold],
                [0, threshold],
              ],
            },
          ]}
          refs={[
            { axis: "y", at: threshold, label: "", color: palette.limit, dash: [4, 3] },
            { axis: "y", at: -threshold, label: "", color: palette.limit, dash: [4, 3] },
          ]}
          xLabel="Segment"
          yLabel="F-k ratio"
          xRange={[0, segments + 1]}
          yRange={[-1.05, 1.05]}
          height={260}
        />
        <div className="viz-legend-inline">
          <span style={{ color: palette.series }}>
            <i className="dot" style={{ background: palette.series }} />
            kept
          </span>
          {data.flipped_count > 0 && (
            <span style={{ color: palette.chains[1] }}>
              <i className="dot" style={{ background: palette.chains[1] }} />
              kept, flipped
            </span>
          )}
          <span>
            <i className="dot" style={{ background: palette.use.inside }} />
            left out
          </span>
          <span>
            <i className="dashed" style={{ borderTopWidth: 1 }} />
            ±{threshold}, the threshold
          </span>
        </div>
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
  const shown = card.shown;
  // The image's own zoom, back to the whole image with the next window's.
  const imaged = useShownUnit(xmid, image.loading);
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
            card={shown}
            error={card.error}
            selected={selected}
            loading={card.loading}
            lead={sources && shown && sources.key === shown.key ? sources.sentences : []}
            cells={overview?.cells ?? []}
            onSelect={onSelect}
            details={
              shown && (
                <>
                  <Details gates={shown.gates} attempts={shown.attempts} />
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
              <div style={{ marginTop: 16 }} className={image.loading ? "viz-stale" : undefined}>
                <PlotHead
                  title={`Dispersion image${shown?.picked_by === "auto" ? " · picked automatically" : shown?.picked_by === "hand" ? " · picked by hand" : ""}`}
                />
                {image.shown ? (
                  <DispersionImageCanvas key={imaged} image={image.shown} />
                ) : image.error ? (
                  <Empty>No dispersion image for this window.</Empty>
                ) : (
                  <Skeleton height={300} />
                )}
              </div>
            </PlotBox>
            {xmid !== null && <WindowGather folder={folder} xmid={xmid} />}
            {xmid !== null && <WindowSelection folder={folder} xmid={xmid} />}
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
