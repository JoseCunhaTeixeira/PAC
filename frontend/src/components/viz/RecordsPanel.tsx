import { useMemo, useState } from "react";
import { API, type Acquisition } from "../../api";
import { PlotBox, Segmented } from "../kit";
import { SpectrumCanvas, type TraceSpectra } from "../SpectrumCanvas";
import type { Range } from "../useZoom";
import { LineGather, type GatherData } from "./LineGather";
import { StageHead, UnitCard } from "./panel";
import type { Overview, RecordCard } from "./types";
import { runFigures, useRunFigures } from "./runFigures";
import { Details, Empty, GateTables, PlotHead, SavedFigures, Skeleton } from "./ui";
import { useJson, useShownUnit } from "./useJson";

// The records: the selected record's card (its signal, what the checks said and redid, the
// windows that stack it) and its traces along the line, under the line plot and aligned with it.

/** A run's record as the windows used it, along the line. */
function RunGather({
  folder,
  name,
  extent,
  xZoom,
  onXZoom,
}: {
  folder: string;
  name: string;
  extent: Range;
  xZoom: Range | null;
  onXZoom: (x: Range | null) => void;
}) {
  const [norm, setNorm] = useState<"trace" | "global">("trace");
  const gather = useJson<GatherData>(
    `${API}/quality/records/gather/${encodeURIComponent(folder)}/${encodeURIComponent(name)}?norm=${norm}`,
  );
  const shown = useShownUnit(name, gather.loading);
  return (
    <PlotBox>
      <div style={{ marginTop: 16 }} className={gather.loading ? "viz-stale" : undefined}>
        <PlotHead title={`${shown} · preprocessed, as the windows used it`}>
          <Normalization value={norm} onChange={setNorm} />
        </PlotHead>
        {gather.shown ? (
          <LineGather key={shown} data={gather.shown} extent={extent} xZoom={xZoom} onXZoom={onXZoom} />
        ) : gather.error ? (
          <Empty>{gather.error}</Empty>
        ) : (
          <Skeleton height={440} />
        )}
      </div>
    </PlotBox>
  );
}

interface Spectra {
  positions: number[];
  freqs: number[];
  amplitude: number[][];
  band_hz: [number, number] | null;
}

/** The spectra a job saved beside a record or a window's stacked correlations, under their
 * gather, on its extent, zoomed apart from it; nothing when none were saved (a run from
 * before). */
export function SavedSpectrum({
  url,
  title,
  outside,
  extent,
}: {
  url: string;
  title: string;
  /** What the hover says outside the band drawn. */
  outside: string;
  extent: Range;
}) {
  const spectra = useJson<Spectra>(url);
  const shown = useShownUnit(url, spectra.loading);
  if (!spectra.shown) return null;
  return (
    <PlotBox>
      <div style={{ marginTop: 16 }} className={spectra.loading ? "viz-stale" : undefined}>
        <PlotHead title={title} />
        <SpectrumCanvas
          key={shown}
          spectra={spectra.shown}
          positions={spectra.shown.positions}
          band={spectra.shown.band_hz}
          outside={outside}
          extent={extent}
        />
      </div>
    </PlotBox>
  );
}

export function Normalization({ value, onChange }: { value: "trace" | "global"; onChange: (value: "trace" | "global") => void }) {
  return (
    <Segmented
      size="sm"
      label="Normalization"
      value={value}
      onChange={onChange}
      options={[
        { value: "trace", label: "By trace" },
        { value: "global", label: "Global" },
      ]}
    />
  );
}

interface RawGather {
  dt: number;
  n_samples: number;
  traces: number[][];
}

/** A profile's record as recorded, along the line (a profile no run has processed yet), and
 * under it its spectrum, each trace's, at its receiver, each zoomed on its own: as a run's
 * record. */
export function Gather({ profile, file }: { profile: string; file: string }) {
  const acquisition = useJson<Acquisition>(`${API}/acquisitions/${encodeURIComponent(profile)}`);
  const [norm, setNorm] = useState<"trace" | "global">("trace");
  // The zoom along the line, back to the whole record with another shot.
  const [zoomed, setZoomed] = useState<{ file: string; x: Range | null }>({ file, x: null });
  const xZoom = zoomed.file === file ? zoomed.x : null;
  const setXZoom = (x: Range | null) => setZoomed({ file, x });
  const raw = useJson<RawGather>(
    `${API}/gather/${encodeURIComponent(profile)}/${encodeURIComponent(file)}?norm=${norm}`,
  );
  const spectra = useJson<TraceSpectra>(`${API}/spectrum/${encodeURIComponent(profile)}/${encodeURIComponent(file)}`);
  // While the next record loads, the last one's traces and spectrum, with its own shot.
  const traced = useShownUnit(file, raw.loading);
  const spectral = useShownUnit(file, spectra.loading);
  const data = useMemo((): GatherData | null => {
    if (!raw.shown || !acquisition.data) return null;
    const index = acquisition.data.files.indexOf(traced);
    const source = acquisition.data.kind === "active" ? acquisition.data.source_positions[index]?.[0] ?? null : null;
    // At most about 1,200 samples a trace: the plot shows no more.
    const stride = Math.max(1, Math.ceil(raw.shown.n_samples / 1200));
    return {
      positions: acquisition.data.receiver_positions.map((position) => position[0]),
      source,
      dt: raw.shown.dt * stride,
      traces: raw.shown.traces.map((trace) => trace.filter((_, i) => i % stride === 0)),
      excluded: [],
    };
  }, [raw.shown, acquisition.data, traced]);
  const extent = useMemo((): Range => {
    const xs = [...(data?.positions ?? []), ...(data?.source !== null && data?.source !== undefined ? [data.source] : [])];
    if (xs.length === 0) return [0, 1];
    const lo = Math.min(...xs);
    const hi = Math.max(...xs);
    const pad = (hi - lo) * 0.015 || 1;
    return [lo - pad, hi + pad];
  }, [data]);
  if (acquisition.error || raw.error) return <Empty>{acquisition.error ?? raw.error}</Empty>;
  if (!data) return <Skeleton height={440} />;
  return (
    <>
      <PlotBox>
        <div className={raw.loading ? "viz-stale" : undefined}>
          <PlotHead title={`${traced} · as recorded`}>
            <Normalization value={norm} onChange={setNorm} />
          </PlotHead>
          <LineGather key={traced} data={data} extent={extent} xZoom={xZoom} onXZoom={setXZoom} />
        </div>
      </PlotBox>
      {spectra.shown && (
        <PlotBox>
          <div style={{ marginTop: 16 }} className={spectra.loading ? "viz-stale" : undefined}>
            <PlotHead title={`${spectral} · its spectrum, as recorded`} />
            <SpectrumCanvas
              key={spectral}
              spectra={spectra.shown}
              positions={data.positions}
              extent={extent}
            />
          </div>
        </PlotBox>
      )}
    </>
  );
}

export function RecordsPanel({
  folder,
  selected,
  card,
  loading,
  cardError,
  overview,
  overviewError,
  onSelect,
  extent,
  xZoom,
  onXZoom,
}: {
  folder: string;
  /** The record selected; its card, or while it loads the last one's (`loading`). */
  selected: string | null;
  card: RecordCard | null;
  loading: boolean;
  cardError: string | null;
  overview: Overview | null;
  overviewError: string | null;
  onSelect: (key: string) => void;
  extent: Range;
  xZoom: Range | null;
  onXZoom: (x: Range | null) => void;
}) {
  const place = card ? { record: card.key } : null;
  const figures = useRunFigures(folder, place);
  // A passive record (no shot): its signal and its spectrum, and its measures folded in their menu.
  const passive = card !== null && card.x === null;
  return (
    <>
      <StageHead overview={overview} error={overviewError} />
      <div className="viz-section">
        <UnitCard
          stage="records"
          bare={passive}
          card={card}
          error={cardError}
          selected={selected}
          loading={loading}
          cells={overview?.cells ?? []}
          onSelect={onSelect}
          details={
            card &&
            (passive ? (
              <GateTables gates={card.gates} />
            ) : (
              <>
                <Details gates={card.gates} attempts={card.attempts} />
                <SavedFigures figures={runFigures(folder, figures, "", place ?? {})} />
              </>
            ))
          }
        >
          {card && (
            <>
              <RunGather folder={folder} name={card.key} extent={extent} xZoom={xZoom} onXZoom={onXZoom} />
              <SavedSpectrum
                url={`${API}/quality/records/spectrum/${encodeURIComponent(folder)}/${encodeURIComponent(card.key)}`}
                title={`${card.key} · its spectrum, preprocessed`}
                outside="outside its usable band"
                extent={extent}
              />
            </>
          )}
        </UnitCard>
      </div>
    </>
  );
}
