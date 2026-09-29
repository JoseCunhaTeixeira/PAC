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
import { useJson } from "./useJson";

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
  return (
    <PlotBox>
      <div>
        <PlotHead title={`${name} · preprocessed, as the windows used it`}>
          <Normalization value={norm} onChange={setNorm} />
        </PlotHead>
        {gather.data ? (
          <LineGather key={name} data={gather.data} extent={extent} xZoom={xZoom} onXZoom={onXZoom} />
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
 * gather and along the line with it; nothing when none were saved (a run from before). */
export function SavedSpectrum({
  url,
  title,
  outside,
  extent,
  xZoom,
  onXZoom,
}: {
  url: string;
  title: string;
  /** What the hover says outside the band drawn. */
  outside: string;
  extent: Range;
  xZoom: Range | null;
  onXZoom: (x: Range | null) => void;
}) {
  const spectra = useJson<Spectra>(url);
  if (!spectra.data) return null;
  return (
    <PlotBox>
      <div style={{ marginTop: 16 }}>
        <PlotHead title={title} />
        <SpectrumCanvas
          key={url}
          spectra={spectra.data}
          positions={spectra.data.positions}
          band={spectra.data.band_hz}
          outside={outside}
          extent={extent}
          xZoom={xZoom}
          onXZoom={onXZoom}
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
 * under it its spectrum, each trace's, at its receiver, zoomed with it: as a run's record. */
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
  const data = useMemo((): GatherData | null => {
    if (!raw.data || !acquisition.data) return null;
    const index = acquisition.data.files.indexOf(file);
    const source = acquisition.data.kind === "active" ? acquisition.data.source_positions[index]?.[0] ?? null : null;
    // At most about 1,200 samples a trace: the plot shows no more.
    const stride = Math.max(1, Math.ceil(raw.data.n_samples / 1200));
    return {
      positions: acquisition.data.receiver_positions.map((position) => position[0]),
      source,
      dt: raw.data.dt * stride,
      traces: raw.data.traces.map((trace) => trace.filter((_, i) => i % stride === 0)),
      excluded: [],
    };
  }, [raw.data, acquisition.data, file]);
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
        <div>
          <PlotHead title={`${file} · as recorded`}>
            <Normalization value={norm} onChange={setNorm} />
          </PlotHead>
          <LineGather key={file} data={data} extent={extent} xZoom={xZoom} onXZoom={setXZoom} />
        </div>
      </PlotBox>
      {spectra.data && (
        <PlotBox>
          <div style={{ marginTop: 16 }}>
            <PlotHead title={`${file} · its spectrum, as recorded`} />
            <SpectrumCanvas
              key={file}
              spectra={spectra.data}
              positions={data.positions}
              extent={extent}
              xZoom={xZoom}
              onXZoom={setXZoom}
            />
          </div>
        </PlotBox>
      )}
    </>
  );
}

export function RecordsPanel({
  folder,
  card,
  cardError,
  overview,
  overviewError,
  onSelect,
  extent,
  xZoom,
  onXZoom,
}: {
  folder: string;
  card: RecordCard | null;
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
            <div style={{ marginTop: 16 }}>
              <RunGather folder={folder} name={card.key} extent={extent} xZoom={xZoom} onXZoom={onXZoom} />
              <SavedSpectrum
                url={`${API}/quality/records/spectrum/${encodeURIComponent(folder)}/${encodeURIComponent(card.key)}`}
                title={`${card.key} · its spectrum, preprocessed`}
                outside="outside its usable band"
                extent={extent}
                xZoom={xZoom}
                onXZoom={onXZoom}
              />
            </div>
          )}
        </UnitCard>
      </div>
    </>
  );
}
