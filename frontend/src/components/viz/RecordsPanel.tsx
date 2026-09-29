import { useMemo, useState } from "react";
import { API, type Acquisition } from "../../api";
import { PlotBox, Segmented } from "../kit";
import type { Range } from "../useZoom";
import { LineGather, type GatherData } from "./LineGather";
import { StageHead, UnitCard } from "./panel";
import type { Overview, RecordCard } from "./types";
import { Details, Empty, PlotHead, Skeleton } from "./ui";
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

function Normalization({ value, onChange }: { value: "trace" | "global"; onChange: (value: "trace" | "global") => void }) {
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

/** A profile's record as recorded, along the line (a profile no run has processed yet). */
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
    <PlotBox>
      <div>
        <PlotHead>
          <Normalization value={norm} onChange={setNorm} />
        </PlotHead>
        <LineGather key={file} data={data} extent={extent} xZoom={xZoom} onXZoom={setXZoom} />
      </div>
    </PlotBox>
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
  return (
    <>
      <StageHead overview={overview} error={overviewError} />
      <div className="viz-section">
        <UnitCard
          stage="records"
          card={card}
          error={cardError}
          cells={overview?.cells ?? []}
          onSelect={onSelect}
          details={card && <Details gates={card.gates} attempts={card.attempts} />}
        >
          {card && (
            <div style={{ marginTop: 16 }}>
              <RunGather folder={folder} name={card.key} extent={extent} xZoom={xZoom} onXZoom={onXZoom} />
            </div>
          )}
        </UnitCard>
      </div>
    </>
  );
}
