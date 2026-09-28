import { useEffect, useMemo, useState } from "react";
import { API, type Acquisition, type Muting } from "../api";
import type { Range } from "./useZoom";
import { LineGather, type GatherData, type MuteOverlay } from "./viz/LineGather";

// The muting's preview on a computing page: a record drawn as Visualization draws one (each
// trace at its receiver along the line, the shot a star above it), what the muting and the
// trigger's shift remove veiled. A drag zooms, a double-click shows all of it.

interface RawGather {
  dt: number;
  n_samples: number;
  traces: number[][];
}

export function MuteGather({
  acquisition,
  muting,
  trigger = 0,
  file: fileProp,
  norm = "trace",
}: {
  acquisition: Acquisition;
  muting?: Muting;
  /** The trigger's shift, s: what it drops veiled, the muting measured after it; null (or
   * empty) for the previewed file's own. */
  trigger?: number | null;
  // Controlled file selection: when omitted, the component owns its own
  // selector (the config forms' preview, one file at a time).
  file?: string;
  norm?: "trace" | "global";
}) {
  const folder = acquisition.folder_path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";

  const [internalFile, setInternalFile] = useState(acquisition.files[0] ?? "");
  const file = fileProp ?? internalFile;
  const [raw, setRaw] = useState<RawGather | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The zoom along the line, back to the whole record with another shot.
  const [zoomed, setZoomed] = useState<{ file: string; x: Range | null }>({ file, x: null });
  const xZoom = zoomed.file === file ? zoomed.x : null;
  const setXZoom = (x: Range | null) => setZoomed({ file, x });

  useEffect(() => {
    if (fileProp === undefined) {
      Promise.resolve().then(() => setInternalFile(acquisition.files[0] ?? ""));
    }
  }, [acquisition, fileProp]);

  useEffect(() => {
    if (!file) return;
    Promise.resolve().then(() => setError(null));
    fetch(`${API}/gather/${encodeURIComponent(folder)}/${encodeURIComponent(file)}?norm=${norm}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: RawGather) => setRaw(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [file, folder, norm]);

  const index = acquisition.files.indexOf(file);
  const data = useMemo((): GatherData | null => {
    if (!raw) return null;
    // At most about 1,200 samples a trace: the plot shows no more.
    const stride = Math.max(1, Math.ceil(raw.n_samples / 1200));
    return {
      positions: acquisition.receiver_positions.map((position) => position[0]),
      source: acquisition.kind === "active" ? (acquisition.source_positions[index]?.[0] ?? null) : null,
      dt: raw.dt * stride,
      traces: raw.traces.map((trace) => trace.filter((_, i) => i % stride === 0)),
      excluded: [],
    };
  }, [raw, acquisition, index]);

  // Each trace's offset along the ground (x, z), as sigpipe measures it: on a slope, longer than
  // the horizontal distance.
  // The shift keeps the record's length: what a late trigger pads at the end is veiled too. A
  // bound left empty is none.
  const shift = trigger !== null && Number.isFinite(trigger) ? trigger : (acquisition.triggers?.[index] ?? 0);
  const mute = useMemo((): MuteOverlay | undefined => {
    if ((!muting && shift === 0) || !raw) return undefined;
    const source = acquisition.source_positions[index] ?? [0, 0];
    const duration = (raw.n_samples - 1) * raw.dt;
    const given = (value: number | null | undefined, none: number) =>
      value !== null && value !== undefined && Number.isFinite(value) ? value : none;
    return {
      offsets: acquisition.receiver_positions.map(([x, z]) => Math.hypot(x - source[0], z - source[1])),
      tmin: given(muting?.tmin, 0),
      tmax: Math.min(given(muting?.tmax, duration), duration - Math.max(0, shift)),
      vmin: given(muting?.vmin, 0),
      vmax: given(muting?.vmax, 0),
      width: muting?.width ?? 0,
      taper: (muting?.taper ?? 0) * raw.dt,
      shift,
    };
  }, [muting, shift, raw, acquisition, index]);

  const extent = useMemo((): Range => {
    const xs = [...(data?.positions ?? []), ...(data?.source != null ? [data.source] : [])];
    if (xs.length === 0) return [0, 1];
    const lo = Math.min(...xs);
    const hi = Math.max(...xs);
    const pad = (hi - lo) * 0.015 || 1;
    return [lo - pad, hi + pad];
  }, [data]);

  return (
    <div>
      {fileProp === undefined && (
        <label className="inline-field">
          Preview on
          <select value={file} onChange={(e) => setInternalFile(e.target.value)}>
            {acquisition.files.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
      )}
      {error && <p style={{ color: "var(--accent)" }}>Error: {error}</p>}
      {data && (
        <div style={{ marginTop: 8 }}>
          <LineGather key={file} data={data} extent={extent} xZoom={xZoom} onXZoom={setXZoom} height={420} mute={mute} />
        </div>
      )}
    </div>
  );
}
