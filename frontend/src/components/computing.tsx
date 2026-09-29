import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { API, type Acquisition, type Dispersion, type Masw } from "../api";
import { bound } from "../builders";
import type { FilteringState, MutingState, StackingState } from "../presets";
import { GeometryPlot } from "./GeometryPlot";
import { ArrowRightIcon, CpuIcon, FilterIcon, FolderIcon, InfoIcon, RulerIcon, ScissorsIcon, SpectrumIcon, StackIcon } from "./icons";
import { BoxTools, Callout, Card, Empty, Fields, NumberField, NumberInput, Page, PlotBox, Segmented, SelectField, Stat, Stats } from "./kit";
import { runningJob } from "./jobs";
import { boundsOf, tipOf, written } from "./numbers";
import { dataEnd, distinct, sampleOf, shortestRecord } from "./records";
import { useStoredState } from "./stored";
import { MuteGather } from "./MuteGather";
import type { ArtKind } from "./PageArt";
import type { Job } from "./RunPanel";

// What the three computing pages share: the page and its profile, the acquisition's summary,
// the MASW windows with their geometry, the stages' rows, and the links once a run is done.

export type FormProps = {
  acquisition: Acquisition;
  profile: string;
  /** Whether the form's job runs: the page freezes its profile meanwhile. */
  onRunning?: (running: boolean) => void;
};

export function ComputingPage({
  mode,
  title,
  subtitle,
  icon,
  art,
  needsSources,
  Form,
}: {
  /** The processing mode its runs use: a job of it still running reopens its profile. */
  mode: "active" | "passive" | "passive-active";
  title: string;
  subtitle: string;
  icon: ReactNode;
  art: ArtKind;
  /** Active modes: the profile must give its shots' positions. */
  needsSources: boolean;
  Form: (props: FormProps) => ReactNode;
}) {
  const [folders, setFolders] = useState<string[]>([]);
  // The profile, kept when the page is left.
  const [selected, setSelected] = useStoredState(`pac.computing.${mode}.profile`, "");
  const [running, setRunning] = useState(false); // the profile frozen while its job runs
  const [acquisition, setAcquisition] = useState<Acquisition | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingFolders, setLoadingFolders] = useState(true);

  // Back on the page while its job runs: its profile, so that its run bar shows the job.
  useEffect(() => {
    let cancelled = false;
    runningJob("processing", mode).then((job) => {
      if (!cancelled && job) setSelected(job.target);
    });
    return () => {
      cancelled = true;
    };
  }, [mode, setSelected]);

  useEffect(() => {
    fetch(`${API}/input_folders`)
      .then((res) => res.json())
      .then((data: string[]) => setFolders(data))
      .catch((err) => setError(String(err)))
      .finally(() => setLoadingFolders(false));
  }, []);

  useEffect(() => {
    if (!selected) {
      Promise.resolve().then(() => setAcquisition(null));
      return;
    }
    Promise.resolve().then(() => {
      setLoading(true);
      setError(null);
    });
    fetch(`${API}/acquisitions/${selected}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: Acquisition) => setAcquisition(data))
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        setAcquisition(null);
      })
      .finally(() => setLoading(false));
  }, [selected]);

  const missingSources = needsSources && acquisition !== null && acquisition.source_positions.length === 0;

  return (
    <Page
      icon={icon}
      title={title}
      subtitle={subtitle}
      art={art}
      actions={
        <SelectField
          label="Profile"
          value={selected}
          onChange={setSelected}
          icon={<FolderIcon size={15} />}
          disabled={running}
        >
          <option value="">Choose a profile…</option>
          {folders.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </SelectField>
      }
    >
      {!loadingFolders && folders.length === 0 && (
        <Callout tone="warn" title="No profile found">
          Put each profile's records in its own folder of the input folder.
        </Callout>
      )}
      {error && <Callout tone="error" title="The profile could not be read">{error}</Callout>}
      {!selected && folders.length > 0 && (
        <Empty icon={<FolderIcon size={22} />} title="Choose a profile" />
      )}
      {loading && <p className="muted">Reading the profile…</p>}
      {missingSources && (
        <Callout tone="error" title="No shot positions">
          This mode needs source_positions.yaml in the profile's folder.
        </Callout>
      )}
      {acquisition && acquisition.files.length === 0 && (
        <Callout tone="warn" title="Empty profile">
          The folder holds no record.
        </Callout>
      )}
      {acquisition && acquisition.files.length > 0 && !missingSources && (
        <div className="stack">
          <AcquisitionCard acquisition={acquisition} showSources={needsSources} />
          <Form key={selected} acquisition={acquisition} profile={selected} onRunning={setRunning} />
        </div>
      )}
    </Page>
  );
}

function spacingOf(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b);
  const gaps = sorted.slice(1).map((x, i) => x - sorted[i]).sort((a, b) => a - b);
  return gaps.length ? gaps[Math.floor(gaps.length / 2)] : 0;
}

function AcquisitionCard({ acquisition, showSources }: { acquisition: Acquisition; showSources: boolean }) {
  const receivers = acquisition.receiver_positions.map((p) => p[0]);
  const spacing = spacingOf(receivers);
  const length = receivers.length ? Math.max(...receivers) - Math.min(...receivers) : 0;
  // Every length the records have; the shortest sets the settings' limits (see records.ts).
  const durations = distinct(acquisition.durations);
  const rates = acquisition.sampling_frequencies.filter((f) => Number.isFinite(f));
  const rate = rates.length ? rates[0] : 0;
  return (
    <Card title="Acquisition" icon={<FolderIcon size={17} />}>
      <Stats>
        <Stat label="Records" value={acquisition.files.length} />
        <Stat label="Receivers" value={receivers.length} sub={`every ${+spacing.toFixed(3)} m`} />
        <Stat label="Line length" value={`${+length.toFixed(2)} m`} />
        <Stat
          label="Duration"
          value={`${durations.length ? durations.join(", ") : 0} s`}
          sub={durations.length > 1 ? "the shortest sets the limits" : undefined}
        />
        <Stat label="Sampling" value={`${+rate.toFixed(1)} Hz`} sub={`Nyquist ${+(rate / 2).toFixed(1)} Hz`} />
      </Stats>
      <details className="fold" style={{ marginTop: 14 }}>
        <summary>Every record</summary>
        <div className="table-wrap" style={{ marginTop: 10, maxHeight: 320, overflowY: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th className="num">Duration (s)</th>
                <th className="num">Sampling (Hz)</th>
                {showSources && <th className="num">Shot x, z (m)</th>}
              </tr>
            </thead>
            <tbody>
              {acquisition.files.map((file, i) => (
                <tr key={file}>
                  <td className="mono">{file}</td>
                  <td className="num">{acquisition.durations[i]?.toFixed(2) ?? "—"}</td>
                  <td className="num">{acquisition.sampling_frequencies[i]?.toFixed(1) ?? "—"}</td>
                  {showSources && (
                    <td className="num">
                      {acquisition.source_positions[i]
                        ? `${acquisition.source_positions[i][0]}, ${acquisition.source_positions[i][1]}`
                        : "—"}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </Card>
  );
}

/** A stage's row: its name and use, its method on the right, its values under. */
export function Row({
  icon,
  title,
  hint,
  control,
  children,
}: {
  icon?: ReactNode;
  title: string;
  hint?: string;
  control?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="row">
      <div className="row-head">
        <div className="row-title" data-tip={hint}>
          {icon && <span>{icon}</span>}
          <strong>{title}</strong>
        </div>
        {control}
      </div>
      {children && <div className="row-body">{children}</div>}
    </div>
  );
}

/** The MASW windows: their length, step and the shots' distances, and the geometry they make. */
export function WindowsCard({
  acquisition,
  masw,
  setMasw,
  onCount,
  showSources,
  unit,
}: {
  acquisition: Acquisition;
  masw: Masw;
  setMasw: (masw: Masw) => void;
  onCount: (n: number) => void;
  showSources: boolean;
  unit?: string;
}) {
  const receivers = acquisition.receiver_positions.length;
  const spacing = useMemo(() => spacingOf(acquisition.receiver_positions.map((p) => p[0])), [acquisition]);
  return (
    <Card step={1} title="MASW windows">
      <PlotBox>
        <Fields>
          <NumberField
            label="Length"
            unit="receivers"
            value={masw.length}
            onChange={(v) => setMasw({ ...masw, length: v })}
            min={3}
            max={receivers}
            whole
            hint={Number.isFinite(masw.length) ? `${+((masw.length - 1) * spacing).toFixed(2)} m` : undefined}
          />
          <NumberField
            label="Step"
            unit="receivers"
            value={masw.step}
            onChange={(v) => setMasw({ ...masw, step: v })}
            min={1}
            max={receivers}
            whole
            hint={Number.isFinite(masw.step) ? `${+(masw.step * spacing).toFixed(2)} m` : undefined}
          />
          {/* The shots' distances: a passive line has no shot. */}
          {showSources && (
            <>
              <NumberField
                label="Nearest shot"
                title="From the window's middle"
                unit="m"
                value={masw.distance_min ?? Number.NaN}
                optional="0"
                onChange={(v) => setMasw({ ...masw, distance_min: v })}
                min={0}
              />
              <NumberField
                label="Farthest shot"
                title="From the window's middle"
                unit="m"
                value={masw.distance_max ?? Number.NaN}
                optional="∞"
                onChange={(v) => setMasw({ ...masw, distance_max: v })}
                gt={bound(masw.distance_min) ?? 0}
              />
            </>
          )}
          {/* The line's tools at the fields' right end, right above its plot. */}
          <div className="fields-tools">
            <BoxTools />
          </div>
        </Fields>
        <div style={{ marginTop: 18 }}>
          <GeometryPlot acquisition={acquisition} masw={masw} onCount={onCount} showSources={showSources} unit={unit} />
        </div>
      </PlotBox>
    </Card>
  );
}

/** Each shot cut: muting on, the trigger's shift first (left empty, each record's own, from its
 * file), then a time window and the arrivals between two velocities, each bound left empty when
 * none, the shot's pulse kept after the slowest; the record previewed, what the muting removes
 * veiled. Off, none of it applies. Each number kept to what the records hold, as sigpipe checks
 * it: the shortest record, the data that ends first once moved (see records.ts). A passive line
 * has no muting (the user, 2026-09-28): no shot to count a velocity from. */
export function MutingRow({
  acquisition,
  muting,
  setMuting,
  gather = true,
  trigger,
}: {
  acquisition: Acquisition;
  muting: MutingState;
  setMuting: (muting: MutingState) => void;
  gather?: boolean;
  trigger?: { t0: number | null; setT0: (t0: number) => void };
}) {
  // What the files' headers say of the trigger, optional (a file may not say it: no shift for
  // it): one value, in all of them or some, each value when they differ (left empty, each record
  // moved by its own), or nothing.
  const known = (acquisition.triggers ?? []).filter((t): t is number => t !== null);
  const said = distinct(known);
  const where = known.length === acquisition.files.length ? "the files' headers" : `${known.length} of ${acquisition.files.length} files' headers`;
  const files =
    said.length === 0
      ? "none in the files' headers"
      : said.length === 1
        ? `${said[0]} s in ${where}`
        : `${said.join(", ")} s in ${where}: each its own`;
  // The limits: the records' data, once moved by the trigger, end first at `data`; the trigger
  // leaves the pulse at the shot (the signal width, one sample at least), a window starts
  // before the data's end.
  const record = shortestRecord(acquisition);
  const sample = sampleOf(acquisition);
  const data = dataEnd(acquisition, trigger ? trigger.t0 : 0);
  const width = bound(muting.width) ?? sample;
  const tmin = bound(muting.tmin) ?? 0;
  const vmin = bound(muting.vmin) ?? 0;
  return (
    <Row
      icon={<ScissorsIcon size={16} />}
      title="Signal muting"
      hint={
        trigger
          ? "The trigger's time shift, then a time window, or arrivals between two velocities."
          : "Keeps a time window, or arrivals between two velocities."
      }
      control={
        <Segmented
          size="sm"
          label="Muting"
          value={muting.method}
          onChange={(method) => setMuting({ ...muting, method })}
          options={[
            { value: "none", label: "Off" },
            { value: "mute", label: "Mute" },
          ]}
          after={
            // What each setting is, on a sketch: large, on hover; with the muting on only.
            muting.method === "mute" && (
              <span className="sketch-info" tabIndex={0} aria-label="What each muting setting is">
                <InfoIcon size={15} />
                <span className="sketch-pop" role="tooltip">
                  <MuteSketch scale={1.7} />
                </span>
              </span>
            )
          }
        />
      }
    >
      {muting.method === "mute" && (
        <>
          <Fields min={100} fit>
            {trigger && (
              <NumberField
                label="Trigger delay"
                unit="s"
                value={trigger.t0 ?? Number.NaN}
                // Unfilled at start: 0, no shift; the files' values in its hint.
                optional="0"
                hint={files}
                onChange={trigger.setT0}
                min={0}
                max={record - width}
                step={0.001}
              />
            )}
            <NumberField
              label="From"
              unit="s"
              value={muting.tmin ?? Number.NaN}
              optional="none"
              onChange={(v) => setMuting({ ...muting, tmin: v })}
              min={0}
              lt={data}
              step={0.1}
            />
            <NumberField
              label="To"
              unit="s"
              value={muting.tmax ?? Number.NaN}
              optional="none"
              onChange={(v) => setMuting({ ...muting, tmax: v })}
              gt={tmin}
              max={record}
              step={0.1}
            />
            <NumberField label="Slowest" unit="m/s" value={muting.vmin ?? Number.NaN} optional="none" onChange={(v) => setMuting({ ...muting, vmin: v })} min={0} />
            <NumberField
              label="Fastest"
              unit="m/s"
              value={muting.vmax ?? Number.NaN}
              optional="none"
              onChange={(v) => setMuting({ ...muting, vmax: v })}
              gt={vmin}
            />
            {/* Empty: one sample (the least, the default). */}
            <NumberField
              label="Signal width"
              unit="s"
              value={muting.width !== null && Math.abs(muting.width - sample) > 1e-12 ? muting.width : Number.NaN}
              optional={written(sample)}
              onChange={(v) => setMuting({ ...muting, width: v })}
              min={sample}
              max={data}
              step={0.01}
            />
            <NumberField
              label="Taper"
              unit="samples"
              value={muting.taper ? muting.taper : Number.NaN}
              optional="none"
              onChange={(v) => setMuting({ ...muting, taper: v })}
              min={0}
              whole
            />
          </Fields>
          {gather && (
            <MuteGather
              acquisition={acquisition}
              muting={muting}
              trigger={trigger ? (bound(trigger.t0) ?? 0) : 0}
            />
          )}
        </>
      )}
    </Row>
  );
}

// What each muting setting is, on a sketch of a record, always the same (the preview shows the
// record itself): time down from the recording's start, distance from the shot along; the signal
// kept shaded, what is removed veiled, the tapers hatched. Each setting named as its field: at the
// shot, the trigger delay and the window's width on the left; the times From and To on the right,
// spans from the trigger limit (the shot's time, dotted), as the muting measures them; the
// velocities and the taper on their lines.
const SKETCH = {
  w: 440,
  h: 166,
  left: 122, // the shot's distance
  right: 360,
  start: 10, // the recording's start
  shot: 28, // the trigger limit: after the trigger delay
  from: 36, // after the trigger limit
  to: 116,
  bottom: 146, // the record's end
  fast: 0.08, // down a unit along: the fastest arrival
  slow: 0.26, // the slowest
  width: 32, // the window at the shot: the fastest from its top, the slowest from its bottom
  taper: 9,
} as const;

export function MuteSketch({ scale = 1 }: { scale?: number }) {
  const g = SKETCH;
  const fastAt = (x: number) => g.shot + (x - g.left) * g.fast;
  // The slowest from the width's bottom border: what is kept after it at every distance.
  const slowAt = (x: number) => g.shot + g.width + (x - g.left) * g.slow;
  const first = (x: number) => Math.max(fastAt(x), g.from);
  const last = (x: number) => Math.min(slowAt(x), g.to);
  const xs = Array.from({ length: 49 }, (_, i) => g.left + ((g.right - g.left) * i) / 48);
  const points = (edge: (x: number) => number, back: (x: number) => number) =>
    [...xs.map((x) => [x, edge(x)]), ...[...xs].reverse().map((x) => [x, back(x)])]
      .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
      .join(" ");
  const text = (x: number, y: number, words: string, anchor: "start" | "end" | "middle" = "start", slope = 0) => (
    <text x={x} y={y} textAnchor={anchor} transform={slope ? `rotate(${(Math.atan(slope) * 180) / Math.PI} ${x} ${y})` : undefined}>
      {words}
    </text>
  );
  // A time span as a bracket: on the left, opening right; on the right, opening left.
  const bracket = (x: number, y0: number, y1: number, side: "left" | "right") => (
    <path className="bracket" d={side === "left" ? `M${x},${y0}h-4V${y1}h4` : `M${x},${y0}h4V${y1}h-4`} />
  );
  const along = (at: number) => g.left + (g.right - g.left) * at;
  return (
    <svg className="mute-sketch" viewBox={`0 0 ${g.w} ${g.h}`} width={g.w * scale} height={g.h * scale} role="img" aria-label="What each muting setting is">
      <defs>
        <pattern id="mute-sketch-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect className="hatch-ground" width="4" height="4" />
          <line className="hatch" x1="0" y1="0" x2="0" y2="4" />
        </pattern>
      </defs>
      <rect className="removed" x={g.left} y={g.start} width={g.right - g.left} height={g.bottom - g.start} />
      <polygon className="kept" points={points(first, last)} />
      <polygon className="taper" points={points(first, (x) => Math.min(first(x) + g.taper, last(x)))} />
      <polygon className="taper" points={points((x) => Math.max(last(x) - g.taper, first(x)), last)} />
      {/* The trigger limit: the shot's time, where From and To count from. */}
      <line className="limit" x1={g.left} y1={g.shot} x2={g.right + 48} y2={g.shot} />
      <line className="edge" x1={g.left} y1={g.from} x2={g.right} y2={g.from} />
      <line className="edge" x1={g.left} y1={g.to} x2={g.right} y2={g.to} />
      {/* From the width's borders at the shot: the fastest starts what is kept, the slowest ends it. */}
      <line className="arrival" x1={g.left} y1={g.shot} x2={g.right} y2={fastAt(g.right)} />
      <line className="arrival" x1={g.left} y1={g.shot + g.width} x2={g.right} y2={slowAt(g.right)} />
      <rect className="frame" x={g.left} y={g.start} width={g.right - g.left} height={g.bottom - g.start} />
      {bracket(g.left - 5, g.start, g.shot, "left")}
      {text(g.left - 13, (g.start + g.shot) / 2 + 3.5, "Trigger delay", "end")}
      {bracket(g.left - 5, g.shot, g.shot + g.width, "left")}
      {text(g.left - 13, g.shot + g.width / 2 + 3.5, "Signal width", "end")}
      {bracket(g.right + 5, g.shot, g.from, "right")}
      {text(g.right + 14, (g.shot + g.from) / 2 + 3.5, "From")}
      {/* Past From's label, so that its line crosses nothing. */}
      {bracket(g.right + 44, g.shot, g.to, "right")}
      {text(g.right + 53, (g.shot + g.to) / 2 + 3.5, "To")}
      {/* The velocities and the taper, on their lines. */}
      {text(along(0.34), first(along(0.34)) + g.taper + 10, "Fastest", "middle", g.fast)}
      {text(along(0.3), slowAt(along(0.3)) + 13, "Slowest", "middle", g.slow)}
      {text(along(0.8), first(along(0.8)) + g.taper + 11, "Taper", "middle", g.fast)}
      {text(g.left - 13, g.bottom, "time ↓", "end")}
      {text(g.right, g.bottom + 14, "distance from the shot →", "end")}
    </svg>
  );
}

export function FilteringRow({
  filtering,
  setFiltering,
  nyquist,
}: {
  filtering: FilteringState;
  setFiltering: (filtering: FilteringState) => void;
  nyquist: number;
}) {
  return (
    <Row
      icon={<FilterIcon size={16} />}
      title="Spectral filtering"
      hint="A band-pass before the dispersion images."
      control={
        <Segmented
          size="sm"
          label="Filtering"
          value={filtering.method}
          onChange={(method) => setFiltering({ ...filtering, method })}
          options={[
            { value: "none", label: "Off" },
            { value: "iir", label: "IIR" },
          ]}
        />
      }
    >
      {filtering.method === "iir" && (
        <Fields>
          <NumberField label="Low cut" unit="Hz" value={filtering.fmin} onChange={(v) => setFiltering({ ...filtering, fmin: v })} min={0} max={nyquist} step={5} />
          {/* Below Nyquist, strictly: the filter's design refuses it. */}
          <NumberField
            label="High cut"
            unit="Hz"
            value={filtering.fmax}
            onChange={(v) => setFiltering({ ...filtering, fmax: v })}
            gt={filtering.fmin}
            lt={nyquist}
            step={5}
          />
          <NumberField label="Order" value={filtering.order} onChange={(v) => setFiltering({ ...filtering, order: v })} min={4} step={1} whole />
        </Fields>
      )}
    </Row>
  );
}

export function StackingRow({
  title,
  hint,
  stacking,
  setStacking,
  phaseWeighted = true,
}: {
  title: string;
  hint: string;
  stacking: StackingState;
  setStacking: (stacking: StackingState) => void;
  phaseWeighted?: boolean;
}) {
  return (
    <Row
      icon={<StackIcon size={16} />}
      title={title}
      hint={hint}
      control={
        <Segmented
          size="sm"
          label={title}
          value={stacking.method}
          onChange={(method) => setStacking({ ...stacking, method })}
          options={[
            { value: "linear", label: "Linear" },
            ...(phaseWeighted ? [{ value: "phase_weighted", label: "Phase-weighted" }] : []),
            { value: "root", label: "Root" },
          ]}
        />
      }
    >
      {stacking.method === "phase_weighted" && (
        <Fields>
          <NumberField label="Power ν" value={stacking.nu} onChange={(v) => setStacking({ ...stacking, nu: v })} min={0} whole />
        </Fields>
      )}
      {stacking.method === "root" && (
        <Fields>
          <NumberField label="Root n" value={stacking.n} onChange={(v) => setStacking({ ...stacking, n: v })} min={1} whole />
        </Fields>
      )}
    </Row>
  );
}

export function DispersionRows({
  dispersion,
  setDispersion,
  nyquist,
  kept,
}: {
  dispersion: Dispersion;
  setDispersion: (dispersion: Dispersion) => void;
  nyquist: number;
  /** The band the filter (and, passive, the whitening) keeps: the image overlaps it, as sigpipe
   * checks (outside, noise alone). */
  kept?: [number, number];
}) {
  const [keptLow, keptHigh] = kept ?? [0, nyquist];
  return (
    <>
      <Row icon={<SpectrumIcon size={16} />} title="Frequencies">
        <Fields>
          <NumberField label="From" unit="Hz" value={dispersion.fmin} onChange={(v) => setDispersion({ ...dispersion, fmin: v })} min={0} lt={Math.min(nyquist, keptHigh)} />
          <NumberField
            label="To"
            unit="Hz"
            value={dispersion.fmax}
            onChange={(v) => setDispersion({ ...dispersion, fmax: v })}
            gt={Math.max(dispersion.fmin, keptLow)}
            max={nyquist}
          />
        </Fields>
      </Row>
      <Row icon={<RulerIcon size={16} />} title="Phase velocities">
        <Fields>
          <NumberField label="From" unit="m/s" value={dispersion.vmin} onChange={(v) => setDispersion({ ...dispersion, vmin: v })} min={1} />
          <NumberField
            label="To"
            unit="m/s"
            value={dispersion.vmax}
            onChange={(v) => setDispersion({ ...dispersion, vmax: v })}
            gt={dispersion.vmin}
          />
          <NumberField label="Steps" value={dispersion.nv} onChange={(v) => setDispersion({ ...dispersion, nv: v })} min={1000} whole />
        </Fields>
      </Row>
    </>
  );
}

/** The workers of a run bar, alike on every page: its icon, its number, its bounds on hover. */
export function WorkersField({
  workers,
  setWorkers,
  maxWorkers,
  tip = "Windows computed in parallel",
}: {
  workers: number;
  setWorkers: (n: number) => void;
  maxWorkers: number;
  tip?: string;
}) {
  // Its icon and its word say it as its number does: what it is, then its bounds.
  return (
    <label className="workers" data-tip={tipOf(tip, boundsOf({ min: 1, max: maxWorkers }))}>
      <CpuIcon size={15} />
      <NumberInput min={1} max={maxWorkers} value={workers} onChange={setWorkers} data-tip={tip} whole />
      <span>workers</span>
    </label>
  );
}

/** Where to go once a processing run is done. */
export function NextSteps({ job }: { job: Job }) {
  if (!job.run) return null;
  return (
    <span className="run-next">
      <Link to="/dispersion_picking">
        Pick the curves <ArrowRightIcon size={13} />
      </Link>
      <Link to={`/visualization?run=${encodeURIComponent(job.run)}`}>
        Review <ArrowRightIcon size={13} />
      </Link>
    </span>
  );
}
