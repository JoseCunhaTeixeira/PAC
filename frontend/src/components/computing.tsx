import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { API, type Acquisition, type Dispersion, type Masw } from "../api";
import type { FilteringState, MutingState, StackingState } from "../presets";
import { GeometryPlot } from "./GeometryPlot";
import { ArrowRightIcon, CpuIcon, FilterIcon, FolderIcon, RulerIcon, ScissorsIcon, SpectrumIcon, StackIcon } from "./icons";
import { Callout, Card, Empty, Fields, NumberField, NumberInput, Page, Segmented, SelectField, Stat, Stats } from "./kit";
import { runningJob } from "./jobs";
import { above, boundsOf, tipOf } from "./numbers";
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
  const durations = acquisition.durations.filter((d) => Number.isFinite(d));
  const rates = acquisition.sampling_frequencies.filter((f) => Number.isFinite(f));
  const low = durations.length ? Math.min(...durations) : 0;
  const high = durations.length ? Math.max(...durations) : 0;
  const rate = rates.length ? rates[0] : 0;
  return (
    <Card title="Acquisition" icon={<FolderIcon size={17} />}>
      <Stats>
        <Stat label="Records" value={acquisition.files.length} />
        <Stat label="Receivers" value={receivers.length} sub={`every ${+spacing.toFixed(3)} m`} />
        <Stat label="Line length" value={`${+length.toFixed(2)} m`} />
        <Stat label="Duration" value={low === high ? `${+low.toFixed(2)} s` : `${+low.toFixed(2)}–${+high.toFixed(2)} s`} />
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
      <Fields>
        <NumberField
          label="Length"
          unit="receivers"
          value={masw.length}
          onChange={(v) => setMasw({ ...masw, length: v })}
          min={3}
          max={receivers}
          hint={Number.isFinite(masw.length) ? `${+((masw.length - 1) * spacing).toFixed(2)} m` : undefined}
        />
        <NumberField
          label="Step"
          unit="receivers"
          value={masw.step}
          onChange={(v) => setMasw({ ...masw, step: v })}
          min={1}
          max={receivers}
          hint={Number.isFinite(masw.step) ? `${+(masw.step * spacing).toFixed(2)} m` : undefined}
        />
        <NumberField
          label="Nearest shot"
          title="From the window's middle"
          unit="m"
          value={masw.distance_min}
          onChange={(v) => setMasw({ ...masw, distance_min: v })}
          min={0}
        />
        <NumberField
          label="Farthest shot"
          title="From the window's middle"
          unit="m"
          value={masw.distance_max}
          onChange={(v) => setMasw({ ...masw, distance_max: v })}
          min={0}
          check={above(masw.distance_min)}
        />
      </Fields>
      <div style={{ marginTop: 18 }}>
        <GeometryPlot acquisition={acquisition} masw={masw} onCount={onCount} showSources={showSources} unit={unit} />
      </div>
    </Card>
  );
}

/** Each shot or record cut: muting on, the trigger's shift first (`trigger`, the modes of shots:
 * left empty, each record's own, from its file), then a time window and the arrivals between two
 * velocities, each bound left empty when none, the shot's pulse kept after the slowest; the
 * record previewed, what the muting removes veiled. Off, none of it applies. */
export function MutingRow({
  acquisition,
  muting,
  setMuting,
  maxTime,
  gather = true,
  trigger,
}: {
  acquisition: Acquisition;
  muting: MutingState;
  setMuting: (muting: MutingState) => void;
  maxTime: number;
  gather?: boolean;
  trigger?: { t0: number | null; setT0: (t0: number) => void };
}) {
  // What the files say of their trigger, optional (a file may not say it: no shift for it):
  // one value, in all of them or some, each its own, or nothing.
  const known = (acquisition.triggers ?? []).filter((t): t is number => t !== null);
  const said = [...new Set(known)];
  const where = known.length === acquisition.files.length ? "the files" : `${known.length} of ${acquisition.files.length} files`;
  const files =
    said.length === 0 ? "none in the files" : said.length === 1 ? `${+(said[0] * 1000).toFixed(1)} ms in ${where}` : `each file's own (${where})`;
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
        />
      }
    >
      {muting.method === "mute" && (
        <>
          <Fields>
            {trigger && (
              <NumberField
                label="Trigger t0"
                unit="s"
                title={"Moves each shot's time origin\nLate (t0 > 0): drops the record's first t0 seconds\nEarly (t0 < 0): pads its start\nEmpty: each record's own, from its file"}
                value={trigger.t0 ?? Number.NaN}
                optional="files"
                hint={files}
                onChange={trigger.setT0}
                step={0.001}
              />
            )}
            <NumberField label="From" unit="s" value={muting.tmin ?? Number.NaN} optional="none" onChange={(v) => setMuting({ ...muting, tmin: v })} min={0} max={maxTime} step={0.1} />
            <NumberField label="To" unit="s" value={muting.tmax ?? Number.NaN} optional="none" onChange={(v) => setMuting({ ...muting, tmax: v })} min={0} max={maxTime} step={0.1} check={above(muting.tmin ?? Number.NaN)} />
            <NumberField label="Slowest" unit="m/s" value={muting.vmin ?? Number.NaN} optional="none" onChange={(v) => setMuting({ ...muting, vmin: v })} min={0} />
            <NumberField label="Fastest" unit="m/s" value={muting.vmax ?? Number.NaN} optional="none" onChange={(v) => setMuting({ ...muting, vmax: v })} min={0} check={above(muting.vmin ?? Number.NaN)} />
            <NumberField
              label="Signal width"
              unit="s"
              title={"Kept after the slowest arrival: the shot's pulse\nSo that the window is not empty at the shot"}
              value={muting.width}
              onChange={(v) => setMuting({ ...muting, width: v })}
              min={0}
              step={0.01}
            />
            <NumberField label="Taper" unit="samples" value={muting.taper} onChange={(v) => setMuting({ ...muting, taper: v })} min={0} />
          </Fields>
          {gather && <MuteGather acquisition={acquisition} muting={muting} trigger={trigger ? (trigger.t0 ?? null) : 0} />}
        </>
      )}
    </Row>
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
          <NumberField label="High cut" unit="Hz" value={filtering.fmax} onChange={(v) => setFiltering({ ...filtering, fmax: v })} min={0} max={nyquist} step={5} check={above(filtering.fmin)} />
          <NumberField label="Order" value={filtering.order} onChange={(v) => setFiltering({ ...filtering, order: v })} min={4} step={1} />
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
          <NumberField label="Power ν" value={stacking.nu} onChange={(v) => setStacking({ ...stacking, nu: v })} min={0} />
        </Fields>
      )}
      {stacking.method === "root" && (
        <Fields>
          <NumberField label="Root n" value={stacking.n} onChange={(v) => setStacking({ ...stacking, n: v })} min={1} />
        </Fields>
      )}
    </Row>
  );
}

export function DispersionRows({
  dispersion,
  setDispersion,
  nyquist,
}: {
  dispersion: Dispersion;
  setDispersion: (dispersion: Dispersion) => void;
  nyquist: number;
}) {
  return (
    <>
      <Row icon={<SpectrumIcon size={16} />} title="Frequencies">
        <Fields>
          <NumberField label="From" unit="Hz" value={dispersion.fmin} onChange={(v) => setDispersion({ ...dispersion, fmin: v })} min={0} max={nyquist} />
          <NumberField label="To" unit="Hz" value={dispersion.fmax} onChange={(v) => setDispersion({ ...dispersion, fmax: v })} min={0} max={nyquist} check={above(dispersion.fmin)} />
        </Fields>
      </Row>
      <Row icon={<RulerIcon size={16} />} title="Phase velocities">
        <Fields>
          <NumberField label="From" unit="m/s" value={dispersion.vmin} onChange={(v) => setDispersion({ ...dispersion, vmin: v })} min={1} />
          <NumberField label="To" unit="m/s" value={dispersion.vmax} onChange={(v) => setDispersion({ ...dispersion, vmax: v })} min={1} check={above(dispersion.vmin)} />
          <NumberField label="Steps" value={dispersion.nv} onChange={(v) => setDispersion({ ...dispersion, nv: v })} min={1000} />
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
    <label className="workers" data-tip={tipOf(tip, boundsOf(1, maxWorkers))}>
      <CpuIcon size={15} />
      <NumberInput min={1} max={maxWorkers} value={workers} onChange={setWorkers} data-tip={tip} />
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
