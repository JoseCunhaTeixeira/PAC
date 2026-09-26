import { useState } from "react";
import { type Acquisition, type Dispersion, type Masw } from "./api";
import { MaswPreview } from "./components/MaswPreview";
import { MuteGather } from "./components/MuteGather";
import { RunPanel } from "./components/RunPanel";
import { buildFilteringParams, buildMutingParams } from "./builders";
import { type FilteringState, type MutingState, type PresetDefaults, stage, usePreset } from "./presets";

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label style={{ display: "block", margin: "4px 0" }}>
      {label}:{" "}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function ConfigForm({ acquisition, profile }: { acquisition: Acquisition; profile: string }) {
  const { preset, error } = usePreset("active", profile);
  if (error) return <p style={{ color: "crimson" }}>Error: {error}</p>;
  if (!preset) return <p>Loading settings…</p>;
  return <Form acquisition={acquisition} profile={profile} preset={preset} />;
}

function Form({
  acquisition,
  profile,
  preset,
}: {
  acquisition: Acquisition;
  profile: string;
  preset: PresetDefaults;
}) {
  const maxTime = Number(acquisition.durations[0]?.toFixed(2) ?? 0);
  const nyquist = (acquisition.sampling_frequencies[0] ?? 0) / 2;
  const nCpus = navigator.hardwareConcurrency || 1;

  const [masw, setMasw] = useState(() => stage<Masw>(preset, "masw"));
  const [trigger, setTrigger] = useState(() => stage<{ t0: number }>(preset, "trigger"));
  const [muting, setMuting] = useState(() => stage<MutingState>(preset, "muting"));
  const [filtering, setFiltering] = useState(() => stage<FilteringState>(preset, "filtering"));
  const [dispersion, setDispersion] = useState(() => stage<Dispersion>(preset, "dispersion"));
  const [execution, setExecution] = useState({ n_workers: 1 });
  const [nPositions, setNPositions] = useState(0);

  const config = {
    profile,
    mode: "active",
    overrides: {
      masw,
      trigger,
      muting: buildMutingParams(muting),
      filtering: buildFilteringParams(filtering),
      dispersion,
    },
    workers: execution.n_workers,
  };

  const maxWorkers = nPositions > 0 ? Math.min(nCpus, nPositions) : nCpus;

  return (
    <div>

      <h2>MASW windows</h2>
      <NumberField label="Length [#]" value={masw.length} onChange={(v) => setMasw({ ...masw, length: v })} min={3} max={acquisition.receiver_positions.length} />
      <NumberField label="Step [#]" value={masw.step} onChange={(v) => setMasw({ ...masw, step: v })} min={1} max={acquisition.receiver_positions.length} />
      <NumberField label="Min distance from sources [m]" value={masw.distance_min} onChange={(v) => setMasw({ ...masw, distance_min: v })} min={0} />
      <NumberField label="Max distance from sources [m]" value={masw.distance_max} onChange={(v) => setMasw({ ...masw, distance_max: v })} min={0} />
      <MaswPreview acquisition={acquisition} masw={masw} onCount={setNPositions} />

      <h2>Trigger</h2>
      <NumberField label="Time origin shift t0 [s]" value={trigger.t0} onChange={(v) => setTrigger({ t0: v })} step={0.001} />

      <h2>Signal muting</h2>
      <label style={{ display: "block", margin: "4px 0" }}>
        Method:{" "}
        <select value={muting.method} onChange={(e) => setMuting({ ...muting, method: e.target.value })}>
          <option value="none">None</option>
          <option value="mute">Mute</option>
        </select>
      </label>
      {muting.method === "mute" && (
        <>
          <NumberField label="Min time [s]" value={muting.tmin} onChange={(v) => setMuting({ ...muting, tmin: v })} min={0} max={maxTime} step={0.1} />
          <NumberField label="Max time [s]" value={muting.tmax} onChange={(v) => setMuting({ ...muting, tmax: v })} min={0} max={maxTime} step={0.1} />
          <NumberField label="Min group velocity [m/s]" value={muting.vmin} onChange={(v) => setMuting({ ...muting, vmin: v })} min={0} />
          <NumberField label="Max group velocity [m/s]" value={muting.vmax} onChange={(v) => setMuting({ ...muting, vmax: v })} min={0} />
          <NumberField label="Taper width [#]" value={muting.taper} onChange={(v) => setMuting({ ...muting, taper: v })} min={0} />
          <MuteGather acquisition={acquisition} muting={muting} />
        </>
      )}

      <h2>Spectral filtering</h2>
      <label style={{ display: "block", margin: "4px 0" }}>
        Method:{" "}
        <select value={filtering.method} onChange={(e) => setFiltering({ ...filtering, method: e.target.value })}>
          <option value="none">None</option>
          <option value="iir">IIR</option>
        </select>
      </label>
      {filtering.method === "iir" && (
        <>
          <NumberField label="Min frequency [Hz]" value={filtering.fmin} onChange={(v) => setFiltering({ ...filtering, fmin: v })} min={0} max={nyquist} step={5} />
          <NumberField label="Max frequency [Hz]" value={filtering.fmax} onChange={(v) => setFiltering({ ...filtering, fmax: v })} min={0} max={nyquist} step={5} />
          <NumberField label="Max frequency [Hz]" value={filtering.order} onChange={(v) => setFiltering({ ...filtering, order: v })} min={4} step={1} />
        </>
      )}

      <h2>Dispersion</h2>
      <NumberField label="Min frequency [Hz]" value={dispersion.fmin} onChange={(v) => setDispersion({ ...dispersion, fmin: v })} min={0} max={nyquist} />
      <NumberField label="Max frequency [Hz]" value={dispersion.fmax} onChange={(v) => setDispersion({ ...dispersion, fmax: v })} min={0} max={nyquist} />
      <NumberField label="Min phase velocity [m/s]" value={dispersion.vmin} onChange={(v) => setDispersion({ ...dispersion, vmin: v })} min={1} />
      <NumberField label="Max phase velocity [m/s]" value={dispersion.vmax} onChange={(v) => setDispersion({ ...dispersion, vmax: v })} min={1} />
      <NumberField label="Number of samples [#]" value={dispersion.nv} onChange={(v) => setDispersion({ ...dispersion, nv: v })} min={1000} />

      <h2>Execution</h2>
      <NumberField label="Number of workers" value={execution.n_workers} onChange={(v) => setExecution({ n_workers: Math.min(v, maxWorkers) })} min={1} />

      <RunPanel config={config} />
    </div>
  );
}
