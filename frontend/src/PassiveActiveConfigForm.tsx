import { useState } from "react";
import { type Acquisition } from "./api";
import { MaswPreview } from "./components/MaswPreview";
import { MuteGather } from "./components/MuteGather";
import { RunPanel } from "./components/RunPanel";
import { buildMutingParams, buildFilteringParams, buildStackingParams} from "./builders";


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

export function ConfigForm({ acquisition }: { acquisition: Acquisition }) {
  const maxTime = Number(acquisition.durations[0]?.toFixed(2) ?? 0);
  const nyquist = (acquisition.sampling_frequencies[0] ?? 0) / 2;
  const nCpus = navigator.hardwareConcurrency || 1;

  const [masw, setMasw] = useState({ length: 3, step: 1, distance_min: 0, distance_max: 100 });
  const [muting, setMuting] = useState({ method: "none", tmin: 0, tmax: maxTime, vmin: 0, vmax: 100_000, taper: 0 });
  const [filtering, setFiltering] = useState({ method: "none", fmin: 0, fmax: nyquist, order: 4 });
  const [dispersion, setDispersion] = useState({ fmin: 0, fmax: 100, vmin: 1, vmax: 1_000, nv: 1_000 });
  const [stacking, setStacking] = useState({ method: "linear", nu: 2, n : 2});
  const [execution, setExecution] = useState({ n_workers: 1 });
  const [nPositions, setNPositions] = useState(0);


  const config = {
    mode: "passive-active",
    acquisition_params: acquisition,
    masw_params: masw,
    muting_params: buildMutingParams(muting),
    filtering_params: buildFilteringParams(filtering),
    stacking_params: buildStackingParams(stacking),
    dispersion_params: dispersion,
    execution_params: execution,
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

      <h2>Stacking</h2>
      <label style={{ display: "block", margin: "4px 0" }}>
        Method :{" "}
        <select value={stacking.method} onChange={(e) => setStacking({ ...stacking, method: e.target.value })}>
          <option value="linear">Linear</option>
          <option value="phase_weighted">Phase-weighted</option>
          <option value="root">Root</option>
        </select>
      </label>
      {stacking.method === "phase_weighted" && (
        <NumberField label="Power" value={stacking.nu} onChange={(v) => setStacking({ ...stacking, nu: v })} />
      )}
      {stacking.method === "root" && (
        <NumberField label="Power" value={stacking.n} onChange={(v) => setStacking({ ...stacking, n: v })} />
      )}

      <h2>Dispersion</h2>
      <NumberField label="Min frequency [Hz]" value={dispersion.fmin} onChange={(v) => setDispersion({ ...dispersion, fmin: v })} min={0} max={nyquist} />
      <NumberField label="Max frequency [Hz]" value={dispersion.fmax} onChange={(v) => setDispersion({ ...dispersion, fmax: v })} min={0} max={nyquist} />
      <NumberField label="Min phase velocity [m/s]" value={dispersion.vmin} onChange={(v) => setDispersion({ ...dispersion, vmin: v })} min={1} />
      <NumberField label="Max phase velocity [m/s]" value={dispersion.vmax} onChange={(v) => setDispersion({ ...dispersion, vmax: v })} min={1} />
      <NumberField label="Number of samples [#]" value={dispersion.nv} onChange={(v) => setDispersion({ ...dispersion, nv: v })} min={1_000} />

      <h2>Execution</h2>
      <NumberField label="Number of workers" value={execution.n_workers} onChange={(v) => setExecution({ n_workers: Math.min(v, maxWorkers) })} min={1} />

      <RunPanel config={config} />
    </div>
  );
}