import { useState } from "react";
import { API, type Acquisition } from "./api";
import { MaswPreview } from "./components/MaswPreview";
import { MuteGather } from "./components/MuteGather";
import { RunPanel } from "./components/RunPanel";
import { buildMutingParams, buildNormalizationParams, buildFilteringParams, buildSelectionParams, buildStackingParams, buildWhiteningParams} from "./builders";

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
  const maxTime = Number(Math.max(...acquisition.durations).toFixed(2));
  const nyquist = (acquisition.sampling_frequencies[0] ?? 0) / 2;
  const nCpus = navigator.hardwareConcurrency || 1;

  const [masw, setMasw] = useState({ length: 3, step: 1, distance_min: 0, distance_max: 100 });
  const [muting, setMuting] = useState({ method: "none", tmin: 0, tmax: maxTime, vmin: 0, vmax: 100_000, taper: 0 });
  const [filtering, setFiltering] = useState({ method: "none", fmin: 0, fmax: nyquist, order: 4 });
  const [slicing, setSlicing] = useState({ segment_duration: 0.1, segment_step: 0.1});
  const [selection, SetSelection] = useState({ method: "none", threshold: 0.1, vmin: 0, vmax: 100_000});
  const [whitening, SetWhitening] = useState({ method: "none", fmin: 0, fmax : nyquist, taper_width_Hz: 5});
  const [normalization, setNormalization] = useState({ method: "none"});
  const [dispersion, setDispersion] = useState({ fmin: 0, fmax: 100, vmin: 1, vmax: 1_000, nv: 1_000 });
  const [stacking, setStacking] = useState({ method: "linear", nu: 2, n : 2});
  const [execution, setExecution] = useState({ n_workers: 1 });
  const [nPositions, setNPositions] = useState(0);

  const config = {
    mode: "passive",
    acquisition_params: acquisition,
    masw_params: masw,
    muting_params: buildMutingParams(muting),
    filtering_params: buildFilteringParams(filtering),
    slicing_params: slicing,
    selection_params: buildSelectionParams(selection),
    whitening_params: buildWhiteningParams(whitening),
    normalization_params: buildNormalizationParams(normalization),
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

      <h2>Slicing</h2>
      <NumberField label="Segment length [s]" value={slicing.segment_duration} onChange={(v) => setSlicing({ ...slicing, segment_duration: v })} min={0.1} max={maxTime} step={0.05} />
      <NumberField label="Segment step [s]" value={slicing.segment_step} onChange={(v) => setSlicing({ ...slicing, segment_step: v })} min={0.01} max={maxTime} step={0.05} />

      <h2>Slice selection</h2>
      <label style={{ display: "block", margin: "4px 0" }}>
        Method:{" "}
        <select value={selection.method} onChange={(e) => SetSelection({ ...selection, method: e.target.value })}>
          <option value="none">None</option>
          <option value="fk">FK</option>
        </select>
      </label>
      {selection.method === "fk" && (
        <>
          <NumberField label="Threshold" value={selection.threshold} onChange={(v) => SetSelection({ ...selection, threshold: v })} min={0} max={1} step={0.1} />
          <NumberField label="Minimum phase velocity [m/s]" value={selection.vmin} onChange={(v) => SetSelection({ ...selection, vmin: v })} min={0} />
          <NumberField label="Maximum phase velocity [m/s]" value={selection.vmax} onChange={(v) => SetSelection({ ...selection, vmax: v })} min={0} />
        </>
      )}

      <h2>Spectral whitening</h2>
      <label style={{ display: "block", margin: "4px 0" }}>
        Method:{" "}
        <select value={whitening.method} onChange={(e) => SetWhitening({ ...whitening, method: e.target.value })}>
          <option value="none">None</option>
          <option value="onebit">Onebit</option>
          <option value="onebit_apod">Onebit + Taper</option>
        </select>
      </label>
      {whitening.method === "onebit_apod" && (
        <>
          <NumberField label="Min frequency [Hz]" value={whitening.fmin} onChange={(v) => SetWhitening({ ...whitening, fmin: v })} min={0} max={nyquist} step={5} />
          <NumberField label="Max frequency [Hz]" value={whitening.fmax} onChange={(v) => SetWhitening({ ...whitening, fmax: v })} min={0} max={nyquist} step={5} />
          <NumberField label="Taper width [Hz]" value={whitening.taper_width_Hz} onChange={(v) => SetWhitening({ ...whitening, taper_width_Hz: v })} min={0} max={nyquist/4} step={1} />
        </>
      )}

      <h2>Temporal normalization</h2>
      <label style={{ display: "block", margin: "4px 0" }}>
        Method:{" "}
        <select value={normalization.method} onChange={(e) => setNormalization({ ...normalization, method: e.target.value })}>
          <option value="none">None</option>
          <option value="onebit">Onebit</option>
        </select>
      </label>

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