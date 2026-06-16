import { useState } from "react";
import { API, type Acquisition, type Masw, type Muting, type Dispersion } from "../api";
import { MaswPreview } from "./MaswPreview";

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step=1,
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

interface ValidationError {
  loc: (string | number)[];
  msg: string;
}

export function ConfigForm({ acquisition }: { acquisition: Acquisition }) {
  const [slicing, setSlicing] = useState({ length: 0.1, step: 0.1});
  const [masw, setMasw] = useState<Masw>({ length: 3, step: 1, distance_min: 0, distance_max: 100 });
  const [dispersion, setDispersion] = useState<Dispersion>({ fmin: 0, fmax: 100, vmin: 1, vmax: 1000, dv: 1 });
  const [execution, setExecution] = useState({ n_workers: 1 });

  const [result, setResult] = useState<string | null>(null);
  const [errors, setErrors] = useState<ValidationError[]>([]);

  function validate() {
    setResult(null);
    setErrors([]);

    const config = {
      mode: "active",
      acquisition_params: acquisition,
      masw_params: masw,
      dispersion_params: dispersion,
      muting_params: muting,
      execution_params: execution,
    };

    fetch(`${API}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    })
      .then(async (res) => {
        if (res.ok) {
          setResult("Config is valid ✓");
        } else if (res.status === 422) {
          const body = await res.json();
          setErrors(body.detail);
        } else {
          setResult(`Unexpected error: HTTP ${res.status}`);
        }
      })
      .catch((err) => setResult(String(err)));
  }

  return (
    <div>

      <h2>Slicing</h2>
      <NumberField label="Segment length [s]" value={slicing.length} onChange={(v) => setSlicing({ ...slicing, length: v })} min={0.1} max={acquisition.receiver_positions.length} step={0.1}/>
      <NumberField label="Segment step [s]" value={slicing.step} onChange={(v) => setSlicing({ ...slicing, step: v })} min={0.01} max={acquisition.receiver_positions.length} step={0.01} />

      <h2>MASW windows</h2>
      <NumberField label="Length [#]" value={masw.length} onChange={(v) => setMasw({ ...masw, length: v })} min={3} max={acquisition.receiver_positions.length}/>
      <NumberField label="Step [#]" value={masw.step} onChange={(v) => setMasw({ ...masw, step: v })} min={1} max={acquisition.receiver_positions.length} />
      <NumberField label="Min distance from sources [m]" value={masw.distance_min} onChange={(v) => setMasw({ ...masw, distance_min: v })} min={0} />
      <NumberField label="Max distance from sources [m]" value={masw.distance_max} onChange={(v) => setMasw({ ...masw, distance_max: v })} min={0} />

      <MaswPreview acquisition={acquisition} masw={masw} />

      <h2>Dispersion</h2>
      <NumberField label="Min frequency [Hz]" value={dispersion.fmin} onChange={(v) => setDispersion({ ...dispersion, fmin: v })} min={0} max={Number(acquisition.sampling_frequencies[0]?.toFixed(2) ?? 0)/2}/>
      <NumberField label="Max frequency [Hz]" value={dispersion.fmax} onChange={(v) => setDispersion({ ...dispersion, fmax: v })} min={0} max={Number(acquisition.sampling_frequencies[0]?.toFixed(2) ?? 0)/2}/>
      <NumberField label="Min phase velocity [m/s]" value={dispersion.vmin} onChange={(v) => setDispersion({ ...dispersion, vmin: v })} min={1} />
      <NumberField label="Max phase velocity [m/s]" value={dispersion.vmax} onChange={(v) => setDispersion({ ...dispersion, vmax: v })} min={1} />
      <NumberField label="Phase velocity step [m/s]" value={dispersion.dv} onChange={(v) => setDispersion({ ...dispersion, dv: v })} min={1} />

      <h2>Execution</h2>
      <NumberField label="Number of workers" value={execution.n_workers} onChange={(v) => setExecution({ ...execution, n_workers: v })} min={1} />

      <button onClick={validate} style={{ marginTop: 12 }}>Validate</button>

      {result && <p>{result}</p>}
      {errors.length > 0 && (
        <ul>
          {errors.map((e, i) => (
            <li key={i}>{e.loc.slice(1).join(".")}: {e.msg}</li>
          ))}
        </ul>
      )}
    </div>
  );
}