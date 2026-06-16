import { useState } from "react";
import { API, type Acquisition, type Masw } from "./api";
import { MaswPreview } from "./MaswPreview";

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label style={{ display: "block", margin: "4px 0" }}>
      {label}:{" "}
      <input
        type="number"
        value={value}
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
  const [masw, setMasw] = useState<Masw>({ length: 3, step: 1, distance_min: 0, distance_max: 100 });
  const [dispersion, setDispersion] = useState({ fmin: 0, fmax: 100, vmin: 1, vmax: 1000, dv: 1 });
  const [muting, setMuting] = useState({ vmin: 0, vmax: 10000, taper: 0 });
  const [stacking, setStacking] = useState({ method: "linear", power: 2 });
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
      stacking_params: {
        method: stacking.method,
        power: stacking.method === "linear" ? null : stacking.power,
      },
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
      <h2>Configuration</h2>

      <h3>MASW windows</h3>
      <NumberField label="Length [#] " value={masw.length} onChange={(v) => setMasw({ ...masw, length: v })} />
      <NumberField label="Step [#] " value={masw.step} onChange={(v) => setMasw({ ...masw, step: v })} />
      <NumberField label="Min distance from sources [m] " value={masw.distance_min} onChange={(v) => setMasw({ ...masw, distance_min: v })} />
      <NumberField label="Max distance from sources [m] " value={masw.distance_max} onChange={(v) => setMasw({ ...masw, distance_max: v })} />

      <MaswPreview acquisition={acquisition} masw={masw} />

      <h3>Dispersion</h3>
      <NumberField label="Min frequency [Hz] " value={dispersion.fmin} onChange={(v) => setDispersion({ ...dispersion, fmin: v })} />
      <NumberField label="Max frequency [Hz] " value={dispersion.fmax} onChange={(v) => setDispersion({ ...dispersion, fmax: v })} />
      <NumberField label="Min phase velocity [m/s] " value={dispersion.vmin} onChange={(v) => setDispersion({ ...dispersion, vmin: v })} />
      <NumberField label="Max phase velocity [m/s] " value={dispersion.vmax} onChange={(v) => setDispersion({ ...dispersion, vmax: v })} />
      <NumberField label="Phase velocity step [m/s] " value={dispersion.dv} onChange={(v) => setDispersion({ ...dispersion, dv: v })} />

      <h3>Muting</h3>
      <NumberField label="Min group velocity [m/s] " value={muting.vmin} onChange={(v) => setMuting({ ...muting, vmin: v })} />
      <NumberField label="Max group velocity [m/s] " value={muting.vmax} onChange={(v) => setMuting({ ...muting, vmax: v })} />
      <NumberField label="Taper width [#] " value={muting.taper} onChange={(v) => setMuting({ ...muting, taper: v })} />

      <h3>Stacking</h3>
      <label style={{ display: "block", margin: "4px 0" }}>
        Method :{" "}
        <select value={stacking.method} onChange={(e) => setStacking({ ...stacking, method: e.target.value })}>
          <option value="linear">linear</option>
          <option value="pws">pws</option>
          <option value="root">root</option>
        </select>
      </label>
      {stacking.method !== "linear" && (
        <NumberField label="power" value={stacking.power} onChange={(v) => setStacking({ ...stacking, power: v })} />
      )}

      <h3>Execution</h3>
      <NumberField label="Number of workers " value={execution.n_workers} onChange={(v) => setExecution({ ...execution, n_workers: v })} />

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