import { useEffect, useState } from "react";
import { API, type Acquisition } from "../api";
import type { FilteringState } from "../presets";
import { BoxTools, PlotBox } from "./kit";
import { SpectrumCanvas, type TraceSpectra } from "./SpectrumCanvas";

// The filter's preview on a computing page, as the muting's is: a record's spectra as its saved
// figure draws them (each trace's, at its receiver, the whole of it), the band the filter keeps
// dashed as its cuts are typed.

export function FilterSpectrum({ acquisition, filtering }: { acquisition: Acquisition; filtering: FilteringState }) {
  const folder = acquisition.folder_path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
  const [file, setFile] = useState(acquisition.files[0] ?? "");
  const [spectra, setSpectra] = useState<TraceSpectra | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return;
    Promise.resolve().then(() => setError(null));
    fetch(`${API}/spectrum/${encodeURIComponent(folder)}/${encodeURIComponent(file)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: TraceSpectra) => setSpectra(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [file, folder]);

  // The band an IIR filter keeps: from its low cut to its high cut.
  const band: [number, number] | null =
    filtering.method === "iir" && filtering.fmax > filtering.fmin ? [filtering.fmin, filtering.fmax] : null;

  return (
    <PlotBox>
      <div className="box-head">
        <label className="inline-field">
          Preview on
          <select value={file} onChange={(e) => setFile(e.target.value)}>
            {acquisition.files.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
        {spectra && <BoxTools />}
      </div>
      {error && <p style={{ color: "var(--accent)" }}>Error: {error}</p>}
      {spectra && (
        <SpectrumCanvas
          key={file}
          spectra={spectra}
          positions={acquisition.receiver_positions.map((position) => position[0])}
          band={band}
        />
      )}
    </PlotBox>
  );
}
