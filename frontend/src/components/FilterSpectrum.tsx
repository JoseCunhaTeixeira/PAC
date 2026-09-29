import { useEffect, useMemo, useState } from "react";
import { API, type Acquisition } from "../api";
import type { FilteringState } from "../presets";
import { useTheme } from "../theme";
import { BoxTools, PlotBox } from "./kit";
import { LinePlot, type PlotArea } from "./viz/LinePlot";
import { vizPalette } from "./viz/palette";

// The filter's preview on a computing page, as the muting's is: a record's mean power spectrum
// over its traces, the band the filter keeps shaded as its cuts are typed.

interface Spectrum {
  freqs: number[];
  power_db: number[];
}

export function FilterSpectrum({ acquisition, filtering }: { acquisition: Acquisition; filtering: FilteringState }) {
  const folder = acquisition.folder_path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
  const [file, setFile] = useState(acquisition.files[0] ?? "");
  const [spectrum, setSpectrum] = useState<Spectrum | null>(null);
  const [error, setError] = useState<string | null>(null);
  const palette = vizPalette(useTheme());

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
      .then((data: Spectrum) => setSpectrum(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [file, folder]);

  const floor = useMemo(() => Math.min(-60, ...(spectrum?.power_db ?? [0])), [spectrum]);
  // The band an IIR filter keeps: from its low cut to its high cut.
  const band = filtering.method === "iir" && filtering.fmax > filtering.fmin;
  const areas: PlotArea[] = band
    ? [
        {
          color: palette.band,
          polygon: [
            [filtering.fmin, floor],
            [filtering.fmax, floor],
            [filtering.fmax, 0],
            [filtering.fmin, 0],
          ],
        },
      ]
    : [];

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
        {spectrum && <BoxTools />}
      </div>
      {error && <p style={{ color: "var(--accent)" }}>Error: {error}</p>}
      {spectrum && (
        <LinePlot
          key={file}
          series={[
            {
              label: band ? `${file} · the band kept shaded` : file,
              color: palette.series,
              points: spectrum.freqs.map((f, i): [number, number] => [f, spectrum.power_db[i]]),
              width: 1.4,
            },
          ]}
          areas={areas}
          xLabel="Frequency (Hz)"
          yLabel="Power (dB of the peak)"
          height={240}
        />
      )}
    </PlotBox>
  );
}
