import { useEffect, useState } from "react";
import { ConfigForm } from "./PassiveActiveConfigForm";
import { API, type Acquisition } from "./api";

export default function App() {
  const [folders, setFolders] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [acquisition, setAcquisition] = useState<Acquisition | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [missingSource, setMissingSource] = useState(false);

  useEffect(() => {
    fetch(`${API}/input_folders`)
      .then((res) => res.json())
      .then((data: string[]) => setFolders(data))
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!selected) {
      setAcquisition(null);
      return;
    }
    setLoading(true);
    setError(null);
    setMissingSource(false);
    fetch(`${API}/acquisitions/${selected}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: Acquisition) => {
        if (data.source_positions.length === 0) {
          setMissingSource(true);
          setAcquisition(null);
          return;
        }
        setAcquisition(data);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        setAcquisition(null);
      })
      .finally(() => setLoading(false));
  }, [selected]);

  return (
    <div style={{ padding: 24 }}>
      <h1>Passive-Active Computing</h1>

      <label>
        <h2>Loading</h2>
        Acquisition folder:{" "}
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          <option value="">— choose —</option>
          {folders.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>

      {loading && <p>Loading acquisition…</p>}
      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}
      {missingSource && (
        <p
          style={{
            background: "var(--info-bg)",
            color: "var(--info-text)",
            padding: "8px 12px",
            borderRadius: "var(--radius-sm)",
          }}
        >
          ⚠️ source_positions.yaml is missing for this folder — required for active computing
        </p>
      )}

      {acquisition && (
        <>
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Duration [s]</th>
                <th>Sampling frequency [Hz]</th>
                <th>Source x, z [m]</th>
                <th>Receivers [#]</th>
              </tr>
            </thead>
            <tbody>
              {acquisition.files.map((file, i) => (
                <tr key={file}>
                  <td>{file}</td>
                  <td>{acquisition.durations[i]?.toFixed(2) ?? "—"}</td>
                  <td>{acquisition.sampling_frequencies[i]?.toFixed(2) ?? "—"}</td>
                  <td>
                    {acquisition.source_positions[i]
                      ? `${acquisition.source_positions[i][0]}, ${acquisition.source_positions[i][1]}`
                      : "—"}
                  </td>
                  <td>{acquisition.receiver_positions.length ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <ConfigForm acquisition={acquisition} />
        </>
      )}
    </div>
  );
}