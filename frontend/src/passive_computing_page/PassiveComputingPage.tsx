import { useEffect, useState } from "react";
import { ConfigForm } from "./ConfigForm";
import { API, type Acquisition } from "../api";

export default function App() {
  const [folders, setFolders] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [acquisition, setAcquisition] = useState<Acquisition | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/folders`)
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
    fetch(`${API}/acquisitions/${selected}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: Acquisition) => setAcquisition(data))
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [selected]);

  return (
    <div style={{ padding: 24 }}>
      <h1>Passive MASW computing</h1>

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

      {acquisition && (
        <>
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Duration [s]</th>
                <th>Sampling frequency [Hz]</th>
                <th>Source [m]</th>
                <th>Receivers [#]</th>
              </tr>
            </thead>
            <tbody>
              {acquisition.files.map((file, i) => (
                <tr key={file}>
                  <td>{file}</td>
                  <td>{acquisition.durations[i]?.toFixed(2) ?? "—"}</td>
                  <td>{acquisition.sampling_frequencies[i]?.toFixed(2) ?? "—"}</td>
                  <td>{acquisition.source_positions[i] ?? "—"}</td>
                  <td>{acquisition.receiver_positions.length ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <p>
            Receiver positions [m]:{" "}
            {acquisition.receiver_positions.join(", ")}
          </p>

          <ConfigForm acquisition={acquisition} />
        </>
      )}
    </div>
  );
}