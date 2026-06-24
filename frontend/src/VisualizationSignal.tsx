import { useEffect, useState } from "react";
import { API, type Acquisition } from "./api";
import { MuteGather } from "./components/MuteGather";

export function VisualizationSignal({ folder }: { folder: string }) {
  const [acquisition, setAcquisition] = useState<Acquisition | null>(null);
  const [normalize, setNormalize] = useState<"trace" | "global">("trace");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setAcquisition(null);
    setError(null);
    fetch(`${API}/acquisitions/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: Acquisition) => setAcquisition(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [folder]);

  if (error) return <p style={{ color: "var(--accent)" }}>Error: {error}</p>;
  if (!acquisition) return null;

  if (acquisition.files.length === 0) {
    return <p>❌ Selected input data folder empty.</p>;
  }

  return (
    <>
      <h2>Seismic records</h2>
      <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <input
          type="checkbox"
          checked={normalize === "trace"}
          onChange={(e) => setNormalize(e.target.checked ? "trace" : "global")}
        />
        Normalize by trace
      </label>

      {acquisition.files.map((file) => (
        <div key={file} style={{ marginTop: 24 }}>
          <h3>Record {file}</h3>
          <MuteGather acquisition={acquisition} file={file} norm={normalize} />
        </div>
      ))}
    </>
  );
}
