import { useEffect, useState } from "react";
import { API } from "./api";
import { VisualizationSignal } from "./VisualizationSignal";
import { VisualizationDispersion } from "./VisualizationDispersion";
import { VisualizationInversion } from "./VisualizationInversion";

type Mode = "Signal" | "Dispersion" | "Inversion";

const MODES: Mode[] = ["Signal", "Dispersion", "Inversion"];

export default function VisualizationPage() {
  const [mode, setMode] = useState<Mode | "">("");
  const [folders, setFolders] = useState<string[]>([]);
  const [folder, setFolder] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loadingFolders, setLoadingFolders] = useState(false);

  useEffect(() => {
    Promise.resolve().then(() => {
      setFolder("");
      setFolders([]);
      setError(null);
    });
    if (!mode) return;
    Promise.resolve().then(() => setLoadingFolders(true));
    const endpoint = mode === "Signal" ? "input_folders" : "output_folders";
    fetch(`${API}/${endpoint}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: string[]) => setFolders(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoadingFolders(false));
  }, [mode]);

  return (
    <div style={{ padding: 24 }}>
      <h1>Visualization</h1>
      {/* <p>🛈 Browse raw seismic records, dispersion images and inversion results from any folder.</p> */}

      <div style={{ marginBottom: 32 }}>
        <label>
          <h2>Loading</h2>
          Visualization mode:{" "}
          <select
            value={mode}
            onChange={(e) => {
              // Clear folder and flag folders as loading synchronously (not in
              // the effect below), so a mode switch never briefly mounts a
              // visualization with a stale folder, or flashes "No folders
              // found" before the fetch for the new mode has even started.
              setFolder("");
              setLoadingFolders(true);
              setMode(e.target.value as Mode | "");
            }}
          >
            <option value="">— choose —</option>
            {MODES.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>

        {mode && (
          <div style={{ marginTop: 12 }}>
            <label>
              Data folder:{" "}
              <select value={folder} onChange={(e) => setFolder(e.target.value)}>
                <option value="">— choose —</option>
                {folders.map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </label>
            {!loadingFolders && folders.length === 0 && <p>❌ No folders found.</p>}
          </div>
        )}
      </div>

      {error && <p style={{ color: "var(--accent)" }}>Error: {error}</p>}

      {mode === "Signal" && folder && <VisualizationSignal folder={folder} />}
      {mode === "Dispersion" && folder && <VisualizationDispersion folder={folder} />}
      {mode === "Inversion" && folder && <VisualizationInversion folder={folder} />}
    </div>
  );
}
