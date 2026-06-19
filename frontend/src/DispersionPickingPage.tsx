import { useEffect, useState } from "react";
import { API } from "./api";
import { DispersionImageCanvas, type DispersionImage } from "./components/DispersionImageCanvas";
import { PseudoSectionCanvas, type PseudoSection } from "./components/PseudoSectionCanvas";

const LABEL_PATTERN = /^[A-Z]{1,3}[0-9]+$/;

function sanitizeLabel(raw: string): string {
  return raw.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8);
}

export default function DispersionPickingPage() {
  const [folders, setFolders] = useState<string[]>([]);
  const [folder, setFolder] = useState("");

  const [xmids, setXmids] = useState<number[]>([]);
  const [xmid, setXmid] = useState<number | null>(null);

  const [image, setImage] = useState<DispersionImage | null>(null);
  const [pendingPolygon, setPendingPolygon] = useState<[number, number][] | null>(null);
  const [label, setLabel] = useState("M0");

  const [labelCounts, setLabelCounts] = useState<Record<string, number>>({});
  const [pseudoMode, setPseudoMode] = useState<"frequency" | "wavelength">("frequency");
  const [pseudoSections, setPseudoSections] = useState<Record<string, PseudoSection>>({});
  const [positionPicks, setPositionPicks] = useState<{ xmid: number; labels: string[] }[]>([]);

  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/output_folders`)
      .then((res) => res.json())
      .then((data: string[]) => setFolders(data))
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    setXmid(null);
    setImage(null);
    setLabelCounts({});
    setPseudoSections({});
    setPositionPicks([]);
    if (!folder) {
      setXmids([]);
      return;
    }
    fetch(`${API}/xmids/${encodeURIComponent(folder)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: number[]) => setXmids(data))
      .catch((err) => setError(String(err)));
    refreshLabels(folder);
    refreshPositionPicks(folder);
  }, [folder]);

  function loadImage(folderName: string, xmidValue: number) {
    setError(null);
    fetch(`${API}/dispersion_images/${encodeURIComponent(folderName)}/${xmidValue}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: DispersionImage) => {
        setImage(data);
        setPendingPolygon(null);
        setLabel(`M${data.curves.length}`);
      })
      .catch((err) => setError(String(err)));
  }

  useEffect(() => {
    if (folder && xmid !== null) loadImage(folder, xmid);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [folder, xmid]);

  function refreshLabels(folderName: string) {
    fetch(`${API}/dispersion_image_labels/${encodeURIComponent(folderName)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: Record<string, number>) => {
        setLabelCounts(data);
        setPseudoSections((prev) =>
          Object.fromEntries(Object.entries(prev).filter(([lbl]) => lbl in data))
        );
        Object.keys(data).forEach((labelValue) => loadPseudoSection(folderName, labelValue));
      })
      .catch((err) => setError(String(err)));
  }

  function refreshPositionPicks(folderName: string) {
    fetch(`${API}/dispersion_picks_by_position/${encodeURIComponent(folderName)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: { xmid: number; labels: string[] }[]) => setPositionPicks(data))
      .catch((err) => setError(String(err)));
  }

  function loadPseudoSection(folderName: string, labelValue: string) {
    setError(null);
    fetch(`${API}/dispersion_pseudo_section/${encodeURIComponent(folderName)}/${encodeURIComponent(labelValue)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: PseudoSection) => setPseudoSections((prev) => ({ ...prev, [labelValue]: data })))
      .catch((err) => setError(String(err)));
  }

  function handlePick() {
    if (!pendingPolygon || folder === "" || xmid === null) return;
    setError(null);
    fetch(`${API}/dispersion_images/${encodeURIComponent(folder)}/${xmid}/pick/lasso`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: pendingPolygon, label }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: DispersionImage) => {
        setImage(data);
        setPendingPolygon(null);
        setLabel(`M${data.curves.length}`);
        refreshLabels(folder);
        refreshPositionPicks(folder);
      })
      .catch((err) => setError(String(err)));
  }

  function handleDelete(curveLabel: string) {
    if (folder === "" || xmid === null) return;
    setError(null);
    fetch(`${API}/dispersion_images/${encodeURIComponent(folder)}/${xmid}/pick/${encodeURIComponent(curveLabel)}`, {
      method: "DELETE",
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: DispersionImage) => {
        setImage(data);
        refreshLabels(folder);
        refreshPositionPicks(folder);
      })
      .catch((err) => setError(String(err)));
  }

  return (
    <div style={{ padding: 24 }}>
      <h1>Dispersion Picking</h1>

      <div style={{ marginBottom: 32 }}>
        <label>
          <h2>Loading</h2>
          Data folder:{" "}
          <select
            value={folder}
            onChange={(e) => {
              setXmid(null);
              setFolder(e.target.value);
            }}
          >
            <option value="">— choose —</option>
            {folders.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {folder && xmids.length === 0 && <p>No positions found.</p>}

      {folder && positionPicks.length > 0 && (
        <div style={{ marginTop: 16, display: "flex", flexWrap: "wrap", gap: 8 }}>
          {positionPicks.map(({ xmid: posXmid, labels }) => {
            const picked = labels.length > 0;
            const selected = xmid === posXmid;
            return (
              <div
                key={posXmid}
                onClick={() => setXmid(posXmid)}
                title={picked ? `Picked: ${labels.join(", ")}` : "Not picked yet"}
                style={{
                  cursor: "pointer",
                  minWidth: 64,
                  padding: "6px 10px",
                  borderRadius: 6,
                  textAlign: "center",
                  background: picked ? "var(--success-bg)" : "var(--surface-hover)",
                  border: selected ? "2px solid var(--success-text)" : "2px solid transparent",
                  boxShadow: selected ? "0 0 0 1px var(--success-text)" : "none",
                }}
              >
                <div style={{ fontWeight: 600 }}>{posXmid.toFixed(2)} m</div>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  {picked ? labels.join(", ") : "—"}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {error && <p style={{ color: "var(--accent)" }}>Error: {error}</p>}

      {image && (
        <>
          <h2>Picking</h2>
          <p>🛈 Draw a lasso on the dispersion image to auto-pick the curve inside it.</p>
          <DispersionImageCanvas
            image={image}
            pendingPolygon={pendingPolygon}
            onLassoComplete={setPendingPolygon}
          />
          <div style={{ marginTop: 8 }}>
            <label>
              Label:{" "}
              <input
                value={label}
                onChange={(e) => setLabel(sanitizeLabel(e.target.value))}
                style={{ width: 80 }}
              />
            </label>{" "}
            <button onClick={handlePick} disabled={!pendingPolygon || !LABEL_PATTERN.test(label)}>
              Pick
            </button>{" "}
            <button onClick={() => setPendingPolygon(null)} disabled={!pendingPolygon}>
              Clear selection
            </button>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4, marginBottom: 40 }}>
              Format: up to 3 capital letters followed by a number (e.g. M0)
            </div>
          </div>

          <h3>Picked curves</h3>
          {image.curves.length === 0 ? (
            <p>No curve picked yet for this position.</p>
          ) : (
            <ul style={{ listStyle: "none", padding: 0 }}>
              {image.curves.map((curve) => (
                <li
                  key={curve.label}
                  style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}
                >
                  <span>• {curve.label} ({curve.fs.length} points)</span>
                  <button onClick={() => handleDelete(curve.label)}>Delete</button>
                </li>
              ))}
            </ul>
          )}
        </>
      )}

      {folder && Object.keys(labelCounts).length > 0 && (
        <>
          <h2>Picked pseudo-sections</h2>
          <div style={{ display: "inline-flex", border: "1px solid var(--border)", borderRadius: 8 }}>
            {(["frequency", "wavelength"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setPseudoMode(m)}
                style={{
                  borderRadius: 0,
                  background: pseudoMode === m ? "var(--accent)" : "var(--surface)",
                  color: pseudoMode === m ? "var(--accent-text)" : "var(--text-muted)",
                  boxShadow: "none",
                }}
              >
                {m === "frequency" ? "Frequency" : "Wavelength"}
              </button>
            ))}
          </div>

          {Object.entries(labelCounts).map(([lbl, count]) => (
            <div
              key={lbl}
              style={{
                border: "1px solid var(--border)",
                borderRadius: 6,
                padding: 12,
                marginTop: 12,
                background: "var(--surface)",
              }}
            >
              <h3 style={{ marginTop: 0 }}>{lbl}</h3>
              <p>{count}/{xmids.length} positions picked</p>
              {pseudoSections[lbl] && <PseudoSectionCanvas section={pseudoSections[lbl]} mode={pseudoMode} />}
            </div>
          ))}
        </>
      )}
    </div>
  );
}
