import { useEffect, useState } from "react";
import { API } from "./api";
import { DispersionImageCanvas, type DispersionImage } from "./components/DispersionImageCanvas";
import { PseudoSectionCanvas, type PseudoSection } from "./components/PseudoSectionCanvas";

function noop() {}

export function VisualizationDispersion({ folder }: { folder: string }) {
  const [xmids, setXmids] = useState<number[]>([]);
  const [images, setImages] = useState<Record<number, DispersionImage>>({});
  const [labelCounts, setLabelCounts] = useState<Record<string, number>>({});
  const [pseudoSections, setPseudoSections] = useState<Record<string, PseudoSection>>({});
  const [pseudoMode, setPseudoMode] = useState<"frequency" | "wavelength">("frequency");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setXmids([]);
    setImages({});
    setLabelCounts({});
    setPseudoSections({});
    setError(null);

    fetch(`${API}/xmids/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: number[]) => {
        setXmids(data);
        data.forEach((xmid) => {
          fetch(`${API}/dispersion_images/${encodeURIComponent(folder)}/${xmid}`)
            .then(async (res) => {
              if (!res.ok) {
                const body = await res.json().catch(() => null);
                throw new Error(body?.detail ?? `HTTP ${res.status}`);
              }
              return res.json();
            })
            .then((image: DispersionImage) =>
              setImages((prev) => ({ ...prev, [xmid]: image })),
            )
            .catch((err) => setError(err instanceof Error ? err.message : String(err)));
        });
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));

    fetch(`${API}/dispersion_image_labels/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: Record<string, number>) => {
        setLabelCounts(data);
        Object.keys(data).forEach((labelValue) => {
          fetch(
            `${API}/dispersion_pseudo_section/${encodeURIComponent(folder)}/${encodeURIComponent(labelValue)}`,
          )
            .then(async (res) => {
              if (!res.ok) {
                const body = await res.json().catch(() => null);
                throw new Error(body?.detail ?? `HTTP ${res.status}`);
              }
              return res.json();
            })
            .then((section: PseudoSection) =>
              setPseudoSections((prev) => ({ ...prev, [labelValue]: section })),
            )
            .catch((err) => setError(err instanceof Error ? err.message : String(err)));
        });
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [folder]);

  if (error) return <p style={{ color: "var(--accent)" }}>Error: {error}</p>;
  if (xmids.length === 0) return null;

  return (
    <>
      <h2>Dispersion images</h2>
      {xmids.map((xmid) => (
        <div key={xmid} style={{ marginTop: 24 }}>
          <h3>Position: {xmid.toFixed(2)} m</h3>
          {images[xmid] ? (
            <DispersionImageCanvas
              image={images[xmid]}
              pendingPolygon={null}
              onLassoComplete={noop}
            />
          ) : (
            <p>📁 Dispersion data missing.</p>
          )}
        </div>
      ))}

      {Object.keys(labelCounts).length > 0 && (
        <>
          <h2>Pseudo-section</h2>
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
    </>
  );
}
