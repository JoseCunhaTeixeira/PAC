import { useEffect, useState } from "react";
import { API } from "./api";
import { DispersionImageCanvas, type DispersionImage } from "./components/DispersionImageCanvas";
import { PseudoSectionCanvas, type PseudoSection } from "./components/PseudoSectionCanvas";

function noop() {}

export function VisualizationDispersion({ folder }: { folder: string }) {
  const [xmids, setXmids] = useState<number[]>([]);
  // undefined = still loading, null = confirmed missing, object = loaded
  const [images, setImages] = useState<Record<number, DispersionImage | null>>({});
  const [labelCounts, setLabelCounts] = useState<Record<string, number>>({});
  const [pseudoSections, setPseudoSections] = useState<Record<string, PseudoSection>>({});
  const [pseudoMode, setPseudoMode] = useState<"frequency" | "wavelength">("frequency");
  const [error, setError] = useState<string | null>(null);
  // Starts true (not false) so the first render of a freshly-selected folder
  // shows "Loading…" instead of flashing "No positions found" for one frame
  // before the mount effect below gets a chance to run.
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.resolve().then(() => {
      setXmids([]);
      setImages({});
      setLabelCounts({});
      setPseudoSections({});
      setError(null);
      setLoading(true);
    });

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
            // Best-effort per position: a missing image shouldn't blank out
            // the rest of the page (other positions, pseudo-sections).
            .catch(() => setImages((prev) => ({ ...prev, [xmid]: null })));
        });
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));

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
        Object.entries(data)
          .filter(([, count]) => count >= 2)
          .forEach(([labelValue]) => {
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
              // Best-effort per label: a failure here shouldn't blank out the
              // rest of the page (other labels, dispersion images).
              .catch(() => {});
          });
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [folder]);

  if (error) return <p style={{ color: "var(--accent)" }}>Error: {error}</p>;
  if (loading) return <p>Loading…</p>;
  if (xmids.length === 0) return <p>❌ No positions found.</p>;

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
          ) : images[xmid] === null ? (
            <p>❌ Dispersion data missing.</p>
          ) : (
            <p>Loading…</p>
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
              {count < 2 ? (
                <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  🛈 At least 2 picked positions are required to build a pseudo-section.
                </p>
              ) : (
                pseudoSections[lbl] && <PseudoSectionCanvas section={pseudoSections[lbl]} mode={pseudoMode} />
              )}
            </div>
          ))}
        </>
      )}
    </>
  );
}
