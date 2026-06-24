import { useEffect, useState } from "react";
import { API, type Acquisition, type Masw } from "../api";

interface WindowSummary {
  xmid: number;
  start_index: number;
  end_index: number;
  n_shots: number;
}

interface Hover {
  cx: number;
  cy: number;
  label: string;
  color: string;
}

const RECEIVER_COLOR = "#1f77b4";
const SOURCE_COLOR = "#2ca02c";
const XMID_COLOR = "#d62728";

// marker shapes as SVG polygon point strings
const triDown = (cx: number, cy: number, s = 5) =>
  `${cx - s},${cy - s} ${cx + s},${cy - s} ${cx},${cy + s}`;

const star = (cx: number, cy: number, outer = 7, inner = 3, n = 5) => {
  const pts: string[] = [];
  for (let i = 0; i < n * 2; i++) {
    const r = i % 2 === 0 ? outer : inner;
    const a = (Math.PI * i) / n - Math.PI / 2;
    pts.push(`${(cx + r * Math.cos(a)).toFixed(2)},${(cy + r * Math.sin(a)).toFixed(2)}`);
  }
  return pts.join(" ");
};

// taller than the receiver triangle (2*s = 10px) so its ends stay visible
// when a mid position lands on top of a receiver
const vertBar = (cx: number, cy: number, h = 26, w = 2) =>
  `${cx - w / 2},${cy - h / 2} ${cx + w / 2},${cy - h / 2} ${cx + w / 2},${cy + h / 2} ${cx - w / 2},${cy + h / 2}`;

// linear interpolation of elevation along the receiver topography profile;
// clamped at the ends since sources can sit outside the receiver span
function interpZ(profile: readonly (readonly [number, number])[], x: number): number {
  if (x <= profile[0][0]) return profile[0][1];
  if (x >= profile[profile.length - 1][0]) return profile[profile.length - 1][1];
  for (let i = 1; i < profile.length; i++) {
    const [x1, z1] = profile[i];
    if (x <= x1) {
      const [x0, z0] = profile[i - 1];
      const t = (x - x0) / (x1 - x0);
      return z0 + t * (z1 - z0);
    }
  }
  return profile[profile.length - 1][1];
}

export function MaswPreview({
  acquisition,
  masw,
  onCount,
  showSources = true,
}: {
  acquisition: Acquisition;
  masw: Masw;
  onCount?: (n: number) => void;
  showSources?: boolean;
}) {
  const [windows, setWindows] = useState<WindowSummary[]>([]);
  const [invalid, setInvalid] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    const timer = setTimeout(() => {
      setError(null);
      fetch(`${API}/windows`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ acquisition_params: acquisition, masw_params: masw }),
      })
        .then(async (res) => {
          if (res.status === 422) {
            setInvalid(true);
            return null;
          }
          if (!res.ok) {
            const body = await res.json().catch(() => null);
            throw new Error(body?.detail ?? `HTTP ${res.status}`);
          }
          return res.json();
        })
        .then((data: WindowSummary[] | null) => {
          if (data) {
            setWindows(data);
            setInvalid(false);
            onCount?.(data.length);
          }
        })
        .catch((err) => setError(err instanceof Error ? err.message : String(err)));
    }, 300);
    return () => clearTimeout(timer);
  }, [acquisition, masw]);

  const xmids = windows.map((w) => w.xmid);

  const W = 720;
  const lineH = 140;
  const H = lineH;
  const left = 20;
  const right = 20;

  const allX = [
    ...acquisition.receiver_positions.map((p) => p[0]),
    ...(showSources ? acquisition.source_positions.map((p) => p[0]) : []),
    ...xmids,
  ];
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const pad = (xMax - xMin) * 0.05 || 1;
  const domainMin = xMin - pad;
  const domainMax = xMax + pad;

  const scaleX = (x: number) =>
    left + ((x - domainMin) / (domainMax - domainMin)) * (W - left - right);

  const allZ = [
    ...acquisition.receiver_positions.map((p) => p[1]),
    ...(showSources ? acquisition.source_positions.map((p) => p[1]) : []),
  ];
  const zMin = Math.min(...allZ);
  const zMax = Math.max(...allZ);
  const zSpan = zMax - zMin || 1;
  const marginTop = 24;
  const marginBottom = 24;
  const scaleZ = (z: number) =>
    lineH - marginBottom - ((z - zMin) / zSpan) * (lineH - marginTop - marginBottom);

  const receiverProfile = acquisition.receiver_positions
    .map((p) => [p[0], p[1]] as const)
    .sort((a, b) => a[0] - b[0]);

  const hoverProps = (x: number, z: number, cx: number, cy: number, color: string) => ({
    cursor: "pointer",
    onMouseEnter: () =>
      setHover({ cx, cy, color, label: `x: ${x.toFixed(2)} m   z: ${z.toFixed(2)} m` }),
    onMouseLeave: () => setHover(null),
  });

  return (
    <div>
      {error && <p style={{ color: "var(--accent)" }}>Error: {error}</p>}
      <p>
        {invalid ? (
          <em>✘ Invalid window parameters…</em>
        ) : (
          <>
            ➜ <strong>{windows.length}</strong> mid positions to compute
          </>
        )}
      </p>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
        <div style={{ display: "flex", gap: 20, fontSize: 12, color: "var(--text-muted)" }}>
          <span><span style={{ color: RECEIVER_COLOR }}>▼</span> Receivers</span>
          {showSources && <span><span style={{ color: SOURCE_COLOR }}>★</span> Sources</span>}
          <span><span style={{ color: XMID_COLOR }}>│</span> Mid positions</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.max(1, +(z - 0.5).toFixed(1)))}
            disabled={zoom <= 1}
            style={{ width: 22, height: 22, padding: 0, fontSize: 12, lineHeight: 1 }}
          >
            −
          </button>
          <span style={{ fontSize: 12, color: "var(--text-muted)", minWidth: 36, textAlign: "center" }}>
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.min(4, +(z + 0.5).toFixed(1)))}
            disabled={zoom >= 4}
            style={{ width: 22, height: 22, padding: 0, fontSize: 12, lineHeight: 1 }}
          >
            +
          </button>
        </div>
      </div>

      <div style={{ overflow: "auto" }}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          style={{ width: `${zoom * 100}%`, minWidth: W, height: "auto", display: "block" }}
        >
        <polyline
          points={receiverProfile.map(([x, z]) => `${scaleX(x)},${scaleZ(z)}`).join(" ")}
          fill="none"
          stroke={RECEIVER_COLOR}
          strokeWidth={1.5}
          opacity={0.5}
        />

        {xmids.map((x, i) => {
          const cx = scaleX(x);
          const z = interpZ(receiverProfile, x);
          const cy = scaleZ(z);
          return (
            <polygon
              key={`mid-${i}`}
              points={vertBar(cx, cy)}
              fill={XMID_COLOR}
              {...hoverProps(x, z, cx, cy, XMID_COLOR)}
            />
          );
        })}

        {acquisition.receiver_positions.map((p, i) => {
          const [x, z] = p;
          const cx = scaleX(x);
          const cy = scaleZ(z);
          return (
            <polygon
              key={`rx-${i}`}
              points={triDown(cx, cy)}
              fill={RECEIVER_COLOR}
              {...hoverProps(x, z, cx, cy, RECEIVER_COLOR)}
            />
          );
        })}

        {showSources && acquisition.source_positions.map((p, i) => {
          const [x, z] = p;
          const cx = scaleX(x);
          const cy = scaleZ(z);
          return (
            <polygon
              key={`src-${i}`}
              points={star(cx, cy)}
              fill={SOURCE_COLOR}
              {...hoverProps(x, z, cx, cy, SOURCE_COLOR)}
            />
          );
        })}

        {hover && (
          <text
            x={hover.cx}
            y={hover.cy - 12}
            fontSize={12}
            textAnchor="middle"
            style={{ fill: hover.color, stroke: "var(--bg)" }}
            strokeWidth={3}
            paintOrder="stroke"
          >
            {hover.label}
          </text>
        )}
      </svg>
      </div>
    </div>
  );
}
