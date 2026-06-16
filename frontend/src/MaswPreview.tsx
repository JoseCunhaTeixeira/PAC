import { useEffect, useState } from "react";
import { API, type Acquisition, type Masw } from "./api";

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
}

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

const vertBar = (cx, cy, h = 12, w = 2) =>
  `${cx - w/2},${cy - h/2} ${cx + w/2},${cy - h/2} ${cx + w/2},${cy + h/2} ${cx - w/2},${cy + h/2}`;

export function MaswPreview({
  acquisition,
  masw,
}: {
  acquisition: Acquisition;
  masw: Masw;
}) {
  const [windows, setWindows] = useState<WindowSummary[]>([]);
  const [invalid, setInvalid] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setError(null);
      fetch(`${API}/windows`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ acquisition_params: acquisition, masw_params: masw }),
      })
        .then((res) => {
          if (res.status === 422) {
            setInvalid(true);
            return null;
          }
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data: WindowSummary[] | null) => {
          if (data) {
            setWindows(data);
            setInvalid(false);
          }
        })
        .catch((err) => setError(String(err)));
    }, 300);
    return () => clearTimeout(timer);
  }, [acquisition, masw]);

  const xmids = windows.map((w) => w.xmid);

  const W = 720;
  const H = 100;
  const left = 55;
  const right = 20;

  const allX = [
    ...acquisition.receiver_positions,
    ...acquisition.source_positions,
    ...xmids,
  ];
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const pad = (xMax - xMin) * 0.05 || 1;
  const domainMin = xMin - pad;
  const domainMax = xMax + pad;

  const scaleX = (x: number) =>
    left + ((x - domainMin) / (domainMax - domainMin)) * (W - left - right);

  const yS = 30;
  const ySen = 60;
  const yM = 90;
  const yAxis = 100;

  const hoverProps = (x: number, cx: number, cy: number) => ({
    cursor: "pointer",
    onMouseEnter: () => setHover({ cx, cy, label: `${x} m` }),
    onMouseLeave: () => setHover(null),
  });

  return (
    <div>
      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}
      <p>
        {invalid ? (
          <em>Adjust window parameters…</em>
        ) : (
          <>
            <strong>{windows.length}</strong> positions to compute
          </>
        )}
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
        <text x={8} y={yS + 4} fontSize={11}>Sources</text>
        <text x={8} y={ySen + 4} fontSize={11}>Sensors</text>
        <text x={8} y={yM + 4} fontSize={11}>Positions</text>

        {acquisition.source_positions.map((x, i) => {
          const cx = scaleX(x);
          return (
            <polygon key={`src-${i}`} points={star(cx, yS)} fill="#2ca02c" {...hoverProps(x, cx, yS)} />
          );
        })}
        {acquisition.receiver_positions.map((x, i) => {
          const cx = scaleX(x);
          return (
            <polygon key={`sen-${i}`} points={triDown(cx, ySen)} fill="#1f77b4" {...hoverProps(x, cx, ySen)} />
          );
        })}
        {xmids.map((x, i) => {
          const cx = scaleX(x);
          return (
            <polygon key={`mid-${i}`} points={vertBar(cx, yM)} fill="#d62728" {...hoverProps(x, cx, yM)} />
          );
        })}

        {/* <text x={left} y={yAxis + 18} fontSize={10}>{domainMin.toFixed(1)}</text> */}
        <text x={W / 2} y={yAxis + 18} fontSize={11} textAnchor="middle">Position [m]</text>
        {/* <text x={W - right} y={yAxis + 18} fontSize={10} textAnchor="end">{domainMax.toFixed(1)}</text> */}

        {hover && (
          <text
            x={hover.cx}
            y={hover.cy - 12}
            fontSize={12}
            textAnchor="middle"
            fill="#111"
            stroke="white"
            strokeWidth={3}
            paintOrder="stroke"
          >
            {hover.label}
          </text>
        )}
      </svg>
    </div>
  );
}