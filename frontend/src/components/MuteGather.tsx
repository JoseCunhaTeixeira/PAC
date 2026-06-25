import { useEffect, useMemo, useRef, useState } from "react";
import { API, type Acquisition, type Muting } from "../api";
import { HoverTooltip } from "./HoverTooltip";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { useCanvasHover } from "./useCanvasHover";

interface Gather {
  dt: number;
  n_samples: number;
  traces: number[][];
}

// layout in logical pixels -- shared between the draw effect and the hover
// hit-testing below, so they always agree on where the plot area is.
const ML = 50, MR = 12, MT = 12, MB = 34;
const plotW = 686, plotH = 320;
const totalW = ML + plotW + MR;
const totalH = MT + plotH + MB;

// round-number step for axis ticks
function niceStep(raw: number) {
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  const nice = n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10;
  return nice * mag;
}
function ticks(max: number, count = 5): number[] {
  if (!(max > 0)) return [0];
  const step = niceStep(max / count);
  const out: number[] = [];
  for (let v = 0; v <= max + step * 1e-6; v += step) out.push(+v.toFixed(6));
  return out;
}

export function MuteGather({
  acquisition,
  muting,
  file: fileProp,
  norm = "trace",
}: {
  acquisition: Acquisition;
  muting?: Muting;
  // Controlled file selection: when omitted, the component owns its own
  // selector (the original config-preview behavior, one file at a time).
  file?: string;
  norm?: "trace" | "global";
}) {
  const folder =
    acquisition.folder_path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";

  const [internalFile, setInternalFile] = useState(acquisition.files[0] ?? "");
  const file = fileProp ?? internalFile;
  const [gather, setGather] = useState<Gather | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const palette = canvasPalette(theme);
  const { pos: hoverPos, onMouseMove, onMouseLeave } = useCanvasHover(1);

  // true offset along the (x, z) topography profile (matches sigpipe's
  // Coordinate.distance_to with y=0), so a sloped line between source and
  // receiver gives a longer offset than the flat horizontal distance would.
  const offsets = useMemo(() => {
    const fi = acquisition.files.indexOf(file);
    const src = acquisition.source_positions[fi] ?? [0, 0];
    return acquisition.receiver_positions.map((rp) =>
      Math.sqrt((rp[0] - src[0]) ** 2 + (rp[1] - src[1]) ** 2)
    );
  }, [acquisition, file]);

  const hover = useMemo(() => {
    if (!hoverPos || !gather) return null;
    if (
      hoverPos.x < ML || hoverPos.x > ML + plotW ||
      hoverPos.y < MT || hoverPos.y > MT + plotH
    ) {
      return null;
    }

    const nt = gather.traces.length;
    const Tmax = gather.n_samples * gather.dt;
    const spacing = plotW / nt;
    const c = Math.max(0, Math.min(nt - 1, Math.floor((hoverPos.x - ML) / spacing)));
    const time = ((hoverPos.y - MT) / plotH) * Tmax;

    return {
      px: hoverPos.x,
      py: hoverPos.y,
      lines: [`Offset: ${offsets[c].toFixed(2)} m`, `Time: ${time.toFixed(3)} s`],
    };
  }, [hoverPos, gather, offsets]);

  useEffect(() => {
    if (fileProp === undefined) setInternalFile(acquisition.files[0] ?? "");
  }, [acquisition, fileProp]);

  useEffect(() => {
    if (!file) return;
    setError(null);
    fetch(`${API}/gather/${encodeURIComponent(folder)}/${encodeURIComponent(file)}?norm=${norm}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: Gather) => setGather(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [file, folder, norm]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !gather) return;

    const nt = gather.traces.length;
    const ns = gather.n_samples;
    const dt = gather.dt;
    const Tmax = ns * dt;

    // crisp on high-DPI screens
    const dpr = window.devicePixelRatio || 1;
    canvas.width = totalW * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = totalW + "px";
    canvas.style.height = totalH + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, totalW, totalH);

    // coordinate mappings
    const yOfSample = (r: number) => MT + (r / ns) * plotH;
    const yOfTime = (t: number) => MT + (t / Tmax) * plotH;
    const spacing = plotW / nt;
    const xOf = (c: number) => ML + (c + 0.5) * spacing;
    const amp = spacing * 0.45; // wiggle excursion

    // white plot background
    ctx.fillStyle = "#fff";
    ctx.fillRect(ML, MT, plotW, plotH);

    // clip wiggles + mute to the plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, MT, plotW, plotH);
    ctx.clip();

    ctx.fillStyle = "#222";
    ctx.strokeStyle = "#222";
    ctx.lineWidth = 1;
    for (let c = 0; c < nt; c++) {
      const cx = xOf(c);
      const trace = gather.traces[c];
      // filled positive lobes
      ctx.beginPath();
      ctx.moveTo(cx, yOfSample(0));
      for (let r = 0; r < ns; r++) {
        const x = cx + trace[r] * amp;
        ctx.lineTo(x > cx ? x : cx, yOfSample(r));
      }
      ctx.lineTo(cx, yOfSample(ns - 1));
      ctx.fill();
      // full wiggle line
      ctx.beginPath();
      for (let r = 0; r < ns; r++) {
        const x = cx + trace[r] * amp;
        const y = yOfSample(r);
        if (r === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // mute overlay (only when a muting config was passed in)
    if (muting) {
      const { tmin, tmax, vmin, vmax, taper } = muting;
      const clampR = (r: number) => Math.max(0, Math.min(ns, r));
      for (let c = 0; c < nt; c++) {
        const off = offsets[c];
        const rFast = vmax > 0 ? off / vmax / dt : 0;
        const rSlow = vmin > 0 ? off / vmin / dt : Infinity;
        const rTop = clampR(Math.max(tmin / dt, rFast));
        const rBot = clampR(Math.min(tmax / dt, rSlow));
        const xL = xOf(c) - spacing / 2;
        ctx.fillStyle = "rgba(110,110,120,0.55)";
        ctx.fillRect(xL, yOfSample(0), spacing, yOfSample(rTop) - yOfSample(0));
        ctx.fillRect(xL, yOfSample(rBot), spacing, yOfSample(ns) - yOfSample(rBot));
        if (taper > 0 && rBot > rTop) {
          const tt = Math.min(taper, rBot - rTop);
          ctx.fillStyle = "rgba(110,110,120,0.3)";
          ctx.fillRect(xL, yOfSample(rTop), spacing, yOfSample(rTop + tt) - yOfSample(rTop));
          ctx.fillRect(xL, yOfSample(rBot - tt), spacing, yOfSample(rBot) - yOfSample(rBot - tt));
        }
      }
    }
    ctx.restore();

    // axes
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.font = CANVAS_FONT;
    ctx.fillStyle = palette.tick;

    // time axis (y)
    ctx.beginPath();
    ctx.moveTo(ML, MT);
    ctx.lineTo(ML, MT + plotH);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of ticks(Tmax, 5)) {
      const y = yOfTime(t);
      ctx.beginPath();
      ctx.moveTo(ML - 4, y);
      ctx.lineTo(ML, y);
      ctx.stroke();
      ctx.fillText(t.toFixed(2), ML - 7, y);
    }

    // offset axis (x)
    ctx.beginPath();
    ctx.moveTo(ML, MT + plotH);
    ctx.lineTo(ML + plotW, MT + plotH);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const nLabels = 6;
    for (let i = 0; i <= nLabels; i++) {
      const c = Math.round((i / nLabels) * (nt - 1));
      const x = xOf(c);
      ctx.beginPath();
      ctx.moveTo(x, MT + plotH);
      ctx.lineTo(x, MT + plotH + 4);
      ctx.stroke();
      ctx.fillText(offsets[c].toFixed(1), x, MT + plotH + 6);
    }

    // axis titles
    ctx.fillStyle = palette.title;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Offset [m]", ML + plotW / 2, totalH - 4);
    ctx.save();
    ctx.translate(12, MT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Time [s]", 0, 0);
    ctx.restore();
  }, [gather, muting, offsets, theme]);

  return (
    <div>
      {fileProp === undefined && (
        <label>
          Preview shot file:{" "}
          <select value={file} onChange={(e) => setInternalFile(e.target.value)}>
            {acquisition.files.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </label>
      )}
      {error && <p style={{ color: "var(--accent)" }}>Error: {error}</p>}
      <div style={{ marginTop: 8, overflowX: "auto" }}>
        <div style={{ position: "relative", display: "inline-block" }}>
          <canvas
            ref={canvasRef}
            onMouseMove={onMouseMove}
            onMouseLeave={onMouseLeave}
          />
          {hover && <HoverTooltip x={hover.px} y={hover.py} lines={hover.lines} />}
        </div>
      </div>
    </div>
  );
}