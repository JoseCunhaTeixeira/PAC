import { useEffect, useRef, useState } from "react";
import { API, type Acquisition, type Muting } from "../api";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";

interface Gather {
  dt: number;
  n_samples: number;
  traces: number[][];
}

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
}: {
  acquisition: Acquisition;
  muting: Muting;
}) {
  const folder =
    acquisition.folder_path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";

  const [file, setFile] = useState(acquisition.files[0] ?? "");
  const [gather, setGather] = useState<Gather | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const palette = canvasPalette(theme);

  useEffect(() => {
    setFile(acquisition.files[0] ?? "");
  }, [acquisition]);

  useEffect(() => {
    if (!file) return;
    setError(null);
    fetch(`${API}/gather/${encodeURIComponent(folder)}/${encodeURIComponent(file)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: Gather) => setGather(data))
      .catch((err) => setError(String(err)));
  }, [file, folder]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !gather) return;

    const nt = gather.traces.length;
    const ns = gather.n_samples;
    const dt = gather.dt;
    const Tmax = ns * dt;

    // layout in logical pixels
    const ML = 50, MR = 12, MT = 12, MB = 34;
    const plotW = 686, plotH = 320;
    const totalW = ML + plotW + MR;
    const totalH = MT + plotH + MB;

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

    const fi = acquisition.files.indexOf(file);
    const src = acquisition.source_positions[fi] ?? 0;
    const offsets = acquisition.receiver_positions.map((rp) => Math.abs(rp - src));

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

    // mute overlay
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
  }, [gather, muting, acquisition, file, theme]);

  return (
    <div>
      <label>
        Preview shot file:{" "}
        <select value={file} onChange={(e) => setFile(e.target.value)}>
          {acquisition.files.map((f) => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
      </label>
      {error && <p style={{ color: "var(--accent)" }}>Error: {error}</p>}
      <div style={{ marginTop: 8, overflowX: "auto" }}>
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}