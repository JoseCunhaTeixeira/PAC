import { useEffect, useRef } from "react";
import { API } from "../../api";
import type { DispersionImage } from "../DispersionImageCanvas";
import { dispersionHeatmap } from "../dispersionHeatmap";
import { num } from "./format";
import type { LineWindow } from "./types";
import { useJson } from "./useJson";

// A run's dispersion images, small, at a few places along its line: each window nearest the
// middle of a quarter of the line, as the assistant's search shows its combinations; each image
// scaled by frequency as PAC draws it, its picked curve over it.

const PLACES = 4;
const W = 220;
const H = 150;

/** The windows nearest the middles of PLACES equal parts of the line, left to right, each once. */
function windowsAlong(receivers: number[], windows: LineWindow[]): LineWindow[] {
  if (receivers.length === 0 || windows.length === 0) return [];
  const first = Math.min(...receivers);
  const last = Math.max(...receivers);
  const found = new Map<string, LineWindow>();
  for (let i = 0; i < PLACES; i++) {
    const place = first + ((last - first) * (2 * i + 1)) / (2 * PLACES);
    const nearest = windows.reduce((best, one) =>
      Math.abs(one.xmid - place) < Math.abs(best.xmid - place) ? one : best,
    );
    found.set(nearest.key, nearest);
  }
  return [...found.values()].sort((a, b) => a.xmid - b.xmid);
}

function draw(canvas: HTMLCanvasElement, image: DispersionImage) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx || image.fs.length < 2 || image.vs.length < 2) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.drawImage(dispersionHeatmap(image.fv_map), 0, 0, W, H);
  const [f0, f1] = [image.fs[0], image.fs[image.fs.length - 1]];
  const [v0, v1] = [image.vs[0], image.vs[image.vs.length - 1]];
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  for (const curve of image.curves) {
    ctx.beginPath();
    curve.fs.forEach((f, i) => {
      const x = ((f - f0) / (f1 - f0)) * W;
      const y = H - ((curve.vs[i] - v0) / (v1 - v0)) * H;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

function Thumb({ folder, window }: { folder: string; window: LineWindow }) {
  const image = useJson<DispersionImage>(`${API}/dispersion_images/${encodeURIComponent(folder)}/${window.xmid}`);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const data = image.data;
  useEffect(() => {
    if (canvasRef.current && data) draw(canvasRef.current, data);
  }, [data]);
  const ranges = data
    ? `${num(data.fs[0])}–${num(data.fs[data.fs.length - 1])} Hz across\n${num(data.vs[0])}–${num(data.vs[data.vs.length - 1])} m/s up`
    : undefined;
  return (
    <figure className="viz-thumb" data-tip={ranges}>
      {image.error ? (
        <div className="viz-thumb-empty">No dispersion image</div>
      ) : (
        <canvas ref={canvasRef} />
      )}
      <figcaption>x = {num(window.xmid)} m</figcaption>
    </figure>
  );
}

/** The run's dispersion images at PLACES places along its line. */
export function DispersionThumbs({ folder, receivers, windows }: { folder: string; receivers: number[]; windows: LineWindow[] }) {
  const shown = windowsAlong(receivers, windows);
  if (shown.length === 0) return null;
  return (
    <div className="viz-thumbs">
      {shown.map((one) => (
        <Thumb key={one.key} folder={folder} window={one} />
      ))}
    </div>
  );
}
