import { useEffect, useRef } from "react";
import { API } from "../../api";
import { thumbHeatmap } from "../dispersionHeatmap";
import { num } from "./format";
import type { OptionImage, Tried } from "./types";
import { useJson } from "./useJson";

// An option the agent compared, its dispersion images at the windows it was judged on, small, each
// with where it is along the line: the images alone, as the agent compared them, no pick drawn.

function draw(canvas: HTMLCanvasElement, one: OptionImage) {
  const { width, height } = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx || width === 0) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.drawImage(thumbHeatmap(one.image), 0, 0, width, height);
}

function Thumb({ one }: { one: OptionImage }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (canvasRef.current) draw(canvasRef.current, one);
  }, [one]);
  return (
    <figure className="viz-thumb">
      <canvas ref={canvasRef} />
      <figcaption>x = {num(one.xmid)} m</figcaption>
    </figure>
  );
}

/** The option's dispersion images; where none can be shown, said so. */
export function OptionImages({ folder, option }: { folder: string; option: Tried }) {
  const url =
    option.images && option.trial
      ? `${API}/quality/option_images/${encodeURIComponent(folder)}/${option.trial}?line=${option.kept}`
      : null;
  const images = useJson<OptionImage[]>(url);
  if (url === null || images.error || (images.data && images.data.length === 0)) {
    return <div className="viz-thumbs-loading">No dispersion images kept for this option</div>;
  }
  if (!images.data) return <div className="viz-thumbs-loading">Loading the dispersion images…</div>;
  return (
    <div className="viz-option-thumbs">
      {images.data.map((one) => (
        <Thumb key={one.xmid} one={one} />
      ))}
    </div>
  );
}
