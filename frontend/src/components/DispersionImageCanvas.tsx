import { useEffect, useRef } from "react";
import { gistSternR } from "./colormaps";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { useContainerWidth } from "./useContainerWidth";

export interface DispersionCurve {
  label: string;
  fs: number[];
  vs: number[];
  vs_std?: number[] | null;
}

export interface DispersionImage {
  fv_map: number[][];
  fs: number[];
  vs: number[];
  type: string;
  curves: DispersionCurve[];
  lambda_min: number;
  lambda_max: number;
}

const ML = 60, MR = 16, MT = 16, MB = 38;
const PLOT_W = 640, PLOT_H = 420;
const TOTAL_W = ML + PLOT_W + MR;
const TOTAL_H = MT + PLOT_H + MB;
const FONT = CANVAS_FONT;

const CURVE_COLORS = ["#ffffff", "#ff5050", "#50ff90", "#ffd24d", "#5ab8ff", "#ff8cf0"];

export function DispersionImageCanvas({
  image,
  pendingPolygon,
  onLassoComplete,
}: {
  image: DispersionImage;
  pendingPolygon: [number, number][] | null;
  onLassoComplete: (polygon: [number, number][]) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapRef = useRef<HTMLCanvasElement | null>(null);
  const draggingRef = useRef(false);
  const dragPointsRef = useRef<[number, number][]>([]);
  const theme = useTheme();
  const palette = canvasPalette(theme);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const scale = containerWidth > 0 ? Math.min(containerWidth / TOTAL_W, 1) : 1;

  const fMin = image.fs[0], fMax = image.fs[image.fs.length - 1];
  const vMin = image.vs[0], vMax = image.vs[image.vs.length - 1];

  const xOf = (f: number) => ML + ((f - fMin) / (fMax - fMin)) * PLOT_W;
  const yOf = (v: number) => MT + PLOT_H - ((v - vMin) / (vMax - vMin)) * PLOT_H;
  const fOf = (px: number) => fMin + ((px - ML) / PLOT_W) * (fMax - fMin);
  const vOf = (py: number) => vMin + ((MT + PLOT_H - py) / PLOT_H) * (vMax - vMin);

  // build the heatmap once per image, at native (nf x nv) resolution
  useEffect(() => {
    const nf = image.fv_map.length;
    const nv = nf > 0 ? image.fv_map[0].length : 0;
    const off = document.createElement("canvas");
    off.width = nf;
    off.height = nv;
    const octx = off.getContext("2d");
    if (octx && nf > 0 && nv > 0) {
      const imgData = octx.createImageData(nf, nv);
      for (let i = 0; i < nf; i++) {
        const row = image.fv_map[i];
        let max = 0;
        for (let j = 0; j < nv; j++) max = Math.max(max, row[j]);
        if (max <= 0) max = 1;
        for (let j = 0; j < nv; j++) {
          const t = (row[j] / max) ** 2;
          const [r, g, b] = gistSternR(t);
          const y = nv - 1 - j;
          const idx = (y * nf + i) * 4;
          imgData.data[idx] = r;
          imgData.data[idx + 1] = g;
          imgData.data[idx + 2] = b;
          imgData.data[idx + 3] = 255;
        }
      }
      octx.putImageData(imgData, 0, 0);
    }
    heatmapRef.current = off;
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image, theme, scale]);

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const renderScale = dpr * scale;
    canvas.width = Math.round(TOTAL_W * renderScale);
    canvas.height = Math.round(TOTAL_H * renderScale);
    canvas.style.width = TOTAL_W * scale + "px";
    canvas.style.height = TOTAL_H * scale + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(renderScale, 0, 0, renderScale, 0, 0);
    ctx.clearRect(0, 0, TOTAL_W, TOTAL_H);

    if (heatmapRef.current) {
      ctx.drawImage(heatmapRef.current, 0, 0, heatmapRef.current.width, heatmapRef.current.height, ML, MT, PLOT_W, PLOT_H);
    }

    // array resolution bounds: v = f * lambda, clipped to the velocity axis.
    // Below lambda_min picks are spatially aliased; above lambda_max they
    // aren't resolvable by the array's aperture.
    function drawLambdaBound(lambda: number, labelText: string) {
      if (!ctx) return;
      ctx.save();
      ctx.strokeStyle = "#999999";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      let started = false;
      let lastPoint: [number, number] | null = null;
      image.fs.forEach((f) => {
        const v = f * lambda;
        if (v < vMin || v > vMax) {
          started = false;
          return;
        }
        const x = xOf(f);
        const y = yOf(v);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
        lastPoint = [x, y];
      });
      ctx.stroke();
      ctx.restore();

      if (lastPoint) {
        ctx.save();
        ctx.font = FONT;
        ctx.fillStyle = "#999999";
        ctx.textAlign = "right";
        ctx.textBaseline = "bottom";
        ctx.fillText(labelText, lastPoint[0] - 4, lastPoint[1] - 2);
        ctx.restore();
      }
    }
    drawLambdaBound(image.lambda_min, "λmin");
    drawLambdaBound(image.lambda_max, "λmax");

    // curves
    image.curves.forEach((curve, i) => {
      const color = CURVE_COLORS[i % CURVE_COLORS.length];

      // error bar whiskers, drawn under the curve line
      if (curve.vs_std) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.6;
        curve.fs.forEach((f, k) => {
          const std = curve.vs_std?.[k];
          if (std == null) return;
          const x = xOf(f);
          const yTop = yOf(curve.vs[k] + std);
          const yBot = yOf(curve.vs[k] - std);
          ctx.beginPath();
          ctx.moveTo(x, yTop);
          ctx.lineTo(x, yBot);
          ctx.stroke();
        });
        ctx.globalAlpha = 1;
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      curve.fs.forEach((f, k) => {
        const x = xOf(f);
        const y = yOf(curve.vs[k]);
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // legend
    ctx.font = FONT;
    ctx.textBaseline = "middle";
    image.curves.forEach((curve, i) => {
      const ly = MT + 14 + i * 17;
      ctx.fillStyle = CURVE_COLORS[i % CURVE_COLORS.length];
      ctx.fillRect(ML + PLOT_W - 80, ly - 4, 10, 8);
      ctx.fillStyle = palette.title;
      ctx.fillText(curve.label, ML + PLOT_W - 64, ly);
    });

    // in-progress or pending lasso
    const polygon = draggingRef.current ? dragPointsRef.current : pendingPolygon;
    if (polygon && polygon.length > 1) {
      ctx.save();
      ctx.strokeStyle = "#00e0ff";
      ctx.fillStyle = "rgba(0,224,255,0.15)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash(draggingRef.current ? [] : [4, 3]);
      ctx.beginPath();
      polygon.forEach(([f, v], k) => {
        const x = xOf(f);
        const y = yOf(v);
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // axes
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.font = FONT;
    ctx.fillStyle = palette.tick;

    ctx.beginPath();
    ctx.moveTo(ML, MT);
    ctx.lineTo(ML, MT + PLOT_H);
    ctx.lineTo(ML + PLOT_W, MT + PLOT_H);
    ctx.stroke();

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const nvTicks = 6;
    for (let i = 0; i <= nvTicks; i++) {
      const v = vMin + (i / nvTicks) * (vMax - vMin);
      const y = yOf(v);
      ctx.beginPath();
      ctx.moveTo(ML - 4, y);
      ctx.lineTo(ML, y);
      ctx.stroke();
      ctx.fillText(v.toFixed(0), ML - 7, y);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const nfTicks = 6;
    for (let i = 0; i <= nfTicks; i++) {
      const f = fMin + (i / nfTicks) * (fMax - fMin);
      const x = xOf(f);
      ctx.beginPath();
      ctx.moveTo(x, MT + PLOT_H);
      ctx.lineTo(x, MT + PLOT_H + 4);
      ctx.stroke();
      ctx.fillText(f.toFixed(0), x, MT + PLOT_H + 6);
    }

    ctx.fillStyle = palette.title;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Frequency [Hz]", ML + PLOT_W / 2, TOTAL_H - 4);
    ctx.save();
    ctx.translate(16, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Phase velocity [m/s]", 0, 0);
    ctx.restore();
  }

  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingPolygon, image, theme, scale]);

  function clampedDataPoint(offsetX: number, offsetY: number): [number, number] {
    // offsetX/Y are in displayed CSS pixels; rescale to the logical
    // coordinate space (TOTAL_W x TOTAL_H) the drawing code above uses,
    // since the canvas is displayed smaller than that when its container
    // is narrower than TOTAL_W.
    const canvas = canvasRef.current;
    const scaleX = canvas && canvas.clientWidth ? TOTAL_W / canvas.clientWidth : 1;
    const scaleY = canvas && canvas.clientHeight ? TOTAL_H / canvas.clientHeight : 1;
    const px = Math.min(Math.max(offsetX * scaleX, ML), ML + PLOT_W);
    const py = Math.min(Math.max(offsetY * scaleY, MT), MT + PLOT_H);
    return [fOf(px), vOf(py)];
  }

  function onMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    draggingRef.current = true;
    dragPointsRef.current = [clampedDataPoint(e.nativeEvent.offsetX, e.nativeEvent.offsetY)];
  }

  function onMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!draggingRef.current) return;
    dragPointsRef.current.push(clampedDataPoint(e.nativeEvent.offsetX, e.nativeEvent.offsetY));
    draw();
  }

  function onMouseUp() {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    const polygon = dragPointsRef.current;
    dragPointsRef.current = [];
    if (polygon.length >= 3) onLassoComplete(polygon);
    draw();
  }

  return (
    <div ref={containerRef} style={{ width: "100%", maxWidth: TOTAL_W }}>
      <canvas
        ref={canvasRef}
        style={{ cursor: "crosshair", touchAction: "none", display: "block" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      />
    </div>
  );
}
