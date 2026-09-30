import { useEffect, useMemo, useRef } from "react";
import { gistSternR } from "./colormaps";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { nearestIndex, useCanvasHover } from "./useCanvasHover";
import { useContainerWidth } from "./useContainerWidth";
import { CLICK_PX, evenTicks, tickDecimals, useZoom, visibleCells, type PlotRect } from "./useZoom";
import type { DragTool } from "./plotBox";
import { ZoomSelection } from "./ZoomOverlay";
import { HoverTooltip } from "./HoverTooltip";
import type { Tip } from "./tips";

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
  lambda_min: number | null;
  lambda_max: number | null;
}

// The top margin holds the colour bar.
const ML = 60, MR = 16, MT = 30, MB = 38;
const LEGEND_W = 120, LEGEND_H = 8;
const BASE_W = 716; // the drawing's width until its card is measured
const FONT = CANVAS_FONT;

const CURVE_COLORS = ["#ffffff", "#ff5050", "#50ff90", "#ffd24d", "#5ab8ff", "#ff8cf0"];

export function DispersionImageCanvas({
  image,
  pendingPolygon = null,
  onLassoComplete,
  dragMode = "pan",
}: {
  image: DispersionImage;
  pendingPolygon?: [number, number][] | null;
  // Without it the image has no lasso: a drag only zooms.
  onLassoComplete?: (polygon: [number, number][]) => void;
  /** What a drag does, with a lasso: the picking's tools (a box's tool in Visualization). */
  dragMode?: DragTool;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapRef = useRef<HTMLCanvasElement | null>(null);
  const draggingRef = useRef(false);
  const dragPointsRef = useRef<[number, number][]>([]);
  const theme = useTheme();
  const palette = canvasPalette(theme);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  // As wide as its card, the text at its own size; the height follows, within bounds.
  const TOTAL_W = Math.max(420, Math.round(containerWidth || BASE_W));
  const PLOT_W = TOTAL_W - ML - MR;
  const PLOT_H = Math.round(Math.min(540, Math.max(360, PLOT_W * 0.6)));
  const TOTAL_H = MT + PLOT_H + MB;
  // The axis strips take the whole margins under and left of the plot.
  const PLOTS: PlotRect[] = [{ left: ML, top: MT, width: PLOT_W, height: PLOT_H, xAxis: MB, yAxis: ML }];
  const scale = 1;
  const lasso = dragMode === "lasso" && onLassoComplete !== undefined;
  // The pointer: the zoom's cursor, and what the image holds under it.
  const { pos: hoverPos, onMouseMove: onHoverMove, onMouseLeave: onHoverLeave } = useCanvasHover(scale);

  const fMin = image.fs[0], fMax = image.fs[image.fs.length - 1];
  const vMin = image.vs[0], vMax = image.vs[image.vs.length - 1];

  const zoom = useZoom({
    extent: { x: [fMin, fMax], y: [vMin, vMax] },
    plots: PLOTS,
    width: TOTAL_W,
    height: TOTAL_H,
    // With a lasso, the picking's tools choose (lasso, zoom, hand); else its box's.
    tool: onLassoComplete !== undefined ? (dragMode === "lasso" ? "pan" : dragMode) : undefined,
    canvas: canvasRef,
  });
  // The frequencies and velocities on show: the whole image, or the zoom.
  const [f0, f1] = zoom.view.x;
  const [v0, v1] = zoom.view.y;

  const xOf = (f: number) => ML + ((f - f0) / (f1 - f0)) * PLOT_W;
  const yOf = (v: number) => MT + PLOT_H - ((v - v0) / (v1 - v0)) * PLOT_H;
  const fOf = (px: number) => f0 + ((px - ML) / PLOT_W) * (f1 - f0);
  const vOf = (py: number) => v0 + ((MT + PLOT_H - py) / PLOT_H) * (v1 - v0);

  // The colour bar's image: the amplitude (normalized per frequency) from 0 on the left, in the
  // colours the heatmap gives it (its square's, below).
  const legend = useMemo(() => {
    const off = document.createElement("canvas");
    off.width = 256;
    off.height = 1;
    const octx = off.getContext("2d");
    if (octx) {
      const data = octx.createImageData(256, 1);
      for (let px = 0; px < 256; px++) {
        const [r, g, b] = gistSternR((px / 255) ** 2);
        data.data.set([r, g, b, 255], px * 4);
      }
      octx.putImageData(data, 0, 0);
    }
    return off;
  }, []);

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
  }, [image, theme, scale, PLOT_W, PLOT_H]);

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

    // Everything in data units is clipped to the plot, which a zoom cuts through.
    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, MT, PLOT_W, PLOT_H);
    ctx.clip();

    const heatmap = heatmapRef.current;
    if (heatmap && heatmap.width > 0 && heatmap.height > 0) {
      // Only the heatmap's cells on show, scaled to the plot: its columns
      // run from fMin to fMax, its rows from vMax (top) down to vMin.
      const nf = heatmap.width, nv = heatmap.height;
      const [c0, c1] = visibleCells(nf, fMin, fMax, f0, f1);
      const [r0, r1] = visibleCells(nv, vMax, vMin, v0, v1);
      if (c1 > c0 && r1 > r0) {
        const fEdge = (c: number) => fMin + (c / nf) * (fMax - fMin);
        const vEdge = (r: number) => vMax - (r / nv) * (vMax - vMin);
        const x0 = xOf(fEdge(c0)), x1 = xOf(fEdge(c1));
        const y0 = yOf(vEdge(r0)), y1 = yOf(vEdge(r1));
        ctx.drawImage(heatmap, c0, r0, c1 - c0, r1 - r0, x0, y0, x1 - x0, y1 - y0);
      }
    }

    // Where the checks' flags start: v = f * lambda, clipped to the velocity axis. Below
    // lambda_min (two spacings) picks may be aliased; above lambda_max (three window lengths)
    // they lie beyond the window's reach.
    const lambdaLabels: { x: number; y: number; text: string }[] = [];
    function drawLambdaBound(lambda: number | null, labelText: string) {
      if (!ctx || lambda === null) return;
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
        // labelled at its last point on show
        if (f >= f0 && f <= f1 && v >= v0 && v <= v1) lastPoint = [x, y];
      });
      ctx.stroke();
      ctx.restore();
      if (lastPoint) lambdaLabels.push({ x: lastPoint[0], y: lastPoint[1], text: labelText });
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
    ctx.restore();

    lambdaLabels.forEach(({ x, y, text }) => {
      ctx.save();
      ctx.font = FONT;
      ctx.fillStyle = "#999999";
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      ctx.fillText(text, x - 4, y - 2);
      ctx.restore();
    });

    // The colour bar above the image, on its right.
    {
      const legendX = ML + PLOT_W - LEGEND_W - 12;
      const legendY = (MT - LEGEND_H) / 2 - 2;
      ctx.drawImage(legend, 0, 0, legend.width, 1, legendX, legendY, LEGEND_W, LEGEND_H);
      ctx.strokeStyle = palette.axis;
      ctx.lineWidth = 1;
      ctx.strokeRect(legendX, legendY, LEGEND_W, LEGEND_H);
      ctx.font = FONT;
      ctx.textBaseline = "middle";
      ctx.fillStyle = palette.tick;
      ctx.textAlign = "right";
      ctx.fillText("0", legendX - 5, legendY + LEGEND_H / 2);
      ctx.textAlign = "left";
      ctx.fillText("1", legendX + LEGEND_W + 5, legendY + LEGEND_H / 2);
      ctx.fillStyle = palette.title;
      ctx.textAlign = "right";
      ctx.fillText("Amplitude (normalized per frequency)", legendX - 18, legendY + LEGEND_H / 2);
      ctx.textAlign = "left";
    }

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
    const vDecimals = tickDecimals((v1 - v0) / nvTicks);
    for (const v of evenTicks(v0, v1, nvTicks)) {
      const y = yOf(v);
      ctx.beginPath();
      ctx.moveTo(ML - 4, y);
      ctx.lineTo(ML, y);
      ctx.stroke();
      ctx.fillText(v.toFixed(vDecimals), ML - 7, y);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const nfTicks = 6;
    const fDecimals = tickDecimals((f1 - f0) / nfTicks);
    for (const f of evenTicks(f0, f1, nfTicks)) {
      const x = xOf(f);
      ctx.beginPath();
      ctx.moveTo(x, MT + PLOT_H);
      ctx.lineTo(x, MT + PLOT_H + 4);
      ctx.stroke();
      ctx.fillText(f.toFixed(fDecimals), x, MT + PLOT_H + 6);
    }

    ctx.fillStyle = palette.title;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Frequency (Hz)", ML + PLOT_W / 2, TOTAL_H - 4);
    ctx.save();
    ctx.translate(16, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Phase velocity (m/s)", 0, 0);
    ctx.restore();
  }

  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingPolygon, image, theme, scale, f0, f1, v0, v1, TOTAL_W, TOTAL_H, PLOT_W, PLOT_H]);

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
    // In data units, through the view on show: the same (frequency,
    // velocity) polygon whatever the zoom.
    return [fOf(px), vOf(py)];
  }

  // A lasso within a few pixels is a click (a double-click's, say), which
  // keeps the pending lasso.
  function isClick(polygon: [number, number][]) {
    const xs = polygon.map(([f]) => xOf(f));
    const ys = polygon.map(([, v]) => yOf(v));
    const w = (Math.max(...xs) - Math.min(...xs)) * scale;
    const h = (Math.max(...ys) - Math.min(...ys)) * scale;
    return w < CLICK_PX && h < CLICK_PX;
  }

  function onMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    // Along an axis's tick labels, a zoom of that axis, as with every tool.
    if (zoom.axisAt(hoverPos)) {
      zoom.onMouseDown(e);
      return;
    }
    draggingRef.current = true;
    dragPointsRef.current = [clampedDataPoint(e.nativeEvent.offsetX, e.nativeEvent.offsetY)];
  }

  function onMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    onHoverMove(e);
    if (!draggingRef.current) return;
    dragPointsRef.current.push(clampedDataPoint(e.nativeEvent.offsetX, e.nativeEvent.offsetY));
    draw();
  }

  function onMouseUp() {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    const polygon = dragPointsRef.current;
    dragPointsRef.current = [];
    if (polygon.length >= 3 && !isClick(polygon)) onLassoComplete?.(polygon);
    draw();
  }

  // The amplitude as drawn at (f, v): a share of its frequency's peak ("; 0.87"), or nothing.
  const amplitudeAt = (f: number, v: number): string => {
    const column = image.fv_map[nearestIndex(image.fs, f)] ?? [];
    const peak = column.reduce((most, value) => Math.max(most, value), 0);
    const value = column[nearestIndex(image.vs, v)];
    return peak > 0 && value !== undefined ? `; ${(value / peak).toFixed(2)}` : "";
  };

  // Over a λ line: how the checks define it, with its value.
  const lambdaTip = ((): Tip | null => {
    if (!hoverPos) return null;
    const { x, y } = hoverPos;
    if (x < ML || x > ML + PLOT_W || y < MT || y > MT + PLOT_H) return null;
    const near = (lambda: number | null) => {
      if (lambda === null) return false;
      // Pixels from the line v = f λ, across it or along it, whichever is shorter.
      const across = Math.abs(yOf(fOf(x) * lambda) - y);
      const along = Math.abs(xOf(vOf(y) / lambda) - x);
      return Math.min(across, along) <= 6;
    };
    const metres = (value: number) => `${+value.toFixed(value < 10 ? 2 : 1)} m`;
    // The line's point under the pointer: its frequency and velocity, and the wavelength.
    const onLine = (lambda: number) => {
      const f = fOf(x);
      const v = f * lambda;
      return `${f.toFixed(1)} Hz; ${v.toFixed(0)} m/s${amplitudeAt(f, v)}; λ ${metres(lambda)}`;
    };
    if (near(image.lambda_min)) {
      const lambda = image.lambda_min as number;
      return { title: "λmin", values: onLine(lambda), notes: [`2 receiver spacings (${metres(lambda / 2)})`] };
    }
    if (near(image.lambda_max)) {
      const lambda = image.lambda_max as number;
      return { title: "λmax", values: onLine(lambda), notes: [`3 window lengths (${metres(lambda / 3)})`] };
    }
    return null;
  })();

  // Elsewhere on the image: what is under the pointer, a pick first when one is that near.
  const dataTip = ((): Tip | null => {
    if (!hoverPos) return null;
    const { x, y } = hoverPos;
    if (x < ML || x > ML + PLOT_W || y < MT || y > MT + PLOT_H) return null;
    const f = fOf(x);
    const v = vOf(y);
    let pick: { label: string; f: number; v: number; err: number | null; d: number } | null = null;
    for (const curve of image.curves) {
      for (let k = 0; k < curve.fs.length; k++) {
        const d = Math.hypot(xOf(curve.fs[k]) - x, yOf(curve.vs[k]) - y);
        if (d <= 6 && (!pick || d < pick.d)) {
          pick = { label: curve.label, f: curve.fs[k], v: curve.vs[k], err: curve.vs_std?.[k] ?? null, d };
        }
      }
    }
    if (pick) {
      const { label, f: pf, v: pv, err } = pick;
      return {
        title: label,
        values: `${pf.toFixed(1)} Hz; ${pv.toFixed(0)}${err !== null ? ` ± ${err.toFixed(0)}` : ""} m/s${amplitudeAt(pf, pv)}; λ ${(pv / pf).toFixed(1)} m`,
      };
    }
    // x; y; z (the amplitude); the wavelength.
    return { values: `${f.toFixed(1)} Hz; ${v.toFixed(0)} m/s${amplitudeAt(f, v)}; λ ${(v / f).toFixed(1)} m` };
  })();
  const tip = lambdaTip ?? dataTip;

  return (
    <div ref={containerRef} style={{ width: "100%", position: "relative" }}>
      <canvas
        ref={canvasRef}
        style={{
          cursor: lasso && !zoom.axisAt(hoverPos) ? "crosshair" : zoom.cursorAt(hoverPos),
          touchAction: "none",
          display: "block",
          height: TOTAL_H * scale,
        }}
        onMouseDown={lasso ? onMouseDown : zoom.onMouseDown}
        onMouseMove={lasso ? onMouseMove : onHoverMove}
        onMouseUp={lasso ? onMouseUp : undefined}
        onMouseLeave={
          lasso
            ? () => {
                onMouseUp();
                onHoverLeave();
              }
            : onHoverLeave
        }
      />
      <ZoomSelection box={zoom.selection} />
      {tip && hoverPos && <HoverTooltip x={hoverPos.x * scale} y={hoverPos.y * scale} tip={tip} />}
    </div>
  );
}
