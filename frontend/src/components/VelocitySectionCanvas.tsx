import { useEffect, useMemo, useRef } from "react";
import { cividis } from "./colormaps";
import { HoverTooltip } from "./HoverTooltip";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { nearestIndex, useCanvasHover } from "./useCanvasHover";
import { useContainerWidth } from "./useContainerWidth";

const ML = 60, MR = 120, MT = 16, MB = 40;
const PLOT_W = 640;
const FONT = CANVAS_FONT;
const TOTAL_W = ML + PLOT_W + MR;

export function VelocitySectionCanvas({
  positions,
  elevations,
  values,
  colorLabel,
  colormap = cividis,
  height = 320,
}: {
  positions: number[];
  elevations: number[];
  values: (number | null)[][];
  colorLabel: string;
  colormap?: (t: number) => [number, number, number];
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const palette = canvasPalette(theme);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const scale = containerWidth > 0 ? Math.min(containerWidth / TOTAL_W, 1) : 1;
  const PLOT_H = height;
  const TOTAL_H = MT + PLOT_H + MB;
  const { pos: hoverPos, onMouseMove, onMouseLeave } = useCanvasHover(scale);

  const hover = useMemo(() => {
    if (!hoverPos) return null;
    if (
      hoverPos.x < ML || hoverPos.x > ML + PLOT_W ||
      hoverPos.y < MT || hoverPos.y > MT + PLOT_H
    ) {
      return null;
    }

    const np = positions.length;
    const xMin = positions[0];
    const xMax = positions[np - 1];
    const xSpan = xMax - xMin || 1;
    const position = xMin + ((hoverPos.x - ML) / PLOT_W) * xSpan;

    const zMin = elevations[elevations.length - 1];
    const zMax = elevations[0];
    const zSpan = zMax - zMin || 1;
    const elevation = zMax - ((hoverPos.y - MT) / PLOT_H) * zSpan;

    const posIdx = nearestIndex(positions, position);
    const zIdx = nearestIndex(elevations, elevation);
    const value = values[posIdx]?.[zIdx] ?? null;

    return {
      px: hoverPos.x * scale,
      py: hoverPos.y * scale,
      lines: [
        `Position: ${positions[posIdx].toFixed(2)} m`,
        `Elevation: ${elevations[zIdx].toFixed(2)} m`,
        `${colorLabel}: ${value === null ? "—" : value.toFixed(1)}`,
      ],
    };
  }, [hoverPos, positions, elevations, values, colorLabel, scale, PLOT_H]);

  useEffect(() => {
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

    const np = positions.length;
    const nz = elevations.length;

    // Elevation decreases downward (shallow/high elevation at the top of the
    // chart, deep/low elevation at the bottom) — the same orientation as a
    // geological cross-section.
    const zMin = elevations[elevations.length - 1];
    const zMax = elevations[0];
    const zSpan = zMax - zMin || 1;
    const yOf = (z: number) => MT + ((zMax - z) / zSpan) * PLOT_H;

    let vMin = Infinity, vMax = -Infinity;
    for (const row of values) {
      for (const v of row) {
        if (v !== null) {
          if (v < vMin) vMin = v;
          if (v > vMax) vMax = v;
        }
      }
    }
    if (!Number.isFinite(vMin)) { vMin = 0; vMax = 1; }
    const vSpan = vMax - vMin || 1;

    // Plot exactly between min(position) and max(position), like
    // PseudoSectionCanvas/PseudoSectionComparisonCanvas.
    const xMin = positions[0];
    const xMax = positions[np - 1];
    const xSpan = xMax - xMin || 1;
    const xOf = (p: number) => ML + ((p - xMin) / xSpan) * PLOT_W;

    // Midpoint boundaries, clipped to plot limits
    const cellEdges: number[] = new Array(np + 1);
    cellEdges[0] = xMin;
    cellEdges[np] = xMax;
    for (let i = 1; i < np; i++) cellEdges[i] = (positions[i - 1] + positions[i]) / 2;

    ctx.imageSmoothingEnabled = false;
    for (let i = 0; i < np; i++) {
      // Rounded to whole pixels so adjacent columns share an exact integer
      // boundary -- left as floats, each column's edge gets anti-aliased
      // against the background independently, leaving a thin seam of
      // blended color between every pair of columns.
      const xLeft = Math.round(xOf(cellEdges[i]));
      const xRight = Math.round(xOf(cellEdges[i + 1]));
      const off = document.createElement("canvas");
      off.width = 1;
      off.height = nz;
      const octx = off.getContext("2d");
      if (!octx) continue;
      const imgData = octx.createImageData(1, nz);
      for (let j = 0; j < nz; j++) {
        const v = values[i][j];
        const idx = j * 4;
        if (v === null) {
          imgData.data[idx + 3] = 0;
          continue;
        }
        const [r, g, b] = colormap((v - vMin) / vSpan);
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
      octx.putImageData(imgData, 0, 0);
      ctx.drawImage(off, 0, 0, 1, nz, xLeft, MT, Math.max(1, xRight - xLeft), PLOT_H);
    }
    ctx.imageSmoothingEnabled = true;

    // axes
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.font = FONT;
    ctx.strokeRect(ML, MT, PLOT_W, PLOT_H);

    ctx.fillStyle = palette.tick;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const nzTicks = 6;
    for (let i = 0; i <= nzTicks; i++) {
      const z = zMin + (i / nzTicks) * zSpan;
      const py = yOf(z);
      ctx.beginPath();
      ctx.moveTo(ML - 4, py);
      ctx.lineTo(ML, py);
      ctx.stroke();
      ctx.fillText(z.toFixed(1), ML - 7, py);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const nxTicks = Math.min(8, np - 1);
    for (let i = 0; i <= nxTicks; i++) {
      const idx = nxTicks > 0 ? Math.round((i / nxTicks) * (np - 1)) : 0;
      const p = positions[idx];
      const x = xOf(p);
      ctx.beginPath();
      ctx.moveTo(x, MT + PLOT_H);
      ctx.lineTo(x, MT + PLOT_H + 4);
      ctx.stroke();
      ctx.fillText(p.toFixed(1), x, MT + PLOT_H + 6);
    }

    // color legend
    const legendX = ML + PLOT_W + 20;
    const legendW = 14;
    const legendRes = 256;
    const legendOff = document.createElement("canvas");
    legendOff.width = 1;
    legendOff.height = legendRes;
    const legendOctx = legendOff.getContext("2d");
    if (legendOctx) {
      const legendImg = legendOctx.createImageData(1, legendRes);
      for (let py = 0; py < legendRes; py++) {
        const t = 1 - py / (legendRes - 1);
        const [r, g, b] = colormap(t);
        const idx = py * 4;
        legendImg.data[idx] = r;
        legendImg.data[idx + 1] = g;
        legendImg.data[idx + 2] = b;
        legendImg.data[idx + 3] = 255;
      }
      legendOctx.putImageData(legendImg, 0, 0);
      ctx.drawImage(legendOff, 0, 0, 1, legendRes, legendX, MT, legendW, PLOT_H);
    }
    ctx.strokeStyle = palette.axis;
    ctx.strokeRect(legendX, MT, legendW, PLOT_H);
    ctx.fillStyle = palette.tick;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const nLegendTicks = 4;
    for (let i = 0; i <= nLegendTicks; i++) {
      const v = vMin + (i / nLegendTicks) * vSpan;
      const py = MT + PLOT_H - (i / nLegendTicks) * PLOT_H;
      ctx.fillText(v.toFixed(0), legendX + legendW + 6, py);
    }

    ctx.fillStyle = palette.title;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Position [m]", ML + PLOT_W / 2, TOTAL_H - 4);
    ctx.save();
    ctx.translate(16, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Elevation [m]", 0, 0);
    ctx.restore();
    ctx.save();
    ctx.translate(TOTAL_W - 14, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText(colorLabel, 0, 0);
    ctx.restore();
  }, [positions, elevations, values, colorLabel, colormap, height, theme, scale]);

  return (
    <div ref={containerRef} style={{ width: "100%", maxWidth: TOTAL_W, position: "relative" }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block" }}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
      />
      {hover && <HoverTooltip x={hover.px} y={hover.py} lines={hover.lines} />}
    </div>
  );
}
