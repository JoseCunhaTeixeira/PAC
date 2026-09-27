import { useEffect, useMemo, useRef } from "react";
import { cividis } from "./colormaps";
import { HoverTooltip } from "./HoverTooltip";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { nearestIndex, useCanvasHover } from "./useCanvasHover";
import { useContainerWidth } from "./useContainerWidth";
import {
  evenTicks,
  positionTicks,
  tickDecimals,
  useZoom,
  valueRange,
  visibleCells,
  visibleColumns,
  type ZoomLink,
} from "./useZoom";
import { ZoomReset, ZoomSelection } from "./ZoomOverlay";
import { drawMarker, useClick } from "./sectionPick";

const ML = 60, MR = 120, MT = 16, MB = 40;
const BASE_W = 820; // the drawing's width until its card is measured
const FONT = CANVAS_FONT;

// Module-level (not inline) so it's referentially stable across renders when
// callers don't override it, matching `colormap`'s default (cividis) --
// an inline arrow default would be a new function every render, forcing the
// hover/draw effects below to always see a "changed" dependency.
const DEFAULT_FORMAT_VALUE = (v: number) => v.toFixed(1);

export function VelocitySectionCanvas({
  positions,
  elevations,
  values,
  colorLabel,
  colormap = cividis,
  height = 320,
  formatValue = DEFAULT_FORMAT_VALUE,
  link,
  colorRange,
  marker,
  onPick,
}: {
  positions: number[];
  elevations: number[];
  values: (number | null)[][];
  colorLabel: string;
  colormap?: (t: number) => [number, number, number];
  height?: number;
  // Values are typically Vs-scale (tens to thousands) but this canvas is
  // reused for other continuous fields too (e.g. shear modulus in GPa,
  // ~0.05-0.5) where the default 1-decimal formatting would round
  // everything to "0.0" -- callers with a different value scale should
  // override this.
  formatValue?: (v: number) => string;
  // One zoom with the sections of the same grid shown beside it (a Vs
  // section and its std): zooming or resetting one does both.
  link?: ZoomLink;
  // Ends of the colour scale the user fixed. An end left out follows the
  // data: all of it in the full view, the values on show when zoomed.
  colorRange?: { min?: number; max?: number };
  // A position to mark down the section (the selected window), and what a click on a column
  // (not a zoom's drag) selects: its position.
  marker?: number;
  onPick?: (position: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const palette = useMemo(() => canvasPalette(theme), [theme]);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  // As wide as its card, the text at its own size.
  const TOTAL_W = Math.max(480, Math.round(containerWidth || BASE_W));
  const PLOT_W = TOTAL_W - ML - MR;
  const scale = 1;
  const PLOT_H = height;
  const TOTAL_H = MT + PLOT_H + MB;
  const { pos: hoverPos, onMouseMove, onMouseLeave } = useCanvasHover(scale);

  // The full view: positions left to right, elevations from the highest at
  // the top down.
  const xFirst = positions[0];
  const xExtent = positions[positions.length - 1] - xFirst || 1;
  const zTop = elevations[0];
  const zExtent = zTop - elevations[elevations.length - 1] || 1;
  const zoom = useZoom({
    extent: { x: [xFirst, xFirst + xExtent], y: [zTop - zExtent, zTop] },
    plots: [{ left: ML, top: MT, width: PLOT_W, height: PLOT_H, xAxis: MB, yAxis: ML }],
    width: TOTAL_W,
    height: TOTAL_H,
    link,
  });
  // The positions and elevations on show: the whole section, or the zoom.
  const [x0, x1] = zoom.view.x;
  const [z0, z1] = zoom.view.y;
  const fixedMin = colorRange?.min;
  const fixedMax = colorRange?.max;

  const hover = useMemo(() => {
    if (!hoverPos) return null;
    if (
      hoverPos.x < ML || hoverPos.x > ML + PLOT_W ||
      hoverPos.y < MT || hoverPos.y > MT + PLOT_H
    ) {
      return null;
    }

    const position = x0 + ((hoverPos.x - ML) / PLOT_W) * (x1 - x0);
    const elevation = z1 - ((hoverPos.y - MT) / PLOT_H) * (z1 - z0);

    const posIdx = nearestIndex(positions, position);
    const zIdx = nearestIndex(elevations, elevation);
    const value = values[posIdx]?.[zIdx] ?? null;

    return {
      px: hoverPos.x * scale,
      py: hoverPos.y * scale,
      lines: [
        `xmid ${positions[posIdx].toFixed(2)} m`,
        `elevation ${elevations[zIdx].toFixed(2)} m`,
        `${colorLabel}: ${value === null ? "—" : formatValue(value)}`,
      ],
    };
  }, [hoverPos, positions, elevations, values, colorLabel, scale, PLOT_H, PLOT_W, formatValue, x0, x1, z0, z1]);

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
    const yOf = (z: number) => MT + ((z1 - z) / (z1 - z0)) * PLOT_H;

    // Plot exactly between min(position) and max(position), like
    // PseudoSectionCanvas/PseudoSectionComparisonCanvas.
    const xMin = positions[0];
    const xMax = positions[np - 1];
    const xOf = (p: number) => ML + ((p - x0) / (x1 - x0)) * PLOT_W;

    // Midpoint boundaries, clipped to plot limits
    const cellEdges: number[] = new Array(np + 1);
    cellEdges[0] = xMin;
    cellEdges[np] = xMax;
    for (let i = 1; i < np; i++) cellEdges[i] = (positions[i - 1] + positions[i]) / 2;

    // Only the cells on show: a column's rows run from zMax at the top down
    // by zSpan, and the plot's edges clip those they cut through.
    const [i0, i1] = visibleColumns(cellEdges, x0, x1);
    const [k0, k1] = visibleCells(nz, zMax, zMax - zSpan, z0, z1);
    const yTop = yOf(zMax - (k0 / nz) * zSpan);
    const yBottom = yOf(zMax - (k1 / nz) * zSpan);

    // The colour scale spans the values on show: all of them in the full
    // view, those in the window when zoomed. An end the user fixed stays.
    const [autoMin, autoMax] =
      valueRange(values, i0, i1, k0, k1) || valueRange(values, 0, np, 0, nz) || [0, 1];
    const vMin = fixedMin ?? autoMin;
    const vMax = fixedMax ?? autoMax;
    // (A fixed end beyond all the values on show leaves them at one end of the colours.)
    const vSpan = vMax > vMin ? vMax - vMin : 1;

    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, MT, PLOT_W, PLOT_H);
    ctx.clip();
    ctx.imageSmoothingEnabled = false;
    for (let i = i0; i < i1 && k1 > k0; i++) {
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
      ctx.drawImage(off, 0, k0, 1, k1 - k0, xLeft, yTop, Math.max(1, xRight - xLeft), yBottom - yTop);
    }
    ctx.imageSmoothingEnabled = true;
    ctx.restore();
    if (marker !== undefined && marker >= x0 && marker <= x1) drawMarker(ctx, xOf(marker), MT, MT + PLOT_H);

    // axes
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.font = FONT;
    ctx.strokeRect(ML, MT, PLOT_W, PLOT_H);

    ctx.fillStyle = palette.tick;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const nzTicks = 6;
    const zDecimals = tickDecimals((z1 - z0) / nzTicks, 1);
    for (const z of evenTicks(z0, z1, nzTicks)) {
      const py = yOf(z);
      ctx.beginPath();
      ctx.moveTo(ML - 4, py);
      ctx.lineTo(ML, py);
      ctx.stroke();
      ctx.fillText(z.toFixed(zDecimals), ML - 7, py);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const { p, label } of positionTicks(positions, x0, x1)) {
      const x = xOf(p);
      ctx.beginPath();
      ctx.moveTo(x, MT + PLOT_H);
      ctx.lineTo(x, MT + PLOT_H + 4);
      ctx.stroke();
      ctx.fillText(label, x, MT + PLOT_H + 6);
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
      ctx.fillText(formatValue(v), legendX + legendW + 6, py);
    }

    ctx.fillStyle = palette.title;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Position (m)", ML + PLOT_W / 2, TOTAL_H - 4);
    ctx.save();
    ctx.translate(16, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Elevation (m)", 0, 0);
    ctx.restore();
    ctx.save();
    ctx.translate(TOTAL_W - 14, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText(colorLabel, 0, 0);
    ctx.restore();
  }, [positions, elevations, values, colorLabel, colormap, PLOT_H, TOTAL_H, TOTAL_W, PLOT_W, palette, scale, formatValue, fixedMin, fixedMax, x0, x1, z0, z1, marker]);

  const click = useClick(
    TOTAL_W,
    TOTAL_H,
    onPick &&
      ((x, y) => {
        if (x < ML || x > ML + PLOT_W || y < MT || y > MT + PLOT_H) return;
        onPick(positions[nearestIndex(positions, x0 + ((x - ML) / PLOT_W) * (x1 - x0))]);
      }),
    zoom.onMouseDown,
  );

  return (
    <div ref={containerRef} style={{ width: "100%", position: "relative" }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", cursor: zoom.cursorAt(hoverPos) }}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
        onMouseDown={click.onMouseDown}
        onClick={click.onClick}
        onDoubleClick={zoom.onDoubleClick}
      />
      <ZoomSelection box={zoom.selection} />
      <ZoomReset zoomed={zoom.zoomed} onReset={zoom.reset} style={{ top: 0, right: MR * scale }} />
      {hover && <HoverTooltip x={hover.px} y={hover.py} lines={hover.lines} />}
    </div>
  );
}
