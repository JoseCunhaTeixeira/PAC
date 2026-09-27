import { useEffect, useMemo, useRef } from "react";
import { cividis } from "./colormaps";
import { HoverTooltip } from "./HoverTooltip";
import { Badge } from "./kit";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { nearestIndex, useCanvasHover } from "./useCanvasHover";
import { drawMarker, useClick } from "./sectionPick";
import { useContainerWidth } from "./useContainerWidth";
import {
  evenTicks,
  positionTicks,
  tickDecimals,
  useZoom,
  valueRange,
  visibleCells,
  visibleColumns,
} from "./useZoom";
import { ZoomReset, ZoomSelection } from "./ZoomOverlay";

export interface PseudoSection {
  positions: number[];
  fs_grid: number[];
  velocities_by_frequency: (number | null)[][];
  lambdas_grid: number[];
  velocities_by_wavelength: (number | null)[][];
}

const ML = 60, MR = 120, MT = 16, MB = 40;
const BASE_W = 820; // the drawing's width until its card is measured
const FONT = CANVAS_FONT;

/** A mode's pseudo-section's head, alike on every page: its label (M0, M1…), and on how many of
 * the positions it is drawn. */
export function ModeHead({
  label,
  count,
  total,
  unit,
}: {
  label: string;
  count?: number;
  total: number;
  unit: string;
}) {
  return (
    <div className="section-label">
      <strong>{label}</strong>
      {count !== undefined && total > 0 && (
        <Badge tone={count === total ? "ok" : "neutral"}>
          {count}/{total} {unit}
        </Badge>
      )}
    </div>
  );
}

export function PseudoSectionCanvas({
  section,
  mode,
  height = 320,
  marker,
  onPick,
}: {
  section: PseudoSection;
  mode: "frequency" | "wavelength";
  height?: number;
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

  // The full view. Wavelengths run down like depths, frequencies up; a
  // switch between the two starts from the full view again.
  const xFirst = section.positions[0];
  const xExtent = section.positions[section.positions.length - 1] - xFirst || 1;
  const yAxisGrid = mode === "frequency" ? section.fs_grid : section.lambdas_grid;
  const yFirst = yAxisGrid[0];
  const yExtent = yAxisGrid[yAxisGrid.length - 1] - yFirst || 1;
  const zoom = useZoom({
    extent: { x: [xFirst, xFirst + xExtent], y: [yFirst, yFirst + yExtent] },
    plots: [{ left: ML, top: MT, width: PLOT_W, height: PLOT_H, xAxis: MB, yAxis: ML }],
    width: TOTAL_W,
    height: TOTAL_H,
    yDown: mode === "wavelength",
    resetKey: mode,
  });
  // The positions and frequencies (or wavelengths) on show.
  const [x0, x1] = zoom.view.x;
  const [y0, y1] = zoom.view.y;

  const hover = useMemo(() => {
    if (!hoverPos) return null;
    if (
      hoverPos.x < ML || hoverPos.x > ML + PLOT_W ||
      hoverPos.y < MT || hoverPos.y > MT + PLOT_H
    ) {
      return null;
    }

    const yGrid = mode === "frequency" ? section.fs_grid : section.lambdas_grid;
    const velocities = mode === "frequency" ? section.velocities_by_frequency : section.velocities_by_wavelength;
    const yLabel = mode === "frequency" ? "Frequency" : "Wavelength";
    const yUnit = mode === "frequency" ? "Hz" : "m";
    const invertY = mode === "wavelength";
    const positions = section.positions;

    const position = x0 + ((hoverPos.x - ML) / PLOT_W) * (x1 - x0);
    const yValue = invertY
      ? y0 + ((hoverPos.y - MT) / PLOT_H) * (y1 - y0)
      : y0 + ((MT + PLOT_H - hoverPos.y) / PLOT_H) * (y1 - y0);

    const posIdx = nearestIndex(positions, position);
    const yIdx = nearestIndex(yGrid, yValue);
    const value = velocities[posIdx]?.[yIdx] ?? null;

    return {
      px: hoverPos.x * scale,
      py: hoverPos.y * scale,
      lines: [
        `xmid ${positions[posIdx].toFixed(2)} m`,
        `${yLabel.toLowerCase()} ${yGrid[yIdx].toFixed(2)} ${yUnit}`,
        `phase velocity ${value === null ? "—" : value.toFixed(1)} m/s`,
      ],
    };
  }, [hoverPos, section, mode, scale, PLOT_H, PLOT_W, x0, x1, y0, y1]);

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

    const yGrid = mode === "frequency" ? section.fs_grid : section.lambdas_grid;
    const velocities = mode === "frequency" ? section.velocities_by_frequency : section.velocities_by_wavelength;
    const yLabel = mode === "frequency" ? "Frequency (Hz)" : "Wavelength (m)";
    const positions = section.positions;
    const np = positions.length;
    const ny = yGrid.length;

    // Wavelength roughly tracks depth sensitivity, so unlike frequency it is
    // plotted like a depth axis: smaller (shallower) at the top, larger
    // (deeper) at the bottom — the reverse of the frequency orientation.
    const invertY = mode === "wavelength";
    const yMin = yGrid[0];
    const yMax = yGrid[yGrid.length - 1];
    const ySpan = yMax - yMin || 1;
    const yOf = (y: number) =>
      invertY
        ? MT + ((y - y0) / (y1 - y0)) * PLOT_H
        : MT + PLOT_H - ((y - y0) / (y1 - y0)) * PLOT_H;

    // Plot exactly between min(position) and max(position)
    const xMin = positions[0];
    const xMax = positions[np - 1];

    const xOf = (p: number) =>
      ML + ((p - x0) / (x1 - x0)) * PLOT_W;

    // Midpoint boundaries, clipped to plot limits
    const columnEdges: number[] = new Array(np + 1);

    columnEdges[0] = xMin;
    columnEdges[np] = xMax;

    for (let i = 1; i < np; i++) {
      columnEdges[i] = (positions[i - 1] + positions[i]) / 2;
    }

    // Only the cells on show: a column's rows run from rowTop at the top of
    // the plot to rowBottom, and the plot's edges clip those they cut through.
    const [i0, i1] = visibleColumns(columnEdges, x0, x1);
    const rowTop = invertY ? yMin : yMin + ySpan;
    const rowBottom = invertY ? yMin + ySpan : yMin;
    const [k0, k1] = visibleCells(ny, rowTop, rowBottom, y0, y1);
    const yTop = yOf(rowTop + (k0 / ny) * (rowBottom - rowTop));
    const yBottom = yOf(rowTop + (k1 / ny) * (rowBottom - rowTop));

    // The colour scale spans the values on show: all of them in the full
    // view, those in the window when zoomed. The image's rows are the data's
    // entries top down for wavelengths, bottom up for frequencies.
    const [j0, j1] = invertY ? [k0, k1] : [ny - k1, ny - k0];
    const [vMin, vMax] =
      valueRange(velocities, i0, i1, j0, j1) || valueRange(velocities, 0, np, 0, ny) || [0, 1];
    const vSpan = vMax - vMin || 1;

    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, MT, PLOT_W, PLOT_H);
    ctx.clip();

    for (let i = i0; i < i1 && k1 > k0; i++) {
      const xLeft = xOf(columnEdges[i]);
      const xRight = xOf(columnEdges[i + 1]);

      const off = document.createElement("canvas");
      off.width = 1;
      off.height = ny;

      const octx = off.getContext("2d");
      if (!octx) continue;

      const imgData = octx.createImageData(1, ny);

      for (let j = 0; j < ny; j++) {
        const v = velocities[i][j];
        const y = invertY ? j : ny - 1 - j;
        const idx = y * 4;

        if (v === null) {
          imgData.data[idx + 3] = 0;
          continue;
        }

        const [r, g, b] = cividis((v - vMin) / vSpan);

        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }

      octx.putImageData(imgData, 0, 0);

      ctx.drawImage(
        off,
        0,
        k0,
        1,
        k1 - k0,
        xLeft,
        yTop,
        Math.max(1, xRight - xLeft),
        yBottom - yTop
      );
    }
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
    const nyTicks = 6;
    const yDecimals = tickDecimals((y1 - y0) / nyTicks, 1);
    for (const y of evenTicks(y0, y1, nyTicks)) {
      const py = yOf(y);
      ctx.beginPath();
      ctx.moveTo(ML - 4, py);
      ctx.lineTo(ML, py);
      ctx.stroke();
      ctx.fillText(y.toFixed(yDecimals), ML - 7, py);
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

    // color legend — rendered through an offscreen image + drawImage (like
    // the pcolormesh columns above) rather than per-row fillRect, since
    // tiling many 1px fillRects leaves visible seams once the canvas is
    // rasterized at a non-integer scale (fractional container widths).
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
        const [r, g, b] = cividis(t);
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
    ctx.fillText("Position (m)", ML + PLOT_W / 2, TOTAL_H - 4);
    ctx.save();
    ctx.translate(16, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    ctx.save();
    ctx.translate(TOTAL_W - 14, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Phase velocity (m/s)", 0, 0);
    ctx.restore();
  }, [section, mode, PLOT_H, TOTAL_H, TOTAL_W, PLOT_W, palette, scale, x0, x1, y0, y1, marker]);

  const click = useClick(
    TOTAL_W,
    TOTAL_H,
    onPick &&
      ((x, y) => {
        if (x < ML || x > ML + PLOT_W || y < MT || y > MT + PLOT_H) return;
        const positions = section.positions;
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
