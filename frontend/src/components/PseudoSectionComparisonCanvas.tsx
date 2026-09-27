import { useEffect, useMemo, useRef } from "react";
import { bwr, cividis } from "./colormaps";
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
  type PlotRect,
} from "./useZoom";
import { ZoomReset, ZoomSelection } from "./ZoomOverlay";
import { drawMarker, useClick } from "./sectionPick";

export interface PseudoSectionComparisonData {
  positions: number[];
  fs: number[];
  observed_grid: (number | null)[][];
  predicted_grid: (number | null)[][];
  residual_grid: (number | null)[][];
  lambdas: number[];
  observed_by_wavelength_grid: (number | null)[][];
  predicted_by_wavelength_grid: (number | null)[][];
  residual_by_wavelength_grid: (number | null)[][];
}

/** The comparison along `mode`'s axis: its values, and the three grids on them. */
function along(comparison: PseudoSectionComparisonData, mode: "frequency" | "wavelength") {
  return mode === "frequency"
    ? {
        ys: comparison.fs,
        observed: comparison.observed_grid,
        predicted: comparison.predicted_grid,
        residual: comparison.residual_grid,
      }
    : {
        ys: comparison.lambdas,
        observed: comparison.observed_by_wavelength_grid,
        predicted: comparison.predicted_by_wavelength_grid,
        residual: comparison.residual_by_wavelength_grid,
      };
}

/** `text` written up the canvas, centred on (x, cy): on two lines when longer than `room`. */
function verticalLabel(ctx: CanvasRenderingContext2D, text: string, x: number, cy: number, room: number) {
  ctx.save();
  ctx.translate(x, cy);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  if (ctx.measureText(text).width <= room) {
    ctx.fillText(text, 0, 0);
  } else {
    // Split at the space nearest the middle; the first line left of the second.
    const words = text.split(" ");
    let best = 1;
    for (let k = 1; k < words.length; k++) {
      const gap = Math.abs(words.slice(0, k).join(" ").length - text.length / 2);
      if (gap < Math.abs(words.slice(0, best).join(" ").length - text.length / 2)) best = k;
    }
    ctx.fillText(words.slice(0, best).join(" "), 0, -14);
    ctx.fillText(words.slice(best).join(" "), 0, 0);
  }
  ctx.restore();
}

const ML = 60, MR = 130, MT = 16, MB = 40, PANEL_GAP = 30;
const PLOT_H = 130;
const BASE_W = 830; // the drawing's width until its card is measured
const FONT = CANVAS_FONT;
const TOTAL_H = MT + 3 * PLOT_H + 2 * PANEL_GAP + MB;
// The three panels share their axes: a zoom in one zooms them all. Each has
// its own ticks, in the gap under it and the margin left of it.
function panels(plotW: number): PlotRect[] {
  return [0, 1, 2].map((i) => ({
    left: ML,
    top: MT + i * (PLOT_H + PANEL_GAP),
    width: plotW,
    height: PLOT_H,
    xAxis: i < 2 ? PANEL_GAP : MB,
    yAxis: ML,
  }));
}

// Observed/predicted/residual pseudo-sections stacked vertically, like
// sigpipe's `plot_pseudo_section_comparison`: obs+pred share one scale, in the
// other pseudo-sections' cividis, so they're directly comparable; residual uses
// a symmetric bwr scale.
export function PseudoSectionComparisonCanvas({
  comparison,
  velocityLabel,
  mode = "frequency",
  marker,
  onPick,
}: {
  comparison: PseudoSectionComparisonData;
  velocityLabel: string;
  /** The vertical axis: frequency, up, or wavelength, down like a depth (as a pseudo-section's). */
  mode?: "frequency" | "wavelength";
  // A position to mark down the panels (the selected window), and what a click on a column
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
  const PANELS = panels(PLOT_W);
  const scale = 1;
  const { pos: hoverPos, onMouseMove, onMouseLeave } = useCanvasHover(scale);

  // The full view; a switch between the axes starts from it again.
  const invertY = mode === "wavelength";
  const xFirst = comparison.positions[0];
  const xExtent = comparison.positions[comparison.positions.length - 1] - xFirst || 1;
  const axisValues = along(comparison, mode).ys;
  const fFirst = axisValues[0];
  const fExtent = axisValues[axisValues.length - 1] - fFirst || 1;
  const zoom = useZoom({
    extent: { x: [xFirst, xFirst + xExtent], y: [fFirst, fFirst + fExtent] },
    plots: PANELS,
    width: TOTAL_W,
    height: TOTAL_H,
    yDown: invertY,
    resetKey: mode,
  });
  // The positions and frequencies (or wavelengths) on show, in all three panels.
  const [x0, x1] = zoom.view.x;
  const [f0, f1] = zoom.view.y;

  const hover = useMemo(() => {
    if (!hoverPos) return null;
    if (hoverPos.x < ML || hoverPos.x > ML + PLOT_W) return null;

    const { positions } = comparison;
    const { ys: fs, observed: observed_grid, predicted: predicted_grid, residual: residual_grid } = along(
      comparison,
      mode,
    );

    const top1 = MT;
    const top2 = top1 + PLOT_H + PANEL_GAP;
    const top3 = top2 + PLOT_H + PANEL_GAP;

    let top: number;
    let grid: (number | null)[][];
    let label: string;
    if (hoverPos.y >= top1 && hoverPos.y <= top1 + PLOT_H) {
      top = top1;
      grid = observed_grid;
      label = `Picked ${velocityLabel.toLowerCase()}`;
    } else if (hoverPos.y >= top2 && hoverPos.y <= top2 + PLOT_H) {
      top = top2;
      grid = predicted_grid;
      label = `Modelled ${velocityLabel.toLowerCase()}`;
    } else if (hoverPos.y >= top3 && hoverPos.y <= top3 + PLOT_H) {
      top = top3;
      grid = residual_grid;
      label = "Residual (%)";
    } else {
      return null;
    }

    const position = x0 + ((hoverPos.x - ML) / PLOT_W) * (x1 - x0);
    const freq = invertY
      ? f0 + ((hoverPos.y - top) / PLOT_H) * (f1 - f0)
      : f0 + ((top + PLOT_H - hoverPos.y) / PLOT_H) * (f1 - f0);

    const posIdx = nearestIndex(positions, position);
    const fIdx = nearestIndex(fs, freq);
    const value = grid[posIdx]?.[fIdx] ?? null;

    return {
      px: hoverPos.x * scale,
      py: hoverPos.y * scale,
      lines: [
        `xmid ${positions[posIdx].toFixed(2)} m`,
        invertY ? `wavelength ${fs[fIdx].toFixed(2)} m` : `frequency ${fs[fIdx].toFixed(2)} Hz`,
        `${label}: ${value === null ? "—" : value.toFixed(1)}`,
      ],
    };
  }, [hoverPos, comparison, mode, invertY, velocityLabel, scale, PLOT_W, x0, x1, f0, f1]);

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
    ctx.font = FONT;

    const { positions } = comparison;
    const { ys: fs, observed: observed_grid, predicted: predicted_grid, residual: residual_grid } = along(
      comparison,
      mode,
    );
    const np = positions.length;
    const nf = fs.length;
    const fMin = fs[0];
    const fMax = fs[fs.length - 1];
    const fSpan = fMax - fMin || 1;

    const xMin = positions[0];
    const xMax = positions[np - 1];

    const xOf = (p: number) =>
      ML + ((p - x0) / (x1 - x0)) * PLOT_W;

    // Cell boundaries clipped to the actual position range
    const cellEdges: number[] = new Array(np + 1);

    cellEdges[0] = xMin;
    cellEdges[np] = xMax;

    for (let i = 1; i < np; i++) {
      cellEdges[i] = (positions[i - 1] + positions[i]) / 2;
    }

    // Only the cells on show, the same in the three panels: a column's rows
    // run from rowTop at the top (the highest frequency, or the shortest
    // wavelength) to rowBottom, and each panel's edges clip those they cut
    // through. The image's rows are the data's entries bottom up for
    // frequencies, top down for wavelengths.
    const [i0, i1] = visibleColumns(cellEdges, x0, x1);
    const rowTop = invertY ? fMin : fMin + fSpan;
    const rowBottom = invertY ? fMin + fSpan : fMin;
    const [k0, k1] = visibleCells(nf, rowTop, rowBottom, f0, f1);
    const [j0, j1] = invertY ? [k0, k1] : [nf - k1, nf - k0];

    // The colour scales span the values on show: all of them in the full
    // view, those in the window when zoomed.
    const merge = (a: [number, number] | null, b: [number, number] | null): [number, number] | null =>
      a && b ? [Math.min(a[0], b[0]), Math.max(a[1], b[1])] : (a ?? b);
    // Obs and pred share one color scale so the two panels are directly comparable.
    const [zMin, zMax] =
      merge(valueRange(observed_grid, i0, i1, j0, j1), valueRange(predicted_grid, i0, i1, j0, j1)) ??
      merge(valueRange(observed_grid, 0, np, 0, nf), valueRange(predicted_grid, 0, np, 0, nf)) ??
      [0, 1];
    // The residuals' scale stays symmetric about zero.
    const residuals = valueRange(residual_grid, i0, i1, j0, j1) ?? valueRange(residual_grid, 0, np, 0, nf);
    const resLim = (residuals && Math.max(Math.abs(residuals[0]), Math.abs(residuals[1]))) || 1;

    function drawPanel(
      ctx: CanvasRenderingContext2D,
      top: number,
      grid: (number | null)[][],
      vMin: number,
      vMax: number,
      colormap: (t: number) => [number, number, number],
      legendLabel: string,
      // The colour bar's values: velocities whole, as a pseudo-section's; residuals to 0.1 %.
      decimals: number,
    ) {
      const vSpan = vMax - vMin || 1;
      const yOf = (f: number) =>
        invertY ? top + ((f - f0) / (f1 - f0)) * PLOT_H : top + PLOT_H - ((f - f0) / (f1 - f0)) * PLOT_H;

      const yTop = yOf(rowTop + (k0 / nf) * (rowBottom - rowTop));
      const yBottom = yOf(rowTop + (k1 / nf) * (rowBottom - rowTop));
      ctx.save();
      ctx.beginPath();
      ctx.rect(ML, top, PLOT_W, PLOT_H);
      ctx.clip();
      for (let i = i0; i < i1 && k1 > k0; i++) {
        const xLeft = xOf(cellEdges[i]);
        const xRight = xOf(cellEdges[i + 1]);
        const off = document.createElement("canvas");
        off.width = 1;
        off.height = nf;
        const octx = off.getContext("2d");
        if (!octx) continue;
        const imgData = octx.createImageData(1, nf);
        for (let j = 0; j < nf; j++) {
          const v = grid[i][j];
          const y = invertY ? j : nf - 1 - j;
          const idx = y * 4;
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
      ctx.restore();
      if (marker !== undefined && marker >= x0 && marker <= x1) drawMarker(ctx, xOf(marker), top, top + PLOT_H);

      ctx.strokeStyle = palette.axis;
      ctx.lineWidth = 1;
      ctx.strokeRect(ML, top, PLOT_W, PLOT_H);

      ctx.fillStyle = palette.tick;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const nfTicks = 4;
      const fDecimals = tickDecimals((f1 - f0) / nfTicks, 1);
      for (const f of evenTicks(f0, f1, nfTicks)) {
        const py = yOf(f);
        ctx.beginPath();
        ctx.moveTo(ML - 4, py);
        ctx.lineTo(ML, py);
        ctx.stroke();
        ctx.fillText(f.toFixed(fDecimals), ML - 7, py);
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const { p, label } of positionTicks(positions, x0, x1)) {
        const x = xOf(p);
        ctx.beginPath();
        ctx.moveTo(x, top + PLOT_H);
        ctx.lineTo(x, top + PLOT_H + 4);
        ctx.stroke();
        ctx.fillText(label, x, top + PLOT_H + 6);
      }

      // color legend — offscreen image + drawImage, like the pcolormesh
      // columns above, to avoid antialiasing seams from per-row fillRect.
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
        ctx.drawImage(legendOff, 0, 0, 1, legendRes, legendX, top, legendW, PLOT_H);
      }
      ctx.strokeStyle = palette.axis;
      ctx.strokeRect(legendX, top, legendW, PLOT_H);
      ctx.fillStyle = palette.tick;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const nLegendTicks = 4;
      for (let i = 0; i <= nLegendTicks; i++) {
        const v = vMin + (i / nLegendTicks) * vSpan;
        const py = top + PLOT_H - (i / nLegendTicks) * PLOT_H;
        ctx.fillText(v.toFixed(decimals), legendX + legendW + 6, py);
      }

      ctx.fillStyle = palette.title;
      verticalLabel(ctx, legendLabel, TOTAL_W - 6, top + PLOT_H / 2, PLOT_H - 8);
      verticalLabel(ctx, invertY ? "Wavelength (m)" : "Frequency (Hz)", 16, top + PLOT_H / 2, PLOT_H - 8);
    }

    const top1 = MT;
    const top2 = top1 + PLOT_H + PANEL_GAP;
    const top3 = top2 + PLOT_H + PANEL_GAP;

    drawPanel(ctx, top1, observed_grid, zMin, zMax, cividis, `Picked ${velocityLabel.toLowerCase()}`, 0);
    drawPanel(ctx, top2, predicted_grid, zMin, zMax, cividis, `Modelled ${velocityLabel.toLowerCase()}`, 0);
    drawPanel(ctx, top3, residual_grid, -resLim, resLim, bwr, "Residual (%)", 1);

    ctx.fillStyle = palette.title;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Position (m)", ML + PLOT_W / 2, TOTAL_H - 4);
  }, [comparison, mode, invertY, velocityLabel, palette, scale, TOTAL_W, PLOT_W, x0, x1, f0, f1, marker]);

  const click = useClick(
    TOTAL_W,
    TOTAL_H,
    onPick &&
      ((x, y) => {
        const inside = PANELS.some(
          (panel) => x >= panel.left && x <= panel.left + panel.width && y >= panel.top && y <= panel.top + panel.height,
        );
        if (!inside) return;
        const positions = comparison.positions;
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
