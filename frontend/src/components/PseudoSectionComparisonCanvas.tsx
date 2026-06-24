import { useEffect, useMemo, useRef } from "react";
import { viridis, bwr } from "./colormaps";
import { HoverTooltip } from "./HoverTooltip";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { nearestIndex, useCanvasHover } from "./useCanvasHover";
import { useContainerWidth } from "./useContainerWidth";

export interface PseudoSectionComparisonData {
  positions: number[];
  fs: number[];
  observed_grid: (number | null)[][];
  predicted_grid: (number | null)[][];
  residual_grid: (number | null)[][];
}

const ML = 60, MR = 130, MT = 16, MB = 40, PANEL_GAP = 30;
const PLOT_W = 640, PLOT_H = 130;
const FONT = CANVAS_FONT;
const TOTAL_W = ML + PLOT_W + MR;
const TOTAL_H = MT + 3 * PLOT_H + 2 * PANEL_GAP + MB;

// Observed/predicted/residual pseudo-sections stacked vertically, mirroring
// sigproc's `plot_pseudo_section_comparison` (obs+pred share one viridis
// scale so they're directly comparable; residual uses a symmetric bwr scale).
export function PseudoSectionComparisonCanvas({
  comparison,
  velocityLabel,
}: {
  comparison: PseudoSectionComparisonData;
  velocityLabel: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const palette = canvasPalette(theme);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const scale = containerWidth > 0 ? Math.min(containerWidth / TOTAL_W, 1) : 1;
  const { pos: hoverPos, onMouseMove, onMouseLeave } = useCanvasHover(scale);

  const hover = useMemo(() => {
    if (!hoverPos) return null;
    if (hoverPos.x < ML || hoverPos.x > ML + PLOT_W) return null;

    const { positions, fs, observed_grid, predicted_grid, residual_grid } = comparison;
    const np = positions.length;

    const top1 = MT;
    const top2 = top1 + PLOT_H + PANEL_GAP;
    const top3 = top2 + PLOT_H + PANEL_GAP;

    let top: number;
    let grid: (number | null)[][];
    let label: string;
    if (hoverPos.y >= top1 && hoverPos.y <= top1 + PLOT_H) {
      top = top1;
      grid = observed_grid;
      label = `Obs ${velocityLabel}`;
    } else if (hoverPos.y >= top2 && hoverPos.y <= top2 + PLOT_H) {
      top = top2;
      grid = predicted_grid;
      label = `Pred ${velocityLabel}`;
    } else if (hoverPos.y >= top3 && hoverPos.y <= top3 + PLOT_H) {
      top = top3;
      grid = residual_grid;
      label = "Residuals [%]";
    } else {
      return null;
    }

    const xMin = positions[0];
    const xMax = positions[np - 1];
    const xSpan = xMax - xMin || 1;
    const position = xMin + ((hoverPos.x - ML) / PLOT_W) * xSpan;

    const fMin = fs[0];
    const fMax = fs[fs.length - 1];
    const fSpan = fMax - fMin || 1;
    const freq = fMin + ((top + PLOT_H - hoverPos.y) / PLOT_H) * fSpan;

    const posIdx = nearestIndex(positions, position);
    const fIdx = nearestIndex(fs, freq);
    const value = grid[posIdx]?.[fIdx] ?? null;

    return {
      px: hoverPos.x * scale,
      py: hoverPos.y * scale,
      lines: [
        `Position: ${positions[posIdx].toFixed(2)} m`,
        `Frequency: ${fs[fIdx].toFixed(2)} Hz`,
        `${label}: ${value === null ? "—" : value.toFixed(1)}`,
      ],
    };
  }, [hoverPos, comparison, velocityLabel, scale]);

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

    const { positions, fs, observed_grid, predicted_grid, residual_grid } = comparison;
    const np = positions.length;
    const nf = fs.length;
    const fMin = fs[0];
    const fMax = fs[fs.length - 1];
    const fSpan = fMax - fMin || 1;

    const xMin = positions[0];
    const xMax = positions[np - 1];
    const xSpan = xMax - xMin || 1;

    const xOf = (p: number) =>
      ML + ((p - xMin) / xSpan) * PLOT_W;

    // Cell boundaries clipped to the actual position range
    const cellEdges: number[] = new Array(np + 1);

    cellEdges[0] = xMin;
    cellEdges[np] = xMax;

    for (let i = 1; i < np; i++) {
      cellEdges[i] = (positions[i - 1] + positions[i]) / 2;
    }

    // Obs and pred share one color scale so the two panels are directly comparable.
    let zMin = Infinity, zMax = -Infinity;
    for (const grid of [observed_grid, predicted_grid]) {
      for (const row of grid) {
        for (const v of row) {
          if (v !== null) {
            if (v < zMin) zMin = v;
            if (v > zMax) zMax = v;
          }
        }
      }
    }
    if (!Number.isFinite(zMin)) { zMin = 0; zMax = 1; }

    let resLim = 0;
    for (const row of residual_grid) {
      for (const v of row) {
        if (v !== null) resLim = Math.max(resLim, Math.abs(v));
      }
    }
    if (resLim === 0) resLim = 1;

    function drawPanel(
      ctx: CanvasRenderingContext2D,
      top: number,
      grid: (number | null)[][],
      vMin: number,
      vMax: number,
      colormap: (t: number) => [number, number, number],
      legendLabel: string,
    ) {
      const vSpan = vMax - vMin || 1;
      const yOf = (f: number) => top + PLOT_H - ((f - fMin) / fSpan) * PLOT_H;

      for (let i = 0; i < np; i++) {
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
          const y = nf - 1 - j;
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
        ctx.drawImage(off, 0, 0, 1, nf, xLeft, top, Math.max(1, xRight - xLeft), PLOT_H);
      }

      ctx.strokeStyle = palette.axis;
      ctx.lineWidth = 1;
      ctx.strokeRect(ML, top, PLOT_W, PLOT_H);

      ctx.fillStyle = palette.tick;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const nfTicks = 4;
      for (let i = 0; i <= nfTicks; i++) {
        const f = fMin + (i / nfTicks) * fSpan;
        const py = yOf(f);
        ctx.beginPath();
        ctx.moveTo(ML - 4, py);
        ctx.lineTo(ML, py);
        ctx.stroke();
        ctx.fillText(f.toFixed(1), ML - 7, py);
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const nxTicks = Math.min(8, np - 1);
      for (let i = 0; i <= nxTicks; i++) {
        const idx = nxTicks > 0 ? Math.round((i / nxTicks) * (np - 1)) : 0;
        const p = positions[idx];
        const x = xOf(p);
        ctx.beginPath();
        ctx.moveTo(x, top + PLOT_H);
        ctx.lineTo(x, top + PLOT_H + 4);
        ctx.stroke();
        ctx.fillText(p.toFixed(1), x, top + PLOT_H + 6);
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
        ctx.fillText(v.toFixed(1), legendX + legendW + 6, py);
      }

      ctx.fillStyle = palette.title;
      ctx.save();
      ctx.translate(TOTAL_W - 14, top + PLOT_H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(legendLabel, 0, 0);
      ctx.restore();

      ctx.save();
      ctx.translate(16, top + PLOT_H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText("Frequency [Hz]", 0, 0);
      ctx.restore();
    }

    const top1 = MT;
    const top2 = top1 + PLOT_H + PANEL_GAP;
    const top3 = top2 + PLOT_H + PANEL_GAP;

    drawPanel(ctx, top1, observed_grid, zMin, zMax, viridis, `Obs ${velocityLabel}`);
    drawPanel(ctx, top2, predicted_grid, zMin, zMax, viridis, `Pred ${velocityLabel}`);
    drawPanel(ctx, top3, residual_grid, -resLim, resLim, bwr, "Residuals [%]");

    ctx.fillStyle = palette.title;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Position [m]", ML + PLOT_W / 2, TOTAL_H - 4);
  }, [comparison, velocityLabel, theme, scale]);

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
