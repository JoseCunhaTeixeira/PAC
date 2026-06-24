import { useEffect, useMemo, useRef } from "react";
import { cividis } from "./colormaps";
import { HoverTooltip } from "./HoverTooltip";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { nearestIndex, useCanvasHover } from "./useCanvasHover";
import { useContainerWidth } from "./useContainerWidth";

export interface PseudoSection {
  positions: number[];
  fs_grid: number[];
  velocities_by_frequency: (number | null)[][];
  lambdas_grid: number[];
  velocities_by_wavelength: (number | null)[][];
}

const ML = 60, MR = 120, MT = 16, MB = 40;
const PLOT_W = 640;
const FONT = CANVAS_FONT;
const TOTAL_W = ML + PLOT_W + MR;

export function PseudoSectionCanvas({
  section,
  mode,
  height = 320,
}: {
  section: PseudoSection;
  mode: "frequency" | "wavelength";
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

    const yGrid = mode === "frequency" ? section.fs_grid : section.lambdas_grid;
    const velocities = mode === "frequency" ? section.velocities_by_frequency : section.velocities_by_wavelength;
    const yLabel = mode === "frequency" ? "Frequency" : "Wavelength";
    const yUnit = mode === "frequency" ? "Hz" : "m";
    const invertY = mode === "wavelength";
    const positions = section.positions;
    const np = positions.length;

    const xMin = positions[0];
    const xMax = positions[np - 1];
    const xSpan = xMax - xMin || 1;
    const position = xMin + ((hoverPos.x - ML) / PLOT_W) * xSpan;

    const yMin = yGrid[0];
    const yMax = yGrid[yGrid.length - 1];
    const ySpan = yMax - yMin || 1;
    const yValue = invertY
      ? yMin + ((hoverPos.y - MT) / PLOT_H) * ySpan
      : yMin + ((MT + PLOT_H - hoverPos.y) / PLOT_H) * ySpan;

    const posIdx = nearestIndex(positions, position);
    const yIdx = nearestIndex(yGrid, yValue);
    const value = velocities[posIdx]?.[yIdx] ?? null;

    return {
      px: hoverPos.x * scale,
      py: hoverPos.y * scale,
      lines: [
        `Position: ${positions[posIdx].toFixed(2)} m`,
        `${yLabel}: ${yGrid[yIdx].toFixed(2)} ${yUnit}`,
        `Velocity: ${value === null ? "—" : value.toFixed(1)} m/s`,
      ],
    };
  }, [hoverPos, section, mode, scale, PLOT_H]);

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
    const yLabel = mode === "frequency" ? "Frequency [Hz]" : "Wavelength [m]";
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
        ? MT + ((y - yMin) / ySpan) * PLOT_H
        : MT + PLOT_H - ((y - yMin) / ySpan) * PLOT_H;

    let vMin = Infinity, vMax = -Infinity;
    for (const row of velocities) {
      for (const v of row) {
        if (v !== null) {
          if (v < vMin) vMin = v;
          if (v > vMax) vMax = v;
        }
      }
    }
    if (!Number.isFinite(vMin)) { vMin = 0; vMax = 1; }
    const vSpan = vMax - vMin || 1;

    // Plot exactly between min(position) and max(position)
    const xMin = positions[0];
    const xMax = positions[np - 1];
    const xSpan = xMax - xMin || 1;

    const xOf = (p: number) =>
      ML + ((p - xMin) / xSpan) * PLOT_W;

    // Midpoint boundaries, clipped to plot limits
    const columnEdges: number[] = new Array(np + 1);

    columnEdges[0] = xMin;
    columnEdges[np] = xMax;

    for (let i = 1; i < np; i++) {
      columnEdges[i] = (positions[i - 1] + positions[i]) / 2;
    }

    for (let i = 0; i < np; i++) {
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
        0,
        1,
        ny,
        xLeft,
        MT,
        Math.max(1, xRight - xLeft),
        PLOT_H
      );
    }

    // axes
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.font = FONT;
    ctx.strokeRect(ML, MT, PLOT_W, PLOT_H);

    ctx.fillStyle = palette.tick;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const nyTicks = 6;
    for (let i = 0; i <= nyTicks; i++) {
      const y = yMin + (i / nyTicks) * ySpan;
      const py = yOf(y);
      ctx.beginPath();
      ctx.moveTo(ML - 4, py);
      ctx.lineTo(ML, py);
      ctx.stroke();
      ctx.fillText(y.toFixed(1), ML - 7, py);
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
    ctx.fillText("Position [m]", ML + PLOT_W / 2, TOTAL_H - 4);
    ctx.save();
    ctx.translate(16, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    ctx.save();
    ctx.translate(TOTAL_W - 14, MT + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Phase velocity [m/s]", 0, 0);
    ctx.restore();
  }, [section, mode, height, theme, scale]);

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
