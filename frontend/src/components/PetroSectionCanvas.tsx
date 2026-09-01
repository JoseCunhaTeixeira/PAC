import { useEffect, useMemo, useRef } from "react";
import { HoverTooltip } from "./HoverTooltip";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { nearestIndex, useCanvasHover } from "./useCanvasHover";
import { useContainerWidth } from "./useContainerWidth";

export interface PetroSectionData {
  positions: number[];
  elevations: number[];
  soil_grid: (string | null)[][];
  n_grid: (number | null)[][];
  water_table_elevations: number[];
}

// Fixed order/colors shared by every soil-type plot, matching sigpipe's
// dataio.plot_config.SOIL_TYPE_COLORS -- the same soil always reads as the
// same color as in sigpipe's own matplotlib section plots.
const SOIL_ORDER = ["clay", "silt", "loam", "sand"];
const SOIL_COLORS: Record<string, string> = {
  clay: "#8B5A2B",
  silt: "#C2B280",
  loam: "#6B6B3A",
  sand: "#F2D57E",
};

// matplotlib's tab10/tab20 categorical palettes, mirroring sigpipe's
// dataio.plot_config.n_value_colors (tab10 for <=10 distinct values, else
// tab20), so N-value colors match sigpipe's own plots too.
const TAB10 = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
];
const TAB20 = [
  "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
  "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
  "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
  "#17becf", "#9edae5",
];

function nValueColors(values: number[]): Record<number, string> {
  const sorted = Array.from(new Set(values.map((v) => Math.round(v)))).sort((a, b) => a - b);
  const palette = sorted.length <= 10 ? TAB10 : TAB20;
  const out: Record<number, string> = {};
  sorted.forEach((v, i) => {
    out[v] = palette[i % palette.length];
  });
  return out;
}

const ML = 60, MR = 130, MT = 16, MB = 40, PANEL_GAP = 40, LEGEND_ITEM_H = 18;
const PLOT_W = 640, PLOT_H = 170;
const FONT = CANVAS_FONT;
const TOTAL_W = ML + PLOT_W + MR;
const TOTAL_H = MT + 2 * PLOT_H + PANEL_GAP + MB;

// Soil-type and N-value depth section, mirroring sigpipe's
// plot_petro_models_section (two stacked panels, water table as a dashed
// step per position on both panels) as an interactive canvas instead of a
// static plot.
export function PetroSectionCanvas({ section }: { section: PetroSectionData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const palette = useMemo(() => canvasPalette(theme), [theme]);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const scale = containerWidth > 0 ? Math.min(containerWidth / TOTAL_W, 1) : 1;
  const { pos: hoverPos, onMouseMove, onMouseLeave } = useCanvasHover(scale);

  const nColors = useMemo(() => {
    const values = section.n_grid.flatMap((row) => row.filter((v): v is number => v !== null));
    return nValueColors(values);
  }, [section.n_grid]);

  const hover = useMemo(() => {
    if (!hoverPos) return null;
    if (hoverPos.x < ML || hoverPos.x > ML + PLOT_W) return null;

    const { positions, elevations, soil_grid, n_grid } = section;
    const np = positions.length;

    const top1 = MT;
    const top2 = top1 + PLOT_H + PANEL_GAP;

    let panel: "soil" | "n";
    let top: number;
    if (hoverPos.y >= top1 && hoverPos.y <= top1 + PLOT_H) {
      panel = "soil";
      top = top1;
    } else if (hoverPos.y >= top2 && hoverPos.y <= top2 + PLOT_H) {
      panel = "n";
      top = top2;
    } else {
      return null;
    }

    const xMin = positions[0];
    const xMax = positions[np - 1];
    const xSpan = xMax - xMin || 1;
    const position = xMin + ((hoverPos.x - ML) / PLOT_W) * xSpan;

    const zMin = elevations[elevations.length - 1];
    const zMax = elevations[0];
    const zSpan = zMax - zMin || 1;
    const elevation = zMax - ((hoverPos.y - top) / PLOT_H) * zSpan;

    const posIdx = nearestIndex(positions, position);
    const zIdx = nearestIndex(elevations, elevation);

    const line =
      panel === "soil"
        ? `Soil: ${soil_grid[posIdx]?.[zIdx] ?? "—"}`
        : `N: ${n_grid[posIdx]?.[zIdx] ?? "—"}`;

    return {
      px: hoverPos.x * scale,
      py: hoverPos.y * scale,
      lines: [
        `Position: ${positions[posIdx].toFixed(2)} m`,
        `Elevation: ${elevations[zIdx].toFixed(2)} m`,
        line,
      ],
    };
  }, [hoverPos, section, scale]);

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

    const { positions, elevations, soil_grid, n_grid, water_table_elevations } = section;
    const np = positions.length;
    const nz = elevations.length;

    const zMin = elevations[elevations.length - 1];
    const zMax = elevations[0];
    const zSpan = zMax - zMin || 1;

    const xMin = positions[0];
    const xMax = positions[np - 1];
    const xSpan = xMax - xMin || 1;
    const xOf = (p: number) => ML + ((p - xMin) / xSpan) * PLOT_W;

    const cellEdges: number[] = new Array(np + 1);
    cellEdges[0] = xMin;
    cellEdges[np] = xMax;
    for (let i = 1; i < np; i++) cellEdges[i] = (positions[i - 1] + positions[i]) / 2;

    function drawCategoricalPanel<T>(
      top: number,
      grid: (T | null)[][],
      colorOf: (v: T) => string,
    ) {
      const yOf = (z: number) => top + ((zMax - z) / zSpan) * PLOT_H;

      ctx!.imageSmoothingEnabled = false;
      for (let i = 0; i < np; i++) {
        const xLeft = Math.round(xOf(cellEdges[i]));
        const xRight = Math.round(xOf(cellEdges[i + 1]));
        const off = document.createElement("canvas");
        off.width = 1;
        off.height = nz;
        const octx = off.getContext("2d");
        if (!octx) continue;
        const imgData = octx.createImageData(1, nz);
        for (let j = 0; j < nz; j++) {
          const v = grid[i][j];
          const idx = j * 4;
          if (v === null) {
            imgData.data[idx + 3] = 0;
            continue;
          }
          const hex = colorOf(v);
          imgData.data[idx] = parseInt(hex.slice(1, 3), 16);
          imgData.data[idx + 1] = parseInt(hex.slice(3, 5), 16);
          imgData.data[idx + 2] = parseInt(hex.slice(5, 7), 16);
          imgData.data[idx + 3] = 255;
        }
        octx.putImageData(imgData, 0, 0);
        ctx!.drawImage(off, 0, 0, 1, nz, xLeft, top, Math.max(1, xRight - xLeft), PLOT_H);
      }
      ctx!.imageSmoothingEnabled = true;

      ctx!.strokeStyle = palette.axis;
      ctx!.lineWidth = 1;
      ctx!.strokeRect(ML, top, PLOT_W, PLOT_H);

      ctx!.fillStyle = palette.tick;
      ctx!.textAlign = "right";
      ctx!.textBaseline = "middle";
      const nzTicks = 5;
      for (let i = 0; i <= nzTicks; i++) {
        const z = zMin + (i / nzTicks) * zSpan;
        const py = yOf(z);
        ctx!.beginPath();
        ctx!.moveTo(ML - 4, py);
        ctx!.lineTo(ML, py);
        ctx!.stroke();
        ctx!.fillText(z.toFixed(1), ML - 7, py);
      }

      ctx!.save();
      ctx!.translate(16, top + PLOT_H / 2);
      ctx!.rotate(-Math.PI / 2);
      ctx!.textAlign = "center";
      ctx!.fillStyle = palette.title;
      ctx!.fillText("Elevation [m]", 0, 0);
      ctx!.restore();

      return { yOf };
    }

    // Water table as a dashed step, flat across each position's own cell
    // width (matching the piecewise-constant soil/N blocks) rather than a
    // continuous line interpolated between positions -- each position's
    // water table depth is its own discrete value, not a slope toward its
    // neighbors'.
    function drawWaterTable(yOf: (z: number) => number) {
      ctx!.save();
      ctx!.strokeStyle = "darkblue";
      ctx!.lineWidth = 1.5;
      ctx!.setLineDash([5, 4]);
      ctx!.beginPath();
      for (let i = 0; i < np; i++) {
        const xLeft = xOf(cellEdges[i]);
        const xRight = xOf(cellEdges[i + 1]);
        const y = yOf(water_table_elevations[i]);
        ctx!.moveTo(xLeft, y);
        ctx!.lineTo(xRight, y);
      }
      ctx!.stroke();
      ctx!.restore();
    }

    function drawWaterTableLegend(top: number, afterCount: number) {
      const legendX = ML + PLOT_W + 16;
      const y = top + 8 + afterCount * LEGEND_ITEM_H;
      ctx!.save();
      ctx!.strokeStyle = "darkblue";
      ctx!.lineWidth = 1.5;
      ctx!.setLineDash([5, 4]);
      ctx!.beginPath();
      ctx!.moveTo(legendX, y);
      ctx!.lineTo(legendX + 12, y);
      ctx!.stroke();
      ctx!.restore();
      ctx!.fillStyle = palette.title;
      ctx!.textAlign = "left";
      ctx!.textBaseline = "middle";
      ctx!.fillText("Water table", legendX + 18, y);
    }

    const top1 = MT;
    const top2 = top1 + PLOT_H + PANEL_GAP;

    const { yOf: yOfSoil } = drawCategoricalPanel(top1, soil_grid, (s) => SOIL_COLORS[s] ?? "#999999");
    drawWaterTable(yOfSoil);

    const { yOf: yOfN } = drawCategoricalPanel(top2, n_grid, (n) => nColors[Math.round(n)] ?? "#999999");
    drawWaterTable(yOfN);

    // x-axis ticks only on the bottom (N) panel.
    ctx.fillStyle = palette.tick;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const nxTicks = Math.min(8, np - 1);
    for (let i = 0; i <= nxTicks; i++) {
      const idx = nxTicks > 0 ? Math.round((i / nxTicks) * (np - 1)) : 0;
      const p = positions[idx];
      const x = xOf(p);
      ctx.beginPath();
      ctx.moveTo(x, top2 + PLOT_H);
      ctx.lineTo(x, top2 + PLOT_H + 4);
      ctx.stroke();
      ctx.fillText(p.toFixed(1), x, top2 + PLOT_H + 6);
    }
    ctx.fillStyle = palette.title;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Position [m]", ML + PLOT_W / 2, TOTAL_H - 4);

    // Categorical legends: swatch + label, stacked vertically.
    function drawLegend(top: number, items: { color: string; label: string }[]) {
      const legendX = ML + PLOT_W + 16;
      ctx!.textAlign = "left";
      ctx!.textBaseline = "middle";
      items.forEach((item, i) => {
        const y = top + 8 + i * LEGEND_ITEM_H;
        ctx!.fillStyle = item.color;
        ctx!.fillRect(legendX, y - 6, 12, 12);
        ctx!.strokeStyle = palette.axis;
        ctx!.strokeRect(legendX, y - 6, 12, 12);
        ctx!.fillStyle = palette.title;
        ctx!.fillText(item.label, legendX + 18, y);
      });
    }

    const soilItems = SOIL_ORDER.map((soil) => ({ color: SOIL_COLORS[soil], label: soil }));
    drawLegend(top1, soilItems);
    drawWaterTableLegend(top1, soilItems.length);

    // Object.entries on numeric-like keys always iterates ascending
    // regardless of insertion order, so nColors' entries come out lowest N
    // first -- reversed here so the legend reads highest-N-at-top,
    // lowest-N-at-bottom (drawLegend draws top-to-bottom in array order).
    const nItems = Object.entries(nColors)
      .map(([n, color]) => ({ color, label: `N = ${n}` }))
      .reverse();
    drawLegend(top2, nItems);
    drawWaterTableLegend(top2, nItems.length);
  }, [section, palette, scale, nColors]);

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
