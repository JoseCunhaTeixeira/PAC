import { useEffect, useMemo, useRef, useState } from "react";
import { CANVAS_FONT, canvasFont, canvasPalette, useTheme } from "../../theme";
import { useContainerWidth } from "../useContainerWidth";
import { TooltipLines } from "../HoverTooltip";
import { evenTicks, tickDecimals, useZoom, type PlotRect } from "../useZoom";
import { ZoomReset, ZoomSelection } from "../ZoomOverlay";
import { num, parameterLabel } from "./format";
import { vizPalette } from "./palette";
import type { ChainTraces, Marginal } from "./types";

// The chains of one window's inversion: each chain's samples of a parameter along the run
// (one line a chain: chains that agree overlap, a stuck chain stays flat), and each
// parameter's marginal between its prior's bounds (the chains pooled, each chain's outline
// over it; a flat prior would fill the dashed level).

const ML = 56, MR = 10, MT = 8, MB = 34;

/** Whole ticks about `n` apart over [lo, hi], each a multiple of 1, 2 or 5 times a power of 10. */
function roundTicks(lo: number, hi: number, n: number): number[] {
  const raw = (hi - lo) / n;
  if (!(raw > 0)) return [lo];
  const power = 10 ** Math.floor(Math.log10(raw));
  const m = raw / power;
  const step = Math.max(1, (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * power);
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
  return ticks;
}

function useLogical(width: number, height: number) {
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null);
  const handlers = {
    onMouseMove: (e: React.MouseEvent<HTMLCanvasElement>) => {
      const box = e.currentTarget.getBoundingClientRect();
      setMouse({ x: ((e.clientX - box.left) / box.width) * width, y: ((e.clientY - box.top) / box.height) * height });
    },
    onMouseLeave: () => setMouse(null),
  };
  return [mouse, handlers] as const;
}

export function ChainLegend({ n }: { n: number }) {
  const palette = vizPalette(useTheme());
  return (
    <span className="viz-legend">
      {Array.from({ length: n }, (_, i) => (
        <span key={i}>
          <i style={{ background: palette.chains[i % palette.chains.length] }} />
          chain {i + 1}
        </span>
      ))}
    </span>
  );
}

export function ChainTracesCanvas({
  traces,
  prior,
  height = 190,
}: {
  traces: ChainTraces;
  prior?: [number, number];
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const axes = canvasPalette(theme);
  const palette = vizPalette(theme);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const width = Math.max(280, Math.floor(containerWidth || 600));
  const plotW = width - ML - MR;
  const plotH = height - MT - MB;
  const [mouse, handlers] = useLogical(width, height);

  const n = Math.max(0, ...traces.chains.map((chain) => chain.length));
  const extent = useMemo(() => {
    const values = traces.chains.flat();
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const pad = (hi - lo) * 0.06 || Math.abs(hi) * 0.05 || 1;
    return { x: [0, Math.max(1, (n - 1) * traces.step)] as const, y: [lo - pad, hi + pad] as const };
  }, [traces, n]);
  const plots: PlotRect[] = [{ left: ML, top: MT, width: plotW, height: plotH, xAxis: MB, yAxis: ML }];
  const zoom = useZoom({ extent, plots, width, height, resetKey: traces.parameter });
  const [x0, x1] = zoom.view.x;
  const [y0, y1] = zoom.view.y;

  const hover = useMemo(() => {
    if (!mouse || mouse.x < ML || mouse.x > ML + plotW || mouse.y < MT || mouse.y > MT + plotH) return null;
    const sample = x0 + ((mouse.x - ML) / plotW) * (x1 - x0);
    const k = Math.max(0, Math.min(n - 1, Math.round(sample / traces.step)));
    return {
      lines: [
        `Sample ${k * traces.step}`,
        ...traces.chains.map((chain, i) => `Chain ${i + 1}: ${num(chain[k])}`),
      ],
    };
  }, [mouse, plotW, plotH, x0, x1, n, traces]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const xOf = (x: number) => ML + ((x - x0) / (x1 - x0)) * plotW;
    const yOf = (y: number) => MT + plotH - ((y - y0) / (y1 - y0)) * plotH;

    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, MT, plotW, plotH);
    ctx.clip();
    if (prior) {
      ctx.strokeStyle = palette.limit;
      ctx.setLineDash([4, 3]);
      for (const bound of prior) {
        ctx.beginPath();
        ctx.moveTo(ML, yOf(bound));
        ctx.lineTo(ML + plotW, yOf(bound));
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    ctx.lineWidth = 1;
    traces.chains.forEach((chain, c) => {
      ctx.strokeStyle = palette.chains[c % palette.chains.length];
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      chain.forEach((v, k) => {
        const x = xOf(k * traces.step);
        if (k === 0) ctx.moveTo(x, yOf(v));
        else ctx.lineTo(x, yOf(v));
      });
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
    ctx.restore();

    ctx.strokeStyle = axes.axis;
    ctx.strokeRect(ML + 0.5, MT + 0.5, plotW - 1, plotH - 1);
    ctx.fillStyle = axes.tick;
    ctx.font = canvasFont(11);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yDecimals = tickDecimals((y1 - y0) / 3);
    for (const t of evenTicks(y0, y1, 3)) ctx.fillText(t.toFixed(yDecimals), ML - 5, yOf(t));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of roundTicks(x0, x1, 5)) ctx.fillText(t.toLocaleString("en-US"), xOf(t), MT + plotH + 4);
    ctx.fillStyle = axes.title;
    ctx.font = CANVAS_FONT;
    ctx.textBaseline = "bottom";
    ctx.fillText("Saved sample", ML + plotW / 2, height - 1);
    ctx.save();
    ctx.translate(12, MT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText(parameterLabel(traces.parameter), 0, 0);
    ctx.restore();
  }, [traces, prior, width, height, plotW, plotH, x0, x1, y0, y1, palette, axes]);

  return (
    <div ref={containerRef} style={{ width: "100%", position: "relative" }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", cursor: zoom.cursorAt(mouse) }}
        {...handlers}
        onMouseDown={zoom.onMouseDown}
        onDoubleClick={zoom.onDoubleClick}
      />
      <ZoomSelection box={zoom.selection} />
      <ZoomReset zoomed={zoom.zoomed} onReset={zoom.reset} style={{ top: 0, right: MR }} />
      {hover && mouse && (
        <div
          className="viz-tooltip"
          style={mouse.x > width / 2 ? { right: width - mouse.x + 12, top: mouse.y + 12 } : { left: mouse.x + 12, top: mouse.y + 12 }}
        >
          <TooltipLines lines={hover.lines} />
        </div>
      )}
    </div>
  );
}

const SM_ML = 34, SM_MR = 8, SM_MT = 18, SM_MB = 18, SM_H = 120;

/** Every parameter's marginal, small multiples as many a row as fit. */
export function MarginalsGrid({ marginals }: { marginals: Marginal[] }) {
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const width = Math.max(260, Math.floor(containerWidth || 600));
  const columns = Math.max(1, Math.min(marginals.length, Math.floor(width / 190)));
  const cell = Math.floor((width - (columns - 1) * 10) / columns);
  return (
    <div ref={containerRef} className="viz-plots">
      {marginals.map((marginal) => (
        <MarginalCanvas key={marginal.parameter} marginal={marginal} width={cell} />
      ))}
    </div>
  );
}

function MarginalCanvas({ marginal, width }: { marginal: Marginal; width: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const axes = canvasPalette(theme);
  const palette = vizPalette(theme);
  const height = SM_H;
  const plotW = width - SM_ML - SM_MR;
  const plotH = height - SM_MT - SM_MB;
  const [mouse, handlers] = useLogical(width, height);

  const bins = marginal.counts[0]?.length ?? 0;
  const pooled = useMemo(
    () => Array.from({ length: bins }, (_, b) => marginal.counts.reduce((sum, chain) => sum + chain[b], 0)),
    [marginal, bins],
  );
  const total = pooled.reduce((a, b) => a + b, 0);
  const binW = (marginal.high - marginal.low) / Math.max(bins, 1);
  // Densities: each chain scaled as if it held every sample, so it overlays the pooled bars.
  const n = marginal.counts.length;
  const top = Math.max(1, ...pooled, ...marginal.counts.flat().map((c) => c * n));
  const zoom = useZoom({
    extent: { x: [marginal.low, marginal.high], y: [0, top * 1.08] },
    plots: [{ left: SM_ML, top: SM_MT, width: plotW, height: plotH, xAxis: SM_MB, yAxis: SM_ML }],
    width,
    height,
  });
  const [x0, x1] = zoom.view.x;
  const [y0, y1] = zoom.view.y;

  const hover = useMemo(() => {
    if (!mouse || mouse.x < SM_ML || mouse.x > SM_ML + plotW || mouse.y < SM_MT || mouse.y > SM_MT + plotH) return null;
    const x = x0 + ((mouse.x - SM_ML) / plotW) * (x1 - x0);
    const b = Math.max(0, Math.min(bins - 1, Math.floor((x - marginal.low) / binW)));
    const lo = marginal.low + b * binW;
    return [
      `${parameterLabel(marginal.parameter, false)} ${num(lo)}–${num(lo + binW)}`,
      `${pooled[b]} of ${total} samples (${total ? ((100 * pooled[b]) / total).toFixed(1) : 0} %)`,
      marginal.counts.map((chain, i) => `c${i + 1} ${chain[b]}`).join(" · "),
    ];
  }, [mouse, plotW, plotH, x0, x1, bins, binW, marginal, pooled, total]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const xOf = (x: number) => SM_ML + ((x - x0) / (x1 - x0)) * plotW;
    const yOf = (y: number) => SM_MT + plotH - ((y - y0) / (y1 - y0)) * plotH;

    ctx.save();
    ctx.beginPath();
    ctx.rect(SM_ML, SM_MT, plotW, plotH);
    ctx.clip();
    ctx.fillStyle = palette.seriesSoft;
    pooled.forEach((count, b) => {
      const l = xOf(marginal.low + b * binW);
      const r = xOf(marginal.low + (b + 1) * binW);
      ctx.fillRect(l + 0.5, yOf(count), Math.max(1, r - l - 1), yOf(0) - yOf(count));
    });
    marginal.counts.forEach((chain, c) => {
      ctx.strokeStyle = palette.chains[c % palette.chains.length];
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      chain.forEach((count, b) => {
        const l = xOf(marginal.low + b * binW);
        const r = xOf(marginal.low + (b + 1) * binW);
        const y = yOf(count * n);
        if (b === 0) ctx.moveTo(l, y);
        else ctx.lineTo(l, y);
        ctx.lineTo(r, y);
      });
      ctx.stroke();
    });
    // The level a flat posterior (the prior's) would fill.
    ctx.strokeStyle = palette.limit;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(SM_ML, yOf(total / Math.max(bins, 1)));
    ctx.lineTo(SM_ML + plotW, yOf(total / Math.max(bins, 1)));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    ctx.strokeStyle = axes.axis;
    ctx.strokeRect(SM_ML + 0.5, SM_MT + 0.5, plotW - 1, plotH - 1);
    ctx.fillStyle = axes.title;
    ctx.font = canvasFont(12);
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(parameterLabel(marginal.parameter), SM_ML, 2);
    ctx.fillStyle = axes.tick;
    ctx.font = canvasFont(10);
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(num(x0), SM_ML, SM_MT + plotH + 3);
    ctx.textAlign = "right";
    ctx.fillText(num(x1), SM_ML + plotW, SM_MT + plotH + 3);
    ctx.textBaseline = "middle";
    ctx.fillText(total ? `${((100 * y1) / total).toFixed(0)}%` : "", SM_ML - 3, SM_MT + 4);
    ctx.fillText("0", SM_ML - 3, SM_MT + plotH);
  }, [marginal, pooled, total, bins, binW, n, width, height, plotW, plotH, x0, x1, y0, y1, palette, axes]);

  return (
    <div style={{ position: "relative", width, height }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", cursor: zoom.cursorAt(mouse) }}
        {...handlers}
        onMouseDown={zoom.onMouseDown}
        onDoubleClick={zoom.onDoubleClick}
      />
      <ZoomSelection box={zoom.selection} />
      <ZoomReset zoomed={zoom.zoomed} onReset={zoom.reset} style={{ top: 0, right: SM_MR }} />
      {hover && mouse && (
        <div
          className="viz-tooltip"
          style={mouse.x > width / 2 ? { right: width - mouse.x + 12, top: mouse.y + 12 } : { left: mouse.x + 12, top: mouse.y + 12 }}
        >
          <TooltipLines lines={hover} />
        </div>
      )}
    </div>
  );
}
