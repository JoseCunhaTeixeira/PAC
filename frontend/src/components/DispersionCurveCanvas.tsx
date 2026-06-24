import { useEffect, useMemo, useRef } from "react";
import { HoverTooltip } from "./HoverTooltip";
import { CANVAS_FONT, canvasPalette, useTheme } from "../theme";
import { useCanvasHover } from "./useCanvasHover";

export interface CurveSeries {
  label: string;
  color: string;
  observedFs: number[] | null;
  observedVs: number[] | null;
  observedVsErr: number[] | null;
  predictedFs: number[] | null;
  predictedVs: number[] | null;
}

const ML = 48, MR = 8, MT = 8, MB = 36;
const PLOT_W = 150, PLOT_H = 110;
const FONT = "10px sans-serif";
const TOTAL_W = ML + PLOT_W + MR;
const TOTAL_H = MT + PLOT_H + MB;

export function DispersionCurveCanvas({
  series,
  title,
  xLabel,
  yLabel,
}: {
  series: CurveSeries[];
  title: string;
  xLabel?: string;
  yLabel?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const palette = canvasPalette(theme);
  const { pos: hoverPos, onMouseMove, onMouseLeave } = useCanvasHover(1);

  const axisRange = useMemo(() => {
    const allFs = series.flatMap((s) => [...(s.observedFs ?? []), ...(s.predictedFs ?? [])]);
    const allVs = series.flatMap((s) => [
      ...(s.observedVs ?? []),
      ...(s.predictedVs ?? []),
      ...(s.observedVs && s.observedVsErr
        ? s.observedVs.flatMap((v, i) => {
            const err = s.observedVsErr![i];
            return err == null ? [] : [v - err, v + err];
          })
        : []),
    ]);
    if (allFs.length === 0 || allVs.length === 0) return null;

    const fMin = Math.min(...allFs);
    const fMax = Math.max(...allFs);
    const vMin = Math.min(...allVs);
    const vMax = Math.max(...allVs);
    return { fMin, fMax, fSpan: fMax - fMin || 1, vMin, vMax, vSpan: vMax - vMin || 1 };
  }, [series]);

  const hover = useMemo(() => {
    if (!hoverPos || !axisRange) return null;
    if (
      hoverPos.x < ML || hoverPos.x > ML + PLOT_W ||
      hoverPos.y < MT || hoverPos.y > MT + PLOT_H
    ) {
      return null;
    }
    const { fMin, fSpan, vMin, vSpan } = axisRange;
    const freq = fMin + ((hoverPos.x - ML) / PLOT_W) * fSpan;
    const vel = vMin + ((MT + PLOT_H - hoverPos.y) / PLOT_H) * vSpan;
    return {
      px: hoverPos.x,
      py: hoverPos.y,
      lines: [`Frequency: ${freq.toFixed(1)} Hz`, `Velocity: ${vel.toFixed(1)} m/s`],
    };
  }, [hoverPos, axisRange]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(TOTAL_W * dpr);
    canvas.height = Math.round(TOTAL_H * dpr);
    canvas.style.width = TOTAL_W + "px";
    canvas.style.height = TOTAL_H + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, TOTAL_W, TOTAL_H);

    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(ML, MT, PLOT_W, PLOT_H);

    ctx.fillStyle = palette.title;
    ctx.font = CANVAS_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(title, ML + PLOT_W / 2, MT - 2 + 10);

    if (!axisRange) {
      ctx.fillStyle = palette.tick;
      ctx.font = FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("No data", ML + PLOT_W / 2, MT + PLOT_H / 2);
      return;
    }

    const { fMin, fMax, vMin, vMax, fSpan, vSpan } = axisRange;

    const xOf = (f: number) => ML + ((f - fMin) / fSpan) * PLOT_W;
    const yOf = (v: number) => MT + PLOT_H - ((v - vMin) / vSpan) * PLOT_H;

    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, MT, PLOT_W, PLOT_H);
    ctx.clip();

    // Observed curves are drawn in a first pass, predicted in a second, so
    // observed always renders behind every predicted curve -- not just its
    // own series' -- regardless of how many series are plotted.
    for (const s of series) {
      if (!s.observedFs || !s.observedVs) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      s.observedFs.forEach((f, i) => {
        const x = xOf(f), y = yOf(s.observedVs![i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      if (s.observedVsErr) {
        ctx.lineWidth = 1;
        for (let i = 0; i < s.observedFs.length; i++) {
          const err = s.observedVsErr[i];
          if (err == null) continue;
          const v = s.observedVs[i];
          const x = xOf(s.observedFs[i]);
          ctx.beginPath();
          ctx.moveTo(x, yOf(v - err));
          ctx.lineTo(x, yOf(v + err));
          ctx.stroke();
        }
      }
    }

    for (const s of series) {
      if (!s.predictedFs || !s.predictedVs) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      s.predictedFs.forEach((f, i) => {
        const x = xOf(f), y = yOf(s.predictedVs![i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();

    ctx.fillStyle = palette.tick;
    ctx.font = FONT;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(vMin.toFixed(0), ML - 4, MT + PLOT_H);
    ctx.fillText(vMax.toFixed(0), ML - 4, MT);

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(fMin.toFixed(0), ML, MT + PLOT_H + 2);
    ctx.fillText(fMax.toFixed(0), ML + PLOT_W, MT + PLOT_H + 2);

    if (xLabel) {
      ctx.fillStyle = palette.title;
      ctx.font = FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(xLabel, ML + PLOT_W / 2, TOTAL_H - 2);
    }

    if (yLabel) {
      ctx.save();
      ctx.fillStyle = palette.title;
      ctx.font = FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.translate(8, MT + PLOT_H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }
  }, [series, axisRange, title, xLabel, yLabel, theme, palette]);

  return (
    <div style={{ position: "relative", width: TOTAL_W, height: TOTAL_H }}>
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
