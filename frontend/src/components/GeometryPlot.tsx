import { useEffect, useMemo, useRef, useState } from "react";
import { API, type Acquisition, type Masw, profileName } from "../api";
import { canvasPalette, useTheme } from "../theme";
import { useContainerWidth } from "./useContainerWidth";
import { AlertCircleIcon } from "./icons";
import {
  drawElevation,
  drawLaneLabels,
  drawLineAxis,
  ELEVATION,
  groundAt,
  LANE,
  LANE_BOTTOM,
  receiverPath,
  reliefOf,
  slopeDegrees,
  starPath,
  symbolSizes,
} from "./lineDraw";
import { useZoom, type PlotRect, type Range } from "./useZoom";
import { ZoomReset, ZoomSelection } from "./ZoomOverlay";
import { TooltipLines } from "./HoverTooltip";
import { vizPalette } from "./viz/palette";

// The line a computing page will process, from above: its shots (stars), its receivers
// (inverted triangles) and the windows the MASW settings make (a cell at each middle), and the
// line's topography under them (straight when it is flat). Hover a shot, a receiver or a window
// for what it is; a window shows its receivers, the shots it stacks and the distances it takes
// them within. A drag zooms along the line, a double-click shows all of it.

interface WindowSummary {
  xmid: number;
  start_index: number;
  end_index: number;
  n_shots: number;
  sources?: number[]; // older backends give none
}

type Hit = { row: "shot" | "receiver" | "window"; index: number };

// The lanes, symbols and axis every line plot shares (lineDraw.ts): Visualization's alike.
const ML = LANE.left;
const MR = LANE.right;
const SHOT_Y = LANE.shotY;
const WINDOW_Y = LANE.windowY;
const WINDOW_H = LANE.windowH;
const ELEVATION_TOP = LANE_BOTTOM + ELEVATION.gap;

const same = (a: number, b: number) => Math.abs(a - b) < 1e-6;

export function GeometryPlot({
  acquisition,
  masw,
  onCount,
  showSources = true,
  unit = "shots",
}: {
  acquisition: Acquisition;
  masw: Masw;
  onCount?: (n: number) => void;
  showSources?: boolean;
  /** What a window stacks: shots, or noise records. */
  unit?: string;
}) {
  // null until the backend has counted them.
  const [counted, setWindows] = useState<WindowSummary[] | null>(null);
  const windows = useMemo(() => counted ?? [], [counted]);
  const [invalid, setInvalid] = useState<string | null>(null);
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null);
  const [xZoom, setXZoom] = useState<Range | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const axes = canvasPalette(theme);
  const palette = vizPalette(theme);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const width = Math.max(360, Math.floor(containerWidth || 900));
  const plotW = width - ML - MR;

  // Read via a ref: a parent often passes a fresh inline callback, which would otherwise
  // retrigger the debounced fetch below on every render.
  const onCountRef = useRef(onCount);
  useEffect(() => {
    onCountRef.current = onCount;
  }, [onCount]);

  // A value being typed or wrong (its field says why): the windows drawn stay as they were.
  const settled =
    Object.values(masw).every((v) => typeof v !== "number" || Number.isFinite(v)) &&
    masw.distance_max > masw.distance_min;
  useEffect(() => {
    if (!settled) return;
    const timer = setTimeout(() => {
      fetch(`${API}/windows`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile: profileName(acquisition), masw }),
      })
        .then(async (res) => {
          const body = await res.json().catch(() => null);
          if (!res.ok) throw new Error(typeof body?.detail === "string" ? body.detail : "");
          return body as WindowSummary[];
        })
        .then((data) => {
          setWindows(data);
          setInvalid(null);
          onCountRef.current?.(data.length);
        })
        .catch((err) => {
          setWindows([]);
          setInvalid(err instanceof Error ? err.message : String(err));
          onCountRef.current?.(0);
        });
    }, 250);
    return () => clearTimeout(timer);
  }, [acquisition, masw, settled]);

  const receiverPoints = useMemo(
    () => acquisition.receiver_positions.map(([x, z]): [number, number] => [x, z]),
    [acquisition],
  );
  const receivers = useMemo(() => receiverPoints.map(([x]) => x), [receiverPoints]);
  const shots = useMemo(
    () =>
      showSources
        ? acquisition.source_positions.map(([x, z], i) => ({ x, z, name: acquisition.files[i] ?? `shot ${i + 1}` }))
        : [],
    [acquisition, showSources],
  );
  // The elevation lane; the hovers say elevations and slopes when the line is not flat.
  const relief = useMemo(
    () => reliefOf(receiverPoints, shots.map((shot): [number, number] => [shot.x, shot.z])),
    [receiverPoints, shots],
  );
  const lanesBottom = relief ? ELEVATION_TOP + ELEVATION.h : LANE_BOTTOM;
  const axisTop = lanesBottom + (relief ? ELEVATION.after : LANE.axisGap);
  const sloped = relief !== null && !relief.flat;
  const height = axisTop + LANE.axisH;
  const extent = useMemo((): Range => {
    const xs = [...receivers, ...shots.map((shot) => shot.x)];
    if (xs.length === 0) return [0, 1];
    const lo = Math.min(...xs);
    const hi = Math.max(...xs);
    const pad = (hi - lo) * 0.02 || 1;
    return [lo - pad, hi + pad];
  }, [receivers, shots]);
  const full = useMemo(() => ({ x: extent, y: [0, 1] as Range }), [extent]);
  const plots: PlotRect[] = [{ left: ML, top: 0, width: plotW, height: axisTop, xAxis: LANE.axisH }];
  const zoom = useZoom({
    extent: full,
    plots,
    width,
    height,
    link: {
      zoom: xZoom ? { x: xZoom, y: full.y } : null,
      setZoom: (view) => setXZoom(view && (Math.abs(view.x[0] - extent[0]) > 1e-9 || Math.abs(view.x[1] - extent[1]) > 1e-9) ? view.x : null),
    },
  });
  const [x0, x1] = zoom.view.x;
  const X = (x: number) => ML + ((x - x0) / (x1 - x0)) * plotW;
  // Symbols sized to their spacing on screen, as in Visualization.
  const { starR, triangleR } = symbolSizes(shots.map((shot) => shot.x), receivers, [x0, x1], plotW);

  // What is under the pointer: a shot, a receiver or a window, by lane; under the windows, the
  // nearest window too.
  function hitAt(px: number, py: number): Hit | null {
    if (px < ML - 6 || px > ML + plotW + 6) return null;
    const nearest = (xs: number[], tolerance: number) => {
      let best = -1;
      let distance = tolerance;
      xs.forEach((x, i) => {
        const d = Math.abs(X(x) - px);
        if (d <= distance) {
          best = i;
          distance = d;
        }
      });
      return best;
    };
    if (showSources && py < (SHOT_Y + LANE.receiverY) / 2) {
      const i = nearest(shots.map((shot) => shot.x), 8);
      return i >= 0 ? { row: "shot", index: i } : null;
    }
    if (py < WINDOW_Y - 4) {
      const i = nearest(receivers, 7);
      return i >= 0 ? { row: "receiver", index: i } : null;
    }
    if (py <= lanesBottom) {
      const i = nearest(windows.map((w) => w.xmid), 40);
      return i >= 0 ? { row: "window", index: i } : null;
    }
    return null;
  }

  const hit = mouse ? hitAt(mouse.x, mouse.y) : null;
  const picked = hit?.row === "window" ? windows[hit.index] : undefined;

  const tooltip = useMemo((): string[] | null => {
    if (!hit) return null;
    if (hit.row === "shot") {
      const shot = shots[hit.index];
      const stacking = windows.filter((w) => (w.sources ?? []).some((x) => same(x, shot.x))).length;
      return [
        `Shot ${shot.name}`,
        `at ${shot.x.toFixed(2)} m`,
        ...(sloped ? [`elevation ${shot.z.toFixed(2)} m`] : []),
        ...(windows.some((w) => w.sources) ? [`stacked by ${stacking} of the ${windows.length} windows`] : []),
      ];
    }
    if (hit.row === "receiver") {
      const using = windows.filter((w) => hit.index >= w.start_index && hit.index <= w.end_index).length;
      return [
        `Receiver ${hit.index + 1}`,
        `at ${receivers[hit.index].toFixed(2)} m`,
        ...(sloped ? [`elevation ${receiverPoints[hit.index][1].toFixed(2)} m`] : []),
        `in ${using} of the ${windows.length} windows`,
      ];
    }
    const w = windows[hit.index];
    const spacing = receivers.length > 1 ? Math.abs(receivers[1] - receivers[0]) : 0;
    // The ground at the window's middle, and its slope under the window: the line best through
    // its receivers.
    const slope = sloped ? slopeDegrees(receiverPoints.slice(w.start_index, w.end_index + 1)) : null;
    return [
      `xmid ${w.xmid.toFixed(2)} m`,
      `receivers ${w.start_index + 1}–${w.end_index + 1}, ${((w.end_index - w.start_index) * spacing).toFixed(2)} m`,
      `${w.n_shots} ${unit} stacked`,
      ...(relief && sloped ? [`elevation ${groundAt(relief, w.xmid).toFixed(2)} m`] : []),
      ...(slope !== null ? [`slope ${slope < 10 ? slope.toFixed(1) : Math.round(slope)}°`] : []),
    ];
  }, [hit, shots, receivers, receiverPoints, relief, sloped, windows, unit]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = globalThis.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const px = (x: number) => ML + ((x - x0) / (x1 - x0)) * plotW;
    drawLaneLabels(ctx, axes.tick, showSources);
    if (relief) drawElevation(ctx, relief, ELEVATION_TOP, [x0, x1], plotW, { tick: axes.tick, ground: palette.muted });

    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, 0, plotW, height);
    ctx.clip();

    // The window under the pointer: a band down every lane over its receivers, and the
    // distances it takes its shots within on the shots' lane.
    if (picked) {
      if (showSources) {
        ctx.fillStyle = palette.reach;
        for (const side of [-1, 1]) {
          const a = px(picked.xmid + side * masw.distance_min);
          const b = px(picked.xmid + side * masw.distance_max);
          ctx.fillRect(Math.min(a, b), SHOT_Y - 11, Math.abs(b - a), 22);
        }
      }
      const first = receivers[picked.start_index];
      const last = receivers[picked.end_index];
      if (first !== undefined && last !== undefined) {
        ctx.fillStyle = palette.selected;
        ctx.fillRect(px(first) - 4, SHOT_Y - 12, px(last) - px(first) + 8, lanesBottom - SHOT_Y + 12);
      }
    }

    // Shots: those the window under the pointer stacks blue, the others grey.
    if (showSources) {
      const stacked = new Set((picked?.sources ?? []).map((x) => x.toFixed(3)));
      shots.forEach((shot, i) => {
        const on = !picked || !picked.sources || stacked.has(shot.x.toFixed(3));
        const hovered = hit?.row === "shot" && hit.index === i;
        ctx.fillStyle = on ? palette.series : palette.faint;
        starPath(ctx, px(shot.x), SHOT_Y, hovered ? starR * 1.35 : starR);
        ctx.fill();
      });
    }

    // Receivers: those of the window under the pointer blue; the one under it bigger and
    // darker (blue is a window's).
    receivers.forEach((x, i) => {
      const inWindow = picked !== undefined && i >= picked.start_index && i <= picked.end_index;
      const hovered = hit?.row === "receiver" && hit.index === i;
      ctx.fillStyle = inWindow ? palette.series : hovered ? palette.ink : palette.muted;
      receiverPath(ctx, px(x), triangleR, hovered ? triangleR * 1.35 : inWindow ? triangleR * 1.25 : triangleR);
      ctx.fill();
    });

    // Windows: a cell at each middle, the one under the pointer outlined.
    const gap = windows.length > 1 ? Math.min(...windows.slice(1).map((w, i) => w.xmid - windows[i].xmid)) : x1 - x0;
    const cellW = Math.max(2, Math.min(26, (gap / (x1 - x0)) * plotW * 0.82));
    windows.forEach((w) => {
      const x = px(w.xmid);
      const on = picked !== undefined && same(w.xmid, picked.xmid);
      ctx.fillStyle = on ? palette.series : palette.seriesSoft;
      ctx.fillRect(x - cellW / 2, WINDOW_Y, cellW, WINDOW_H);
      if (on) {
        ctx.strokeStyle = palette.selectedEdge;
        ctx.lineWidth = 2;
        ctx.strokeRect(x - cellW / 2 - 2, WINDOW_Y - 2, cellW + 4, WINDOW_H + 4);
      }
    });

    // The hovered element: a guide down the lanes.
    if (hit) {
      const x = px(hit.row === "shot" ? shots[hit.index].x : hit.row === "receiver" ? receivers[hit.index] : windows[hit.index].xmid);
      ctx.strokeStyle = palette.hover;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 4);
      ctx.lineTo(x, lanesBottom);
      ctx.stroke();
    }
    ctx.restore();

    drawLineAxis(ctx, [x0, x1], plotW, axes, axisTop);
  }, [
    width, height, plotW, x0, x1, windows, receivers, shots, picked, hit, masw, showSources, axes,
    palette, relief, lanesBottom, axisTop, starR, triangleR,
  ]);

  function logical(e: React.MouseEvent<HTMLCanvasElement>) {
    const box = e.currentTarget.getBoundingClientRect();
    return { x: ((e.clientX - box.left) / box.width) * width, y: ((e.clientY - box.top) / box.height) * height };
  }

  return (
    <div className="geometry">
      <div ref={containerRef} style={{ position: "relative", width: "100%" }}>
        <canvas
          ref={canvasRef}
          style={{ display: "block", cursor: zoom.cursorAt(mouse) }}
          onMouseMove={(e) => setMouse(logical(e))}
          onMouseLeave={() => setMouse(null)}
          onMouseDown={zoom.onMouseDown}
          onDoubleClick={zoom.onDoubleClick}
        />
        <ZoomSelection box={zoom.selection} />
        <ZoomReset zoomed={zoom.zoomed} onReset={zoom.reset} style={{ top: 0, right: MR }} />
        {tooltip && mouse && (
          <div
            className="viz-tooltip"
            style={mouse.x > width * 0.65 ? { right: width - mouse.x + 14, top: mouse.y + 14 } : { left: mouse.x + 14, top: mouse.y + 14 }}
          >
            <TooltipLines lines={tooltip} />
          </div>
        )}
      </div>
      <div className="viz-line-legend geometry-legend">
        {showSources && (
          <span>
            <i className="shot">★</i> {shots.length} shots
          </span>
        )}
        <span>
          <i className="receiver">▼</i> {receivers.length} receivers
        </span>
        {invalid !== null ? (
          <span className="field-error" data-tip={`No window${invalid ? `\n${invalid}` : ""}`}>
            <AlertCircleIcon size={13} />
            No window
          </span>
        ) : (
          <span>
            <i className="window" /> {counted === null ? "counting windows…" : `${windows.length} windows`}
          </span>
        )}
      </div>
    </div>
  );
}
