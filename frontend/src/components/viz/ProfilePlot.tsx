import { useEffect, useMemo, useRef, useState } from "react";
import { canvasPalette, useTheme } from "../../theme";
import { useContainerWidth } from "../useContainerWidth";
import { CLICK_PX, type Range } from "../useZoom";
import {
  cellWidth,
  drawCell,
  drawLaneLabels,
  drawLineAxis,
  LANE,
  receiverPath,
  starPath,
  symbolSizes,
  type CellTone,
} from "../lineDraw";
import { TipLines } from "../HoverTooltip";
import type { Tip } from "../tips";
import { num } from "./format";
import { lineExtent } from "./line";
import { vizPalette } from "./palette";
import type { Cell, RunCard, Shot, Status, WindowSources } from "./types";

// The line from above: every shot (a star) and receiver (an inverted triangle) along it, every
// window as a cell coloured by the stage's state, and the selected window: its receivers, the
// shots its image stacks (blue), those it leaves out (red, grey) and the reach it stacks them
// within. On the records' stage, the cells are the shots themselves, and the windows that stack
// the selected one are blue. What the pointer is on stands out, as on the computing pages. A
// click selects: no zoom of its own (no tools, no wheel), the records' gathers' along the line.

// The lanes every line plot shares (lineDraw.ts).
const ML = LANE.left;
const MR = LANE.right;
const SHOT_Y = LANE.shotY;
const WINDOW_Y = LANE.windowY;
const WINDOW_H = LANE.windowH;
const AXIS_H = LANE.axisH;

type Mode = "windows" | "records";

interface Hit {
  row: "shot" | "receiver" | "window";
  index: number;
}

export function ProfilePlot({
  card,
  cells,
  implies,
  mode,
  selected,
  sources,
  stacking,
  onSelect,
  onShot,
  onWindow,
  xZoom = null,
}: {
  card: RunCard;
  /** The stage's cells: windows at their middle, or records at their shot. */
  cells: Cell[] | null;
  /** What each state implies, said on hover under the window's own lines. */
  implies?: Partial<Record<Status, string>>;
  mode: Mode;
  /** The selected cell's key: a window's folder, or a record's name. */
  selected: string | null;
  /** The selected window's shots: those it stacks, those it leaves out and why. */
  sources: WindowSources | null;
  /** On the records' stage: the windows that stack the selected record. */
  stacking: string[] | null;
  onSelect: (key: string) => void;
  onShot?: (name: string) => void;
  onWindow?: (key: string) => void;
  /** The zoom along the line of the plots under it (the records' gathers), followed to stay
   * aligned with them; null for the whole line. */
  xZoom?: Range | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const axes = canvasPalette(theme);
  const palette = vizPalette(theme);
  const [containerRef, containerWidth] = useContainerWidth<HTMLDivElement>();
  const width = Math.max(320, Math.floor(containerWidth || 800));
  const plotW = width - ML - MR;
  const bottom = WINDOW_Y + WINDOW_H;
  const axisTop = bottom + LANE.axisGap;
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null);
  const down = useRef<{ x: number; y: number } | null>(null);

  const shots = useMemo(
    () => Object.entries(card.sources).map(([name, x]) => ({ name, x })).sort((a, b) => a.x - b.x),
    [card.sources],
  );
  // A line without shots (passive): no shots' lane, everything drawn that much higher (its
  // heights `lift` above the canvas's), as the computing pages' geometry.
  const lift = shots.length > 0 ? 0 : LANE.receiverY - SHOT_Y;
  const height = axisTop + AXIS_H - lift;
  const full = useMemo(() => lineExtent(card), [card]);
  // The gathers' zoom, within the line; the whole line without one.
  const x0 = xZoom ? Math.max(xZoom[0], full[0]) : full[0];
  const x1 = xZoom && Math.min(xZoom[1], full[1]) > x0 ? Math.min(xZoom[1], full[1]) : full[1];
  const xOf = (x: number) => ML + ((x - x0) / (x1 - x0)) * plotW;

  // The cells by key, and those at a position along the line, in order.
  const placed = useMemo(
    () =>
      (cells ?? [])
        .filter((cell): cell is Cell & { x: number } => cell.x !== null)
        .sort((a, b) => a.x - b.x),
    [cells],
  );
  const shotUse = useMemo(() => {
    const uses = new Map<string, Shot>();
    for (const shot of sources?.shots ?? []) uses.set(shot.name, shot);
    return uses;
  }, [sources]);
  const recordCells = useMemo(() => new Map((cells ?? []).map((cell) => [cell.key, cell])), [cells]);
  const windowCells = useMemo(() => (mode === "windows" ? placed : []), [mode, placed]);
  const selectedWindow = mode === "windows" ? card.windows.find((one) => one.key === selected) : undefined;
  const stacked = useMemo(() => new Set(stacking ?? []), [stacking]);

  // A cell's width: a receiver's spacing on screen, whatever the windows' step.
  const cellW = useMemo(
    () => cellWidth(card.windows.map((window) => window.xmid), card.receivers, x1 - x0, plotW),
    [card.windows, card.receivers, x0, x1, plotW],
  );

  function hitAt(px: number, py: number): Hit | null {
    if (px < ML - 4 || px > ML + plotW + 4) return null;
    const nearest = (xs: number[], tolerance: number) => {
      let best = -1;
      let distance = tolerance;
      xs.forEach((x, i) => {
        const d = Math.abs(xOf(x) - px);
        if (d <= distance) {
          best = i;
          distance = d;
        }
      });
      return best;
    };
    if (py >= SHOT_Y - 12 && py <= SHOT_Y + 10) {
      const i = nearest(shots.map((shot) => shot.x), 7);
      return i >= 0 ? { row: "shot", index: i } : null;
    }
    if (py > SHOT_Y + 10 && py < WINDOW_Y - 2) {
      const i = nearest(card.receivers, 5);
      return i >= 0 ? { row: "receiver", index: i } : null;
    }
    if (py >= WINDOW_Y - 2 && py <= bottom) {
      const i = nearest(card.windows.map((window) => window.xmid), Math.max(cellW / 2 + 2, 6));
      return i >= 0 ? { row: "window", index: i } : null;
    }
    return null;
  }

  const hit = mouse ? hitAt(mouse.x, mouse.y + lift) : null;

  const tooltip = useMemo((): Tip | null => {
    if (!hit) return null;
    // A cell's own lines (the checks'): its name, then a bullet a line.
    const cellTip = (lines: readonly string[]): Tip => ({ title: lines[0], notes: lines.slice(1) });
    if (hit.row === "shot") {
      const shot = shots[hit.index];
      if (mode === "records") {
        const cell = recordCells.get(shot.name);
        return cell ? cellTip(cell.hover) : { title: shot.name, values: `${num(shot.x, 4)} m` };
      }
      const use = shotUse.get(shot.name);
      const notes: string[] = [];
      if (use) notes.push(use.why);
      else if (selectedWindow) notes.push(`${num(Math.abs(shot.x - selectedWindow.xmid), 4)} m from the window's middle`);
      if (onShot) notes.push("click to open its record");
      return { title: `Shot ${shot.name}`, values: `${num(shot.x, 4)} m`, notes };
    }
    if (hit.row === "receiver") {
      const x = card.receivers[hit.index];
      const inside = selectedWindow && x >= selectedWindow.first - 1e-9 && x <= selectedWindow.last + 1e-9;
      return { title: `Receiver ${hit.index + 1}`, values: `${num(x, 4)} m`, notes: inside ? ["in the selected window"] : [] };
    }
    const window = card.windows[hit.index];
    if (mode === "records") {
      return {
        title: `xmid ${num(window.xmid, 4)} m`,
        notes: [
          stacked.has(window.key) ? "stacks the selected record" : "does not stack it",
          ...(onWindow ? ["click to open its dispersion"] : []),
        ],
      };
    }
    const cell = windowCells.find((one) => one.key === window.key);
    if (!cell) return { title: `xmid ${num(window.xmid, 4)} m` };
    const implied = implies?.[cell.status];
    return cellTip(implied ? [...cell.hover, implied] : cell.hover);
  }, [hit, shots, mode, recordCells, shotUse, selectedWindow, card, stacked, windowCells, onShot, onWindow, implies]);

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
    ctx.translate(0, -lift);
    const X = (x: number) => ML + ((x - x0) / (x1 - x0)) * plotW;

    drawLaneLabels(ctx, axes.tick, shots.length > 0);

    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, lift, plotW, height);
    ctx.clip();

    // The selected window: a band down every row over its receivers, and the reach its shots
    // lie within on the shots' row.
    if (selectedWindow) {
      // Any distance (no farthest): no reach to draw.
      if (card.reach && card.reach[1] !== null && shots.length) {
        const [, far] = card.reach;
        ctx.fillStyle = palette.reach;
        const l = X(selectedWindow.xmid - far);
        const r = X(selectedWindow.xmid + far);
        ctx.fillRect(l, SHOT_Y - 11, r - l, 22);
      }
      ctx.fillStyle = palette.selected;
      const l = X(selectedWindow.first) - 4;
      const r = X(selectedWindow.last) + 4;
      ctx.fillRect(l, SHOT_Y - 12, r - l, bottom - SHOT_Y + 12);
    }

    // The window under the pointer, as the computing pages show it: a band over its receivers
    // (the selected one has its own), its receivers blue, its cell outlined.
    const hovered = hit?.row === "window" ? card.windows[hit.index] : undefined;
    if (hovered && hovered !== selectedWindow) {
      ctx.fillStyle = palette.selected;
      const l = X(hovered.first) - 4;
      const r = X(hovered.last) + 4;
      ctx.fillRect(l, SHOT_Y - 12, r - l, bottom - SHOT_Y + 12);
    }

    // Symbols sized to their spacing on screen: a line of many shots or receivers keeps them
    // apart.
    const { starR, triangleR } = symbolSizes(shots.map((shot) => shot.x), card.receivers, [x0, x1], plotW);

    // Shots: stars. How the selected window uses each, by colour: stacked (blue), some of its
    // traces left out (hollow blue), rejected (red), left with too few receivers (amber), out of
    // its reach or inside it (grey). On the records' stage, each record's state.
    shots.forEach((shot, i) => {
      const x = X(shot.x);
      let fill: string | null = palette.status.none;
      let stroke: string | null = null;
      let r = starR;
      if (mode === "records") {
        const cell = recordCells.get(shot.name);
        fill = cell ? palette.status[cell.status] : palette.status.none;
      } else if (sources) {
        const use = shotUse.get(shot.name);
        if (use) {
          if (use.use === "used") fill = palette.use.used;
          else if (use.use === "part") {
            fill = null;
            stroke = palette.use.part;
          } else if (use.use === "excluded" || use.use === "failed" || use.use === "traces") {
            fill = palette.use[use.use];
          } else {
            fill = palette.faint;
            r = starR * 0.8;
          }
        }
      }
      // The shot under the pointer: bigger, and a grey one darker.
      if (hit?.row === "shot" && hit.index === i) {
        r = starR * 1.35;
        if (fill === palette.faint || fill === palette.status.none) fill = palette.muted;
      }
      starPath(ctx, x, SHOT_Y, r);
      if (fill) {
        ctx.fillStyle = fill;
        ctx.fill();
      }
      if (stroke) {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }
      if (mode === "records" && shot.name === selected) {
        ctx.strokeStyle = palette.selectedEdge;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.arc(x, SHOT_Y, starR + 4, 0, 2 * Math.PI);
        ctx.stroke();
      }
    });

    // Receivers: inverted triangles, those of the selected window and of the window under the
    // pointer blue; the one under the pointer bigger and darker (blue is a window's).
    const within = (x: number, window: typeof selectedWindow) =>
      window !== undefined && x >= window.first - 1e-9 && x <= window.last + 1e-9;
    card.receivers.forEach((x, i) => {
      const inWindow = within(x, selectedWindow) || within(x, hovered);
      const pointed = hit?.row === "receiver" && hit.index === i;
      receiverPath(ctx, X(x), triangleR, pointed ? triangleR * 1.35 : inWindow ? triangleR * 1.25 : triangleR);
      ctx.fillStyle = inWindow ? palette.series : pointed ? palette.ink : palette.muted;
      ctx.fill();
    });

    // Windows: a cell at each middle, as the rails draw theirs (lineDraw's drawCell); one telling
    // its checks apart (the image's, the curve's), a band each; the one under the pointer
    // stronger and taller, the selected one stronger and edged.
    const cellOf = new Map(windowCells.map((cell) => [cell.key, cell]));
    for (const window of card.windows) {
      let tones: CellTone[] = ["none"];
      if (mode === "windows") {
        const cell = cellOf.get(window.key);
        tones = !cell ? ["none"] : cell.parts?.length ? cell.parts : [cell.status];
      } else if (stacked.has(window.key)) {
        tones = ["series"];
      }
      const chosen = mode === "windows" && window === selectedWindow;
      const strength = chosen ? "active" : window === hovered ? "hover" : "rest";
      // How many modes were picked in it, written in its curve's band when more than one.
      const modes = cellOf.get(window.key)?.modes?.length ?? 0;
      drawCell(ctx, theme, X(window.xmid), cellW, tones, strength, chosen, modes > 1 ? String(modes) : undefined);
    }

    // The selected cell's guide down to the axis.
    const selectedX =
      mode === "windows" ? selectedWindow?.xmid : shots.find((shot) => shot.name === selected)?.x;
    if (selectedX !== undefined) {
      const x = X(selectedX);
      ctx.strokeStyle = palette.selectedEdge;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, mode === "windows" ? WINDOW_Y + WINDOW_H + 2 : SHOT_Y + 10);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // The hovered element.
    if (hit) {
      const x =
        hit.row === "shot"
          ? X(shots[hit.index].x)
          : hit.row === "receiver"
            ? X(card.receivers[hit.index])
            : X(card.windows[hit.index].xmid);
      ctx.strokeStyle = palette.hover;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 4);
      ctx.lineTo(x, bottom);
      ctx.stroke();
    }
    ctx.restore();

    drawLineAxis(ctx, [x0, x1], plotW, axes, axisTop);
  }, [
    width, height, lift, plotW, bottom, axisTop, x0, x1, axes, palette, theme, shots, card, mode, selected, sources,
    shotUse, recordCells, windowCells, selectedWindow, stacked, cellW, hit,
  ]);

  function logical(e: React.MouseEvent<HTMLCanvasElement>) {
    const box = e.currentTarget.getBoundingClientRect();
    return { x: ((e.clientX - box.left) / box.width) * width, y: ((e.clientY - box.top) / box.height) * height };
  }

  function click(e: React.MouseEvent<HTMLCanvasElement>) {
    const at = down.current;
    down.current = null;
    if (!at || Math.hypot(e.clientX - at.x, e.clientY - at.y) >= CLICK_PX) return;
    const { x, y } = logical(e);
    const found = hitAt(x, y + lift);
    if (!found) return;
    if (found.row === "shot") {
      const name = shots[found.index].name;
      if (mode === "records") onSelect(name);
      else onShot?.(name);
    } else if (found.row === "window") {
      const key = card.windows[found.index].key;
      if (mode === "windows") onSelect(key);
      else onWindow?.(key);
    }
  }

  const cursor = hit && hit.row !== "receiver" ? "pointer" : undefined;
  return (
    <div ref={containerRef} style={{ position: "relative", width: "100%" }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", cursor }}
        onMouseMove={(e) => setMouse(logical(e))}
        onMouseLeave={() => setMouse(null)}
        onMouseDown={(e) => {
          down.current = { x: e.clientX, y: e.clientY };
        }}
        onClick={click}
      />
      {tooltip && mouse && (
        <div
          className="viz-tooltip"
          style={
            mouse.x > width * 0.6
              ? { right: width - mouse.x + 14, top: mouse.y + 14 }
              : { left: mouse.x + 14, top: mouse.y + 14 }
          }
        >
          <TipLines tip={tooltip} />
        </div>
      )}
    </div>
  );
}
