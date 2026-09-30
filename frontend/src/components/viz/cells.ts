import type { Cell } from "./types";

// Moving along a stage's units: the one before and after the selected, every unit of the line.

/** The cells before and after `key` along the line. */
export function neighbours(cells: Cell[], key: string | null): { before: Cell | null; after: Cell | null } {
  // Every position along the line, whatever its state; records without a shot (a passive line)
  // in their order.
  const placed = cells.some((cell) => cell.x !== null)
    ? cells.filter((cell) => cell.x !== null).sort((a, b) => (a.x ?? 0) - (b.x ?? 0))
    : cells;
  const index = placed.findIndex((cell) => cell.key === key);
  if (index < 0) return { before: null, after: placed[0] ?? null };
  return { before: placed[index - 1] ?? null, after: placed[index + 1] ?? null };
}

/** The cell a click on a line's plot at `position` selects: the nearest holding a result (a
 * smoothed section's columns fall between windows), the nearest of all when none does. */
export function nearestCell(cells: Cell[], position: number): Cell | null {
  const placed = cells.filter((cell): cell is Cell & { x: number } => cell.x !== null);
  const done = placed.filter((cell) => cell.status !== "none");
  let best: (Cell & { x: number }) | null = null;
  for (const cell of done.length ? done : placed) {
    if (best === null || Math.abs(cell.x - position) < Math.abs(best.x - position)) best = cell;
  }
  return best;
}
