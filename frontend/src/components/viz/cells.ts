import type { Cell } from "./types";

// Moving along a stage's units: the one before and after the selected, skipping those without
// a result (a window not inverted) while some hold one.

/** The cells before and after `key` that hold a result (all of them when none does). */
export function neighbours(cells: Cell[], key: string | null): { before: Cell | null; after: Cell | null } {
  // Along the line; records without a shot (a passive line) in their order.
  const placed = cells.some((cell) => cell.x !== null)
    ? cells.filter((cell) => cell.x !== null).sort((a, b) => (a.x ?? 0) - (b.x ?? 0))
    : cells;
  const some = placed.some((cell) => cell.status !== "none");
  const candidates = some ? placed.filter((cell) => cell.status !== "none" || cell.key === key) : placed;
  const index = candidates.findIndex((cell) => cell.key === key);
  if (index < 0) {
    const x = placed.find((cell) => cell.key === key)?.x ?? null;
    if (x === null) return { before: null, after: candidates[0] ?? null };
    const before = [...candidates].reverse().find((cell) => (cell.x ?? 0) < x) ?? null;
    const after = candidates.find((cell) => (cell.x ?? 0) > x) ?? null;
    return { before, after };
  }
  return { before: candidates[index - 1] ?? null, after: candidates[index + 1] ?? null };
}
