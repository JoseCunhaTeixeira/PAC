import { useMemo } from "react";
import { API } from "../api";
import { useJson } from "./viz/useJson";
import type { RailTone } from "./PositionRail";
import type { Cell, Overview, RunCard, StageKey } from "./viz/types";

// Each window's state at a stage, as Visualization shows it (passed, flagged, rejected; none: no
// curve, or not inverted), for the position rails of the picking and inversion pages.

/** A position's key: its middle, to the millimetre. */
export const xmidKey = (x: number) => x.toFixed(3);

/** The stage's cells by position, and its legends' words (the line's, its parts'); `loading`
 * until they come (empty for a run without them). */
export function useStageStates(folder: string, stage: StageKey, version: unknown = 0) {
  const overview = useJson<Overview>(
    folder ? `${API}/quality/${stage}/overview/${encodeURIComponent(folder)}` : null,
    version,
  );
  const states = useMemo(
    () =>
      new Map(
        (overview.data?.cells ?? []).flatMap((cell) => (cell.x === null ? [] : [[xmidKey(cell.x), cell] as const])),
      ),
    [overview.data],
  );
  return {
    states,
    loading: overview.loading,
    legend: overview.data?.legend ?? {},
    parts: overview.data?.parts ?? [],
  };
}

// An inversion's states on its rail, in Visualization's words when its legend lacks them.
export const INVERTED_TONES: [RailTone, string][] = [
  ["pass", "passed, or within the agent's limits"],
  ["note", "beyond the agent's limits"],
  ["warn", "flagged"],
  ["fail", "rejected"],
];
export const OTHER_TONES: [RailTone, string][] = [
  ["none", "not inverted"],
  ["off", "not picked"],
];

/** A rail's legend: the states present on it (`tones`), each in the words of Visualization's
 * legend (`said`, the overview's) where it has them, else in `fixed`'s. */
export function railItems(
  fixed: [RailTone, string][],
  tones: Set<RailTone>,
  said: Partial<Record<string, string>> = {},
): [RailTone, string][] {
  return fixed.filter(([tone]) => tones.has(tone)).map(([tone, word]) => [tone, said[tone] ?? word]);
}

const NO_RECEIVERS: number[] = [];

/** The line's receivers (its run card's), so that a rail's cells are their spacing wide whatever
 * the windows' step; none without a card, `loading` until it comes. */
export function useReceivers(folder: string) {
  const card = useJson<RunCard>(folder ? `${API}/quality/run/${encodeURIComponent(folder)}` : null);
  return { receivers: card.data?.receivers ?? NO_RECEIVERS, loading: card.loading };
}

/** A cell's verdict, or null when it has none (not picked, not inverted). */
export function judged(cell: Cell | undefined): "pass" | "note" | "warn" | "fail" | null {
  return cell && cell.status !== "none" ? cell.status : null;
}
