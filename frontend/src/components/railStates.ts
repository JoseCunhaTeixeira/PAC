import { useMemo } from "react";
import { API } from "../api";
import { useJson } from "./viz/useJson";
import type { Cell, Overview, StageKey } from "./viz/types";

// Each window's state at a stage, as Visualization shows it (passed, flagged, rejected; none: no
// curve, or not inverted), for the position rails of the picking and inversion pages.

/** A position's key: its middle, to the millimetre. */
export const xmidKey = (x: number) => x.toFixed(3);

/** The stage's cells by position; `loading` until they come (empty for a run without them). */
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
  return { states, loading: overview.loading };
}

/** A cell's verdict, or null when it has none (not picked, not inverted). */
export function judged(cell: Cell | undefined): "pass" | "warn" | "fail" | null {
  return cell && cell.status !== "none" ? cell.status : null;
}
