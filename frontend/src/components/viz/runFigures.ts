import { API } from "../../api";
import { useJson } from "./useJson";

/** A figure a run or a window saved, and where the backend serves it. */
export interface SavedFigure {
  name: string;
  url: string;
}

/** Where in a run its figures are: its root, one of its windows or one of its records. */
export interface FigurePlace {
  xmid?: number;
  record?: string;
}

function query(place: FigurePlace): string {
  if (place.xmid !== undefined) return `?xmid=${place.xmid}`;
  if (place.record !== undefined) return `?record=${encodeURIComponent(place.record)}`;
  return "";
}

/** The figures a run saved at `place` (none asked for with null), by name: PAC's jobs and the
 * assistant draw them, Visualization only shows them. */
export function useRunFigures(folder: string, place: FigurePlace | null = {}): string[] {
  const url = place ? `${API}/quality/run_figures/${encodeURIComponent(folder)}${query(place)}` : null;
  return useJson<string[]>(url).data ?? [];
}

/** Those of the run's `names` starting with `prefix` (one card's; "" for all), with their
 * URLs. */
export function runFigures(folder: string, names: string[], prefix = "", place: FigurePlace = {}): SavedFigure[] {
  return names
    .filter((name) => name.startsWith(prefix))
    .map((name) => ({ name, url: `${API}/quality/run_figure/${encodeURIComponent(folder)}/${name}${query(place)}` }));
}
