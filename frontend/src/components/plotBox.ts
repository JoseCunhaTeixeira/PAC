import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { Tool } from "./useZoom";

// The plots of a box (a card) share its tools, on its head's right (kit's BoxTools): what a
// drag does, for all of them at once, and one reset for those zoomed. A plot outside any box
// (the line's profile, where a click selects) moves with a drag.

/** What a drag does in a box: a plot's tools, or the picking's lasso. */
export type DragTool = Tool | "lasso";

export interface PlotBox {
  tool: DragTool;
  setTool: (tool: DragTool) => void;
  /** Adds a zoomed plot's way back to its full view; returns its removal. */
  add: (reset: () => void) => () => void;
  /** Whether a plot of the box is zoomed, and their way back to their full views. */
  zoomed: boolean;
  reset: () => void;
}

export const PlotBoxContext = createContext<PlotBox | null>(null);

/** The box a plot is in, null outside any. */
export function usePlotBox(): PlotBox | null {
  return useContext(PlotBoxContext);
}

/** A box's state: its tool, the hand at first, or the page's (`held`, changed by `onTool`),
 * and its zoomed plots. */
export function usePlotBoxState(held?: DragTool, onTool?: (tool: DragTool) => void): PlotBox {
  const [own, setOwn] = useState<DragTool>("pan");
  const [resets, setResets] = useState<ReadonlySet<() => void>>(() => new Set());
  const add = useCallback((reset: () => void) => {
    setResets((was) => new Set(was).add(reset));
    return () =>
      setResets((was) => {
        const next = new Set(was);
        next.delete(reset);
        return next;
      });
  }, []);
  const tool = held ?? own;
  const setTool = onTool ?? setOwn;
  return useMemo(
    () => ({ tool, setTool, add, zoomed: resets.size > 0, reset: () => resets.forEach((one) => one()) }),
    [tool, setTool, add, resets],
  );
}
