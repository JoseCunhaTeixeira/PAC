import { useCallback, useState } from "react";

export interface CanvasHoverPos {
  /** Logical x/y in the same pre-scale, pre-DPR coordinate space the canvas
   * draws in (0..TOTAL_W / 0..TOTAL_H) -- i.e. directly usable with a
   * component's own xOf/yOf inverses. */
  x: number;
  y: number;
}

// Tracks the mouse position over a canvas, converted out of CSS pixels back
// into the component's own logical drawing coordinates (undoing the
// responsive `scale` factor every section canvas applies).
export function useCanvasHover(scale: number) {
  const [pos, setPos] = useState<CanvasHoverPos | null>(null);

  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      setPos({
        x: (e.clientX - rect.left) / scale,
        y: (e.clientY - rect.top) / scale,
      });
    },
    [scale],
  );

  const onMouseLeave = useCallback(() => setPos(null), []);

  return { pos, onMouseMove, onMouseLeave };
}

/** Index of the array element closest to `value` (array need not be sorted). */
export function nearestIndex(arr: number[], value: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - value);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}
