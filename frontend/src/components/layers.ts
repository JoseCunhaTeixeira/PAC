// The seismic inversion's layered model, top down, and its edits: a layer above the half-space
// copied below itself, added, removed or moved. The half-space stays last, with no thickness.

export interface VsLayer {
  vs_min: number;
  vs_max: number;
  vs_perturb_std: number;
  vs_fixed?: number | null; // fixed at this value: not sampled
}

export interface ThicknessLayer {
  thickness_min: number;
  thickness_max: number;
  thickness_perturb_std: number;
  thickness_fixed?: number | null; // fixed at this value: not sampled
}

/** The layers, top down: `vs` ends with the half-space; `thickness` has one fewer, none for it. */
export interface Layers {
  vs: VsLayer[];
  thickness: ThicknessLayer[];
}

/** A copy of layer `i` (above the half-space), just below it. */
export function copied(layers: Layers, i: number): Layers {
  const copy = <T>(list: T[]) => [
    ...list.slice(0, i + 1),
    list[i],
    ...list.slice(i + 1),
  ];
  return { vs: copy(layers.vs), thickness: copy(layers.thickness) };
}

/** A layer added just above the half-space: a copy of the one there, or `fallback` with none. */
export function added(
  layers: Layers,
  fallback: { vs: VsLayer; thickness: ThicknessLayer },
): Layers {
  const last = layers.thickness.length - 1;
  if (last >= 0) return copied(layers, last);
  return {
    vs: [fallback.vs, ...layers.vs.slice(-1)],
    thickness: [fallback.thickness],
  };
}

/** Layer `i` (above the half-space) removed. */
export function removed(layers: Layers, i: number): Layers {
  const without = <T>(list: T[]) => list.filter((_, k) => k !== i);
  return { vs: without(layers.vs), thickness: without(layers.thickness) };
}

/** Layer `from` moved to place `to`, both above the half-space. */
export function moved(layers: Layers, from: number, to: number): Layers {
  const move = <T>(list: T[]) => {
    const next = [...list];
    const [one] = next.splice(from, 1);
    next.splice(to, 0, one);
    return next;
  };
  return { vs: move(layers.vs), thickness: move(layers.thickness) };
}
