export const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// [x, z] in meters; y is always 0 and is hardcoded on the backend, not tracked here.
export type Position = [number, number];

export interface Acquisition {
  folder_path: string;
  files: string[];
  durations: number[];
  sampling_frequencies: number[];
  source_positions: Position[];
  receiver_positions: Position[];
  // Each record's trigger, s after its first sample, from its file; null: not said (absent
  // from an older backend).
  triggers?: (number | null)[];
  kind: string; // "active" or "passive"
  modes: string[]; // the modes it can be processed in
}

// A profile's name: its folder's, in the input folder.
export function profileName(acquisition: Acquisition): string {
  return acquisition.folder_path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
}

export interface Masw {
  length: number;
  step: number;
  distance_min: number;
  distance_max: number;
}

export interface Muting {
  tmin: number | null;
  tmax: number | null;
  vmin: number | null;
  vmax: number | null;
  width: number;
  taper: number;
}

export interface Dispersion {
  fmin: number;
  fmax: number;
  vmin: number;
  vmax: number;
  nv: number;
}
