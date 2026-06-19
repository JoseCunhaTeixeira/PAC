export const API = "http://localhost:8000";

// [x, z] in meters; y is always 0 and is hardcoded on the backend, not tracked here.
export type Position = [number, number];

export interface Acquisition {
  folder_path: string;
  files: string[];
  durations: number[];
  sampling_frequencies: number[];
  source_positions: Position[];
  receiver_positions: Position[];
}

export interface Masw {
  length: number;
  step: number;
  distance_min: number;
  distance_max: number;
}

export interface Muting {
  tmin: number;
  tmax: number;
  vmin: number;
  vmax: number;
  taper: number;
}

export interface Dispersion {
  fmin: number;
  fmax: number;
  vmin: number;
  vmax: number;
  nv: number;
}