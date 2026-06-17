export const API = "http://localhost:8000";

export interface Acquisition {
  folder_path: string;
  files: string[];
  durations: number[];
  sampling_frequencies: number[];
  source_positions: number[];
  receiver_positions: number[];
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