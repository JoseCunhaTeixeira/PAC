import type { Metric, Origin, RunEntry, StageKey, Status } from "./types";

// How Visualization says things: the gates' metrics (a name a geophysicist reads, and the value
// in the unit it is judged in), numbers, dates, and where a setting comes from.

const METRIC_LABELS: Record<string, string> = {
  // G1, a record's signal
  dead_traces: "Dead traces",
  clipped_traces: "Clipped traces",
  nan_traces: "Traces holding NaN",
  rms_outliers: "Traces too weak or strong for their offset",
  snr_db: "Median SNR",
  usable_band_hz: "Usable band width",
  lateral_coherence: "Neighbouring traces' coherence",
  trigger_error_s: "Trigger error",
  trigger_shift_s: "Trigger error", // its name before 2026-09-29
  trigger_scatter_s: "First breaks' scatter",
  energy_removed: "Energy the mute removed",
  pulse_s: "Shot pulse",
  spectral_outliers: "Traces off their neighbours' spectra",
  spectral_receivers: "Receivers off their neighbours' spectra",
  off_decay_receivers: "Receivers too weak or strong for their offset",
  // G2, an image
  coherent_columns: "Coherent columns",
  virtual_shot_snr_db: "Virtual shot's SNR",
  ridge_at_vmin: "Columns peaking at vmin",
  ridge_at_vmax: "Columns peaking at vmax",
  band_at_fmin: "Coherent band reaches fmin",
  band_at_fmax: "Coherent band reaches fmax",
  competing_ridges: "Columns with a second ridge",
  aliased_ridges: "Second ridges below the aliasing limit",
  band_share_of_usable: "Coherent band / usable band",
  // a passive window's segments, by its fk selection
  fk_segments: "Segments the fk selection judged",
  fk_kept: "Segments it kept",
  fk_flipped: "Segments it flipped",
  // G3, a pick
  sharpness: "Sharpness",
  prominence: "Prominence",
  on_data: "Points on the brightest value",
  constant_wavelength: "Points at constant wavelength",
  n_points: "Points picked",
  aliased_points: "Points under λmin",
  beyond_reach_points: "Points over λmax",
  curve_points: "Points of the resampled curve",
  wavelength_ratio: "Longest / shortest wavelength",
  max_jump: "Largest step between points",
  air_wave_share: "Points in the air wave's band",
  trend: "Normal dispersion trend",
  uncertainty: "Median uncertainty / velocity",
  near_offset: "Nearest shot's offset",
  // G4, the line's curves
  neighbour_misfit: "Misfit to the neighbours",
  misfit: "Misfit to the neighbours", // its name before 2026-09-29
  sides_compared: "Sides compared",
  curves: "Curves",
  without_curve: "Windows without a curve",
  depth_spread: "Spread of the depths reached",
  inverse_curves: "Curves with an inverse trend",
  // G5, a seismic model
  misfit_short: "Misfit, short wavelengths",
  misfit_middle: "Misfit, middle wavelengths",
  misfit_long: "Misfit, long wavelengths",
  residual_short: "Residual, short wavelengths",
  residual_middle: "Residual, middle wavelengths",
  residual_long: "Residual, long wavelengths",
  misfit_layered: "Misfit of the layered median",
  rhat: "Max R-hat",
  acceptance: "Median acceptance",
  ess: "Min effective samples",
  autocorrelation: "Max lag-1 autocorrelation",
  samples_per_chain: "Samples a chain",
  at_bound: "Share at a prior's bound",
  depth_informed: "Depth informed",
  useful_depth: "Depth informed", // its name before 2026-09-29
  depth_informed_spread: "Spread of the depths informed",
  // Before 2026-09-29, G6 compared the models down to half their curve's longest wavelength.
  useful_depth_spread: "Spread of the depths of investigation",
  contrast: "Least contrast between layers",
  // G7 and G8, a petrophysical model
  water_table: "Water table",
  water_table_jump: "Water table off the neighbours",
  models: "Models",
  without_model: "Windows without a model",
  water_table_min: "Shallowest water table",
  water_table_max: "Deepest water table",
};

// The inversion's models, as the figures name them.
/** How the depth informed is defined, as its hovers say it (sigpipe's useful_depth). */
export const DEPTH_INFORMED_TIP =
  "Depth informed\nFrom the surface down, to where the kept models' Vs uncertainty " +
  "U = (P90 − P10) / (2 P50) goes over 25 %\nNot at an interface they only place at depths a " +
  "little apart: each model's Vs there is its own just above or below it\nBelow it, veiled: " +
  "the data no longer pin Vs";

export const MODEL_LABELS: Record<string, string> = {
  ensemble: "median of the ensemble",
  median: "median, layered",
  smooth_median: "median, smooth",
  best: "best, layered",
  smooth_best: "best, smooth",
};

// The things a gate's measures describe, a menu each: a window's signal is its stacked
// correlations (the virtual shot's); "neighbours" and "line", what each line check compares.
const OBJECT_TITLES: Record<string, string> = {
  signal: "Signal",
  spectrum: "Spectrum",
  selection: "fk selection",
  image: "Image",
  curve: "Curve",
  fit: "Fit to the curve",
  chains: "Chains",
  model: "Model",
  soil: "Soil column",
};
const GATE_OBJECTS: Record<string, Record<string, string>> = {
  G1: { line: "The line's receivers" },
  G2: { signal: "Stacked correlations", spectrum: "Their spectrum" },
  G4: { neighbours: "Curve against its neighbours", line: "The line's curves" },
  G6: { neighbours: "Model against its neighbours", line: "The line's models" },
  G8: { neighbours: "Soil column against its neighbours", line: "The line's soil columns" },
};
// What each gate measures, for one that measured nothing.
const GATE_OBJECT: Record<string, string> = {
  G1: "signal",
  G2: "image",
  G3: "curve",
  G4: "neighbours",
  G5: "model",
  G6: "neighbours",
  G7: "soil",
  G8: "neighbours",
};

/** The title of a gate's menu of the measures of one thing it measures. */
export function objectTitle(gate: string, of: string): string {
  const thing = of || GATE_OBJECT[gate] || "";
  return GATE_OBJECTS[gate]?.[thing] ?? OBJECT_TITLES[thing] ?? capitalized(thing || gate);
}

export function metricLabel(name: string): string {
  return METRIC_LABELS[name] ?? name.replaceAll("_", " ");
}

// Shares shown as percentages; a boolean metric (1 or 0) as yes or no.
const SHARES = new Set([
  "coherent_columns",
  "ridge_at_vmin",
  "ridge_at_vmax",
  "competing_ridges",
  "aliased_ridges",
  "band_share_of_usable",
  "fk_kept",
  "on_data",
  "constant_wavelength",
  "aliased_points",
  "beyond_reach_points",
  "air_wave_share",
  "uncertainty",
  "at_bound",
  "depth_spread",
  "misfit",
  "neighbour_misfit",
  "depth_informed_spread",
  "useful_depth_spread",
  "energy_removed",
]);
const BOOLEANS = new Set(["band_at_fmin", "band_at_fmax"]);

/** A number with about 3 significant digits, grouped by thousands. */
export function num(v: number | null | undefined, digits = 3): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString("en-US");
  if (a === 0) return "0";
  const decimals = Math.max(0, Math.min(4, digits - 1 - Math.floor(Math.log10(a))));
  // Without trailing zeros: 0.5, not 0.500.
  const text = v.toFixed(decimals);
  return text.includes(".") ? text.replace(/\.?0+$/, "") : text;
}

export function pct(v: number | null | undefined, digits = 0): string {
  return v == null ? "—" : `${(100 * v).toFixed(digits)} %`;
}

export function range(r: [number, number] | null | undefined, unit: string, digits = 3): string {
  return r ? `${num(r[0], digits)}–${num(r[1], digits)} ${unit}` : "—";
}

/** A metric's value (or threshold) as it is judged, with its unit. */
export function metricValue(metric: Metric, value: number | null = metric.value): string {
  if (value == null) return "—";
  if (BOOLEANS.has(metric.name)) return value ? "yes" : "no";
  // A share in whole percents; a small one to two significant digits, never 0 when some are
  // (the fk selection's 1 segment of 2,736: 0.04 %).
  if (SHARES.has(metric.name) && metric.unit === "")
    return value > 0 && value < 0.1 ? `${num(100 * value, 2)} %` : pct(value);
  if (metric.unit === "s") return `${num(value * 1000)} ms`;
  if (metric.unit === "%") return `${num(value)} %`;
  return metric.unit ? `${num(value)} ${metric.unit}` : num(value);
}

/** The metric's limit: "≤ 1.1", "≥ 400"; empty when it is reported only. */
export function metricLimit(metric: Metric): string {
  if (metric.threshold == null || metric.bound == null) return "";
  if (BOOLEANS.has(metric.name)) return "no";
  return `${metric.bound === "max" ? "≤" : "≥"} ${metricValue(metric, metric.threshold)}`;
}

/** An inversion parameter's name as the figures say it: vs1 as Vs1, thick1 as H1. */
export function parameterLabel(name: string, unit = true): string {
  // Vs at a depth ("vs@2.5m"), what the chains are judged on.
  if (name.startsWith("vs@"))
    return `Vs at ${name.slice(3, -1)} m${unit ? " [m/s]" : ""}`;
  if (name.startsWith("vs")) return `Vs${name.slice(2)}${unit ? " [m/s]" : ""}`;
  if (name.startsWith("thick")) return `H${name.slice(5)}${unit ? " [m]" : ""}`;
  if (name === "layers") return "Layers";
  if (name === "noise") return "Noise factor";
  return name;
}

/** Why an attempt ran, in words. */
export function triggerLabel(triggeredBy: string): string {
  if (triggeredBy === "initial") return "first run";
  if (triggeredBy === "backtrack") return "an earlier stage ran again";
  return triggeredBy.replace(":", " · ").replaceAll("_", " ");
}

/** A stage's parameters an attempt changed, as "key value" pairs, nested keys dotted. */
export function changes(parameters: Record<string, unknown>, prefix = ""): string[] {
  const found: string[] = [];
  for (const [key, value] of Object.entries(parameters)) {
    const name = prefix ? `${prefix}.${key}` : key;
    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      found.push(...changes(value as Record<string, unknown>, name));
    } else if (Array.isArray(value)) {
      found.push(`${name} [${value.length}]`);
    } else {
      found.push(`${name} ${typeof value === "number" ? num(value, 4) : String(value)}`);
    }
  }
  return found;
}

/** Where a setting comes from, in two words. */
export const ORIGIN_LABELS: Record<Origin, string> = {
  rule: "from the data",
  given: "given",
  default: "default",
  pac: "set by hand",
};

/** What a unit's state means for what comes after it, stage by stage: `full` on its card's
 * badge, `short` on the line's hover and legend. */
export const STATE_MEANINGS: Record<StageKey, Record<Status, { full: string; short: string }>> = {
  records: {
    pass: { full: "The windows stack it.", short: "stacked by its windows" },
    warn: { full: "The windows stack it; worth a look.", short: "stacked, with a warning" },
    fail: { full: "Left out of every window.", short: "left out of every window" },
    none: { full: "Not checked.", short: "" },
  },
  dispersion: {
    pass: { full: "Its curve goes on to the inversion.", short: "its curve goes on to the inversion" },
    warn: { full: "Its curve goes on to the inversion; worth a look.", short: "its curve goes on to the inversion, flagged" },
    fail: { full: "Its curve is not inverted.", short: "its curve is not inverted" },
    none: { full: "No curve: nothing to invert here.", short: "" },
  },
  inversion: {
    pass: { full: "Its model can be trusted.", short: "its model is trusted" },
    warn: { full: "Its model is kept; read it with care.", short: "its model is kept, flagged" },
    fail: { full: "Its model is not to be trusted.", short: "its model is not trusted" },
    none: { full: "Not inverted.", short: "" },
  },
  petro: {
    pass: { full: "Its soil column can be trusted.", short: "its soil column is trusted" },
    warn: { full: "Its soil column is kept; read it with care.", short: "its soil column is kept, flagged" },
    fail: { full: "Its soil column is not to be trusted.", short: "its soil column is not trusted" },
    none: { full: "Not inverted.", short: "" },
  },
};

export const STATUS_LABELS: Record<Status, string> = {
  pass: "pass",
  warn: "flagged",
  fail: "rejected",
  none: "—",
};

/** A run's start, as its selector and header show it: "26 Sep 2026, 15:13". */
export function runDate(iso: string | null): string {
  if (!iso) return "";
  const date = new Date(iso);
  return date.toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** A run in a selector: when, by whom, how many windows. */
export function runLabel(run: RunEntry): string {
  if (run.run_id === null) return `${run.folder} (older layout)`;
  const by = run.by === "assistant" ? "assistant" : "by hand";
  const length = run.window_length ? ` of ${run.window_length} receivers` : "";
  return `${runDate(run.started_at)} · ${by} · ${run.windows} windows${length}`;
}

/** A window's folder as its position: "xmid_103.50" as 103.5. */
export function xmidOf(key: string): number {
  return Number(key.replace(/^xmid_/, ""));
}

export function capitalized(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
