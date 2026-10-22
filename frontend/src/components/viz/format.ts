import { thousands } from "../numbers";
import type { Maker, Metric, Origin, RunEntry, StageKey, Status } from "./types";

// How Visualization says things: the checks' measures (a name a geophysicist reads, what it is
// on its hover, and the value in the unit it is judged in), numbers, dates, and where a setting
// comes from.

// Each measure's name, then what it is, a line each.
const METRICS: Record<string, [string, string]> = {
  // A record's signal
  dead_traces: ["Dead traces", "Traces with almost no signal"],
  clipped_traces: ["Clipped traces", "Traces cut at the recorder's largest value"],
  nan_traces: ["Traces with NaN samples", "Traces holding samples that are not numbers"],
  rms_outliers: [
    "Traces too weak or too loud",
    "Amplitude off the decay with distance from the shot\nReported only: the receivers are judged over the whole line",
  ],
  snr_db: [
    "Median SNR",
    "The surface waves against the noise after them\nThe median over the traces the waves reach\nIn the band the dispersion images use",
  ],
  usable_band_hz: ["Usable band width", "The frequencies where the surface waves stand clearly over the noise"],
  lateral_coherence: [
    "Coherence of neighbouring traces",
    "How alike neighbouring traces are where the surface waves arrive\nTheir median correlation: 1 when identical",
  ],
  trigger_error_s: [
    "Trigger error",
    "The shot time read from the first breaks, against the record's time zero\nJudged on a muted record only: its mute starts at time zero",
  ],
  trigger_shift_s: [
    "Trigger error",
    "The shot time read from the first breaks, against the record's time zero",
  ], // its name in older logs
  trigger_scatter_s: [
    "Scatter of the first breaks",
    "How far the first breaks stray from a straight line\nLarge when the delay differs from trace to trace",
  ],
  energy_removed: ["Energy the mute removed", "The share of the record's energy the mute took out"],
  pulse_s: ["Shot pulse length", "How long the shot's energy lasts after its first break\nOn the traces nearest the shot"],
  spectral_outliers: [
    "Traces with an unusual spectrum",
    "A spectrum unlike their neighbours': a gain, a response or a dead band of their own\nReported only: the receivers are judged over the whole line",
  ],
  fast_arrivals: [
    "Energy faster than the surface waves",
    "The share of the surface-wave energy arriving faster than the gathers' cone\nBody waves or refractions",
  ],
  // The line's receivers
  off_decay_receivers: [
    "Receivers too weak or too loud",
    "Amplitude off the decay with distance in most records reaching them\nLeft out of every window",
  ],
  spectral_receivers: [
    "Receivers with an unusual spectrum",
    "A spectrum unlike their neighbours' in most records reaching them\nKept, to look at",
  ],
  // A dispersion image
  coherent_columns: ["Coherent frequencies", "The share of the image's frequencies whose peak stands over the background"],
  virtual_shot_snr_db: [
    "Median SNR",
    "The stacked correlations' surface waves against the noise after them\nMeasured as a record's, from their virtual source",
  ],
  ridge_at_vmin: [
    "Peaks at the lowest velocity",
    "The share of the coherent frequencies peaking at the image's lowest velocity\nIts velocities may start too high",
  ],
  ridge_at_vmax: [
    "Peaks at the highest velocity",
    "The share of the coherent frequencies peaking at the image's highest velocity\nIts velocities may stop too low",
  ],
  band_at_fmin: [
    "Coherent band at the lowest frequency",
    "The coherent band reaches the image's lowest frequency\nIt may go on below it",
  ],
  band_at_fmax: [
    "Coherent band at the highest frequency",
    "The coherent band reaches the image's highest frequency\nIt may go on above it",
  ],
  competing_ridges: [
    "Frequencies with a second ridge",
    "The share of the coherent frequencies with a second peak of similar strength\nA higher mode or another wave",
  ],
  aliased_ridges: [
    "Second ridges in the aliasing zone",
    "The share of the second ridges at wavelengths under twice the receiver spacing\nThere a ridge may be an alias",
  ],
  background: ["Background level", "The image away from its ridge, against the ridge's peak\n0 when clean, 1 as strong as the ridge"],
  stripes: ["Background stripes", "How much that level jumps from a frequency to the next\n0 when even"],
  band_share_of_usable: [
    "Coherent band / usable band",
    "The coherent band's width over the records' usable band, within the image's frequencies",
  ],
  // A passive window's segments, by its FK selection
  fk_segments: ["Segments judged", "The window's noise segments the FK selection judged"],
  fk_kept: ["Segments kept", "The share of the segments the FK selection kept"],
  fk_flipped: ["Segments flipped", "Segments whose waves travel the other way, turned before stacking"],
  // A curve
  sharpness: ["Sharpness", "How narrow the ridge is, against a plane wave's through the window\n1 when as narrow"],
  prominence: ["Prominence", "How much the ridge stands over the rest of its frequency, against a plane wave's"],
  on_data: ["Points on the ridge's top", "The share of the picked points at their frequency's brightest value, within 10 %"],
  constant_wavelength: [
    "Points at constant wavelength",
    "The share of the points where velocity grows like frequency\nA wavelength set by the window, not by the ground",
  ],
  n_points: ["Points picked", "The points of the pick kept on its ridge"],
  aliased_points: [
    "Points in the aliasing zone",
    "The share of the curve's points at wavelengths under twice the receiver spacing",
  ],
  beyond_reach_points: [
    "Points beyond the window's reach",
    "The share of the curve's points at wavelengths over three window lengths",
  ],
  curve_points: ["Points of the curve", "The pick resampled by wavelength: what the inversion fits"],
  wavelength_ratio: ["Longest / shortest wavelength", "How wide a range of wavelengths the curve spans"],
  max_jump: ["Largest step", "The largest velocity step from a point to the next, relative\nA jump to another mode"],
  air_wave_share: ["Points at the air wave's speed", "The share of the points near 340 m/s: the air wave, not the ground"],
  trend: ["Normal dispersion", "The rank correlation of velocity with wavelength\nAbove 0 when velocity grows with wavelength"],
  uncertainty: ["Median uncertainty", "The points' velocity uncertainty over their velocity, the median"],
  near_offset: [
    "Nearest shot",
    "The distance from the window's receivers to its nearest shot\nNearer than half the longest wavelength, the long wavelengths may read slow (near field)",
  ],
  // A curve, a model, a soil column against its neighbours
  neighbour_misfit: [
    "Difference from the neighbours",
    "Against the neighbours on each side, the median relative difference\nThe worse side",
  ],
  misfit: ["Difference from the neighbours", "Against the neighbours on each side, the median relative difference\nThe worse side"],
  sides_compared: ["Sides compared", "The sides whose neighbours agree with each other, and so could be compared"],
  // The line's curves
  curves: ["Curves passed", "The windows whose curve passed the curve check"],
  without_curve: ["Windows without a curve", "The windows whose curve was rejected or never picked"],
  depth_spread: [
    "Spread of λmax",
    "How much the curves' longest wavelength varies along the line\nThe median absolute deviation over the median",
  ],
  inverse_curves: ["Curves with an inverse trend", "The curves whose velocity falls as wavelength grows"],
  // A seismic model
  misfit_short: ["Misfit, short wavelengths", "The model's curve against the picked points at the shortest wavelengths\nIn uncertainties: under 1 within them"],
  misfit_middle: ["Misfit, middle wavelengths", "The model's curve against the picked points at the middle wavelengths\nIn uncertainties: under 1 within them"],
  misfit_long: ["Misfit, long wavelengths", "The model's curve against the picked points at the longest wavelengths\nIn uncertainties: under 1 within them"],
  residual_short: ["Residual, short wavelengths", "(modelled − picked) / modelled, the median over those points"],
  residual_middle: ["Residual, middle wavelengths", "(modelled − picked) / modelled, the median over those points"],
  residual_long: ["Residual, long wavelengths", "(modelled − picked) / modelled, the median over those points"],
  misfit_layered: ["Misfit of the layered median", "The layered median's curve against every picked point"],
  rhat: ["Chains' agreement (R-hat)", "How well the chains agree on Vs at the depths watched: 1 when they agree\nThe worst over those depths"],
  acceptance: ["Moves accepted", "The share of the moves the chains accepted, the median"],
  ess: ["Effective samples", "The independent samples the chains hold, the fewest over the depths watched"],
  autocorrelation: [
    "Lag-1 autocorrelation",
    "How much each sample repeats the one before it, the most over the depths watched",
  ],
  samples_per_chain: ["Samples per chain", "The models each chain keeps after its burn-in"],
  at_bound: [
    "Samples at a bound",
    "The share of a parameter's samples at the edge of its allowed range, the most over the parameters",
  ],
  depth_informed: ["Depth informed", ""],
  useful_depth: ["Depth informed", ""], // its name in older logs
  depth_informed_spread: [
    "Spread of the depths informed",
    "How much the depth informed varies along the line\nThe median absolute deviation over the median",
  ],
  // In older logs, the models were compared down to half their curve's longest wavelength.
  useful_depth_spread: ["Spread of the depths of investigation", "How much the depth of investigation varies along the line"],
  contrast: [
    "Least contrast between layers",
    "The smallest Vs difference between adjacent layers of the layered median, over their mean",
  ],
  // The line's models
  models: ["Models passed", "The windows whose model passed the model check"],
  without_model: ["Windows without a model", "The windows not inverted, or whose model was rejected"],
  shallow_models: [
    "Models informed too shallow",
    "Informed above the depths the line's models are compared at\nNot compared with their neighbours",
  ],
  // A soil column, and the line's
  water_table: ["Water table", "The depth to water in the soil column"],
  water_table_jump: ["Water table against the neighbours", "The difference from the neighbours' water tables, the worse side"],
  water_table_min: ["Shallowest water table", "Over the line's soil columns"],
  water_table_max: ["Deepest water table", "Over the line's soil columns"],
};
// ...where a measure's name says something else at another check.
const GATE_METRICS: Record<string, Record<string, [string, string]>> = {
  G8: {
    models: ["Soil columns passed", "The windows whose soil column passed its check"],
    without_model: ["Windows without a soil column", "The windows with no soil column, or whose soil column was rejected"],
  },
};

// The inversion's models, as the figures name them.
/** What the depth informed is, as its hovers say it (sigpipe's useful_depth). */
export const DEPTH_INFORMED_TIP =
  "From the surface down to where the models' Vs uncertainty goes over 25 %\n" +
  "The jumps at the layers' interfaces aside\n" +
  "Deeper, veiled: the data no longer constrain Vs";

export const MODEL_LABELS: Record<string, string> = {
  ensemble: "median of the ensemble",
  median: "median, layered",
  smooth_median: "median, smooth",
  best: "best, layered",
  smooth_best: "best, smooth",
};

// The checks, by what they judge: a unit's, and the line's (its measures "of" the line).
const CHECK_NAMES: Record<string, string> = {
  G1: "Signal check",
  G2: "Dispersion image check",
  G3: "Curve check",
  G4: "Curve against its neighbours",
  G5: "Model check",
  G6: "Model against its neighbours",
  G7: "Soil column check",
  G8: "Soil column against its neighbours",
};
const LINE_CHECK_NAMES: Record<string, string> = {
  G1: "The line's receivers",
  G4: "The line's curves",
  G6: "The line's models",
  G8: "The line's soil columns",
};

/** A check's name, as its menu of measures is titled: the line's own (`line`) or a unit's. */
export function checkName(gate: string, line = false): string {
  return (line ? LINE_CHECK_NAMES[gate] : undefined) ?? CHECK_NAMES[gate] ?? gate;
}

// The things a check measures, titled within its menu when it measures several: a window's
// signal is its stacked correlations (the virtual shot's).
const OBJECT_TITLES: Record<string, string> = {
  signal: "Signal",
  spectrum: "Spectrum",
  selection: "FK selection",
  image: "Image",
  curve: "Curve",
  fit: "Fit to the curve",
  chains: "Chains",
  model: "Model",
  soil: "Soil column",
};
const GATE_OBJECTS: Record<string, Record<string, string>> = {
  G2: { signal: "Stacked correlations", spectrum: "Their spectrum" },
};

/** The title of the measures of one thing a check measures. */
export function objectTitle(gate: string, of: string): string {
  return GATE_OBJECTS[gate]?.[of] ?? OBJECT_TITLES[of] ?? capitalized(of);
}

function metricWords(name: string, gate?: string): [string, string] {
  return (gate ? GATE_METRICS[gate]?.[name] : undefined) ?? METRICS[name] ?? [name.replaceAll("_", " "), ""];
}

export function metricLabel(name: string, gate?: string): string {
  return metricWords(name, gate)[0];
}

/** What a measure is, on its name's hover, a line a fact: its name is said beside it. */
export function metricTip(name: string, gate?: string): string {
  if (name === "depth_informed" || name === "useful_depth") return DEPTH_INFORMED_TIP;
  return metricWords(name, gate)[1];
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

/** A number with about 3 significant digits, its thousands as SI writes them. */
export function num(v: number | null | undefined, digits = 3): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1000) return thousands(String(Math.round(v)));
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

// The measures whose value none says all of it: the depth informed, down to the model's bottom.
const WHOLE = new Set(["depth_informed", "useful_depth"]);

/** Whether a metric has a value to judge: measured, or all of it (WHOLE). */
export function measured(metric: Metric): boolean {
  return metric.value != null || WHOLE.has(metric.name);
}

/** A metric's value (or threshold) as it is judged, with its unit. */
export function metricValue(metric: Metric, value: number | null = metric.value): string {
  if (value == null) return WHOLE.has(metric.name) ? "the whole model" : "not measured";
  if (BOOLEANS.has(metric.name)) return value ? "yes" : "no";
  // A share in whole percents; a small one to two significant digits, never 0 when some are
  // (the fk selection's 1 segment of 2,736: 0.04 %).
  if (SHARES.has(metric.name) && metric.unit === "")
    return value > 0 && value < 0.1 ? `${num(100 * value, 2)} %` : pct(value);
  if (metric.unit === "s") return `${num(value * 1000)} ms`;
  if (metric.unit === "%") return `${num(value)} %`;
  return metric.unit ? `${num(value)} ${metric.unit}` : num(value);
}

// The measures judged by their size whichever their sign: a trigger early or late.
const SIGNED = new Set(["trigger_error_s", "trigger_shift_s"]);

/** The metric's limit in words: "at most 1.1", "at least 400 m/s", "at most 0", "within
 * ±10 ms"; "no limit" when it is reported only. */
export function metricLimit(metric: Metric): string {
  if (metric.threshold == null || metric.bound == null) return "no limit";
  if (BOOLEANS.has(metric.name)) return "not allowed";
  const limit = metricValue(metric, metric.threshold);
  if (SIGNED.has(metric.name)) return `within ±${limit}`;
  return `${metric.bound === "max" ? "at most" : "at least"} ${limit}`;
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

/** Where a setting comes from, in two words. */
// Who set a setting, as the card tags it: the assistant whichever its way.
export const ORIGIN_LABELS: Record<Origin, string> = {
  you: "You",
  workflow: "Agent",
  agent: "Agent",
  checks: "Agent",
  default: "Default",
};

// Who made a run, in the assistant's settings' words.
export const MAKER_LABELS: Record<Maker, string> = {
  manual: "Manual",
  workflow: "Workflow",
  autonomous: "Autonomous",
};

/** What a unit's state led to, stage by stage: `full` on its card's badge, `short` on the
 * line's hover; none where it led to nothing more than its badge says. */
export const STATE_MEANINGS: Record<StageKey, Record<Status, { full: string; short: string }>> = {
  records: {
    pass: { full: "Used by its windows", short: "used by its windows" },
    warn: { full: "Used by its windows", short: "used by its windows" },
    fail: { full: "Left out of every window", short: "left out of every window" },
    none: { full: "", short: "" },
  },
  dispersion: {
    pass: { full: "Its curve went on to the inversion", short: "its curve went on to the inversion" },
    warn: { full: "Its curve went on to the inversion", short: "its curve went on to the inversion" },
    fail: { full: "Its curve was not inverted", short: "its curve was not inverted" },
    none: { full: "", short: "" },
  },
  inversion: {
    pass: { full: "", short: "" },
    warn: { full: "", short: "" },
    fail: { full: "", short: "" },
    none: { full: "", short: "" },
  },
  petro: {
    pass: { full: "", short: "" },
    warn: { full: "", short: "" },
    fail: { full: "", short: "" },
    none: { full: "", short: "" },
  },
};

export const STATUS_LABELS: Record<Status, string> = {
  pass: "passed",
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
  const length = run.window_length ? ` of ${run.window_length} receivers` : "";
  const by = run.made_by ? MAKER_LABELS[run.made_by] : null;
  return [runDate(run.started_at), by, `${run.windows} windows${length}`].filter(Boolean).join(" · ");
}

/** A window's folder as its position: "xmid_103.50" as 103.5. */
export function xmidOf(key: string): number {
  return Number(key.replace(/^xmid_/, ""));
}

export function capitalized(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
