// What Visualization reads of a run (src/masw/io/quality): the runs, a run's card and line, each
// stage's units along the line, and the selected unit's card.

export type Status = "pass" | "warn" | "fail" | "none";
export type Mark = "pass" | "warn" | "fail" | "info";
export type Origin = "rule" | "given" | "default" | "pac";
export type Verdict = "pass" | "retry" | "reject";
export type StageKey = "records" | "dispersion" | "inversion" | "petro";
export type ModelName = "best" | "smooth_best" | "median" | "smooth_median" | "ensemble";

export interface RunEntry {
  folder: string;
  run_id: string | null;
  started_at: string | null;
  mode: string | null;
  by: "assistant" | "pac" | null;
  windows: number;
  /** Receivers per window; null: not said (the older layout). */
  window_length: number | null;
}

export interface ProfileRuns {
  profile: string;
  records: boolean;
  runs: RunEntry[];
}

export interface Setting {
  key: string;
  label: string;
  value: string; // what to read first: "6 m"
  detail: string; // what goes with it: "5 receivers"
  why: string;
  origin: Origin;
}

export interface LengthTrial {
  length: number;
  metres: number;
  xmids: number[];
  passed: number;
  windows: number;
  wavelengths_m: [number, number] | null;
  flags: string[];
  uncertainty: number | null; // the passed curves' median velocity uncertainty
  compared: boolean;
}

export interface StageCount {
  key: StageKey;
  done: number;
  total: number;
}

export interface LineWindow {
  key: string;
  xmid: number;
  first: number;
  last: number;
}

export interface RunCard {
  folder: string;
  profile: string;
  run_id: string | null;
  mode: string | null;
  by: "assistant" | "pac" | null;
  started_at: string | null;
  finished_at: string | null;
  records: boolean;
  settings: Setting[];
  trials: LengthTrial[];
  stages: StageCount[];
  receivers: number[];
  sources: Record<string, number>;
  windows: LineWindow[];
  // The shots a window stacks: from, to this far from its middle (to null: any distance).
  reach: [number, number | null] | null;
}

export type Use = "used" | "part" | "excluded" | "failed" | "inside" | "near" | "far" | "traces";

export interface Shot {
  name: string;
  x: number;
  use: Use;
  why: string;
  receivers: number | null;
}

export interface Sentence {
  mark: Mark;
  text: string;
  detail?: string; // the whole of it, on hover, when the text is its short form
}

export interface WindowSources {
  key: string;
  xmid: number;
  first: number;
  last: number;
  receivers: number;
  passive: boolean;
  stacked: number;
  shots: Shot[];
  sentences: Sentence[];
}

/** A check said apart from the others of its unit (a window's image, its curve): its state, or
 * by hand (the user's, which no check judges). */
export type PartState = Status | "hand";

export interface Cell {
  key: string;
  x: number | null;
  status: Status;
  hover: string[];
  value: number | null;
  total: number | null;
  /** Its checks said apart, top down, in the order of the overview's parts; none: `status`. */
  parts?: PartState[];
  /** The modes picked in it (a window, the dispersion stage): M0, M1... */
  modes?: string[];
}

/** What one of the cells' parts says: what it checks, and each of its states present. */
export interface PartLegend {
  title: string;
  legend: Partial<Record<PartState, string>>;
}

export interface Track {
  label: string;
  short: string;
  kind: "value" | "depth";
  limit: number | null;
  bound: "min" | "max" | null;
}

export interface Overview {
  paco: boolean;
  summary: string;
  legend: Partial<Record<Status, string>>;
  cells: Cell[];
  track: Track | null;
  settings: Setting[];
  /** The cells' parts, when they say their checks apart. */
  parts?: PartLegend[];
}

export interface Metric {
  name: string;
  value: number | null;
  threshold: number | null;
  bound: "min" | "max" | null;
  passed: boolean;
  unit: string;
}

export interface GateView {
  gate: string;
  verdict: Verdict | null;
  metrics: Metric[];
  /** The curve was picked by hand: the curve's check passes it, the later ones leave it out. */
  by_hand?: boolean;
}

export interface AttemptSummary {
  attempt: number;
  stage: string;
  triggered_by: string;
  status: string;
  error: string | null;
  started_at: string;
  parameters: Record<string, unknown>;
  notes: string[];
  verdicts: Record<string, Verdict>;
  flags: string[];
}

export interface Card {
  key: string;
  status: Status;
  title: string;
  verdict: Sentence | null;
  sentences: Sentence[];
  gates: GateView[];
  attempts: AttemptSummary[];
  /** Its checks said apart, as its cell's parts: for its badges. */
  parts?: { label: string; state: PartState }[];
}

export interface RecordCard extends Card {
  x: number | null;
  windows: string[];
  excluded_traces: number[];
}

export interface CurveStats {
  label: string;
  n_points: number;
  band_hz: [number, number];
  wavelength_m: [number, number];
  uncertainty: number | null;
}

export interface DispersionCard extends Card {
  picked_by: "auto" | "hand" | null;
  curves: CurveStats[];
  band_hz: [number, number] | null;
  wavelength_limits_m: [number | null, number | null];
}

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

export interface InversionParameters {
  layering?: "free" | "fixed"; // runs saved before: the layers given
  free?: {
    vs_min: number | null;
    vs_max: number | null;
    depth_min?: number | null;
    depth_max: number | null;
    max_layers: number;
  };
  max_vs_drop?: number;
  n_layers: number;
  vs_layers: VsLayer[];
  thickness_layers: ThicknessLayer[];
  n_iterations: number;
  n_burnin_iterations: number;
  n_chains: number;
}

export interface BandFit {
  wavelength_m: [number, number];
  n_points: number;
  misfit: number | null;
  residual: number | null;
}

export interface ModelFit {
  model: string;
  misfit: number | null;
  n_missing: number;
  bands: BandFit[];
  lowest_missing_hz: number | null;
}

export interface BoundShare {
  parameter: string;
  bound: "min" | "max";
  value: number;
  share: number;
}

export interface VsProfile {
  model: ModelName;
  tops: number[];
  vs: number[];
  // The kept models' 10th and 90th percentiles at each depth (none without them), and their
  // relative uncertainty U = (P90 - P10) / (2 P50) (%).
  spread_depths: number[];
  spread_low: number[];
  spread_high: number[];
  uncertainty: number[];
  // %, per `interface_dz` m from the surface down: the share of the kept models placing a layer
  // boundary there.
  interfaces: number[];
  interface_dz: number;
  bottom: number;
  informed: number | null;
  deepest_top: number;
}

export interface FitCurve {
  label: string;
  observed_fs: number[];
  observed_vs: number[];
  observed_err: number[];
  predicted_fs: number[];
  predicted_vs: number[];
  /** The kept models' curves at the picked frequencies: their 10th and 90th percentiles. Empty for an
   * inversion saved before 2026-09-29, or a petrophysical one. */
  spread_fs?: number[];
  spread_low?: number[];
  spread_high?: number[];
}

export interface Convergence {
  parameter: string;
  prior: [number, number] | null;
  median: number | null;
  low: number | null; // 10 % of the samples below
  high: number | null; // 10 % above
  rhat: number | null;
  ess: number | null;
  autocorrelation: number | null;
  step: number | null;
  fixed?: number | null; // the value, fixed: not sampled
  step_unit?: string; // "%": relative to the value (the layers chosen by the data)
}

export interface InversionAttempt extends AttemptSummary {
  n_layers: number | null;
  depth_m: number | null;
}

export type FigureName = "marginals" | "density_curves" | "dispersion_image" | "chains";

export interface InversionCard extends Card {
  attempts: InversionAttempt[];
  inverted: boolean;
  parameters: InversionParameters | null;
  tuning: [number, number][];
  step_factor: number | null;
  acceptance: number[];
  /** When the data chose the layers: each move's acceptance (%) and relative step (%), the
   * chains' medians, and the exchanges between tempered copies (%). */
  moves?: Record<string, number>;
  move_steps?: Record<string, number>;
  exchanges?: number | null;
  samples_per_chain: number;
  convergence: Convergence[];
  fits: ModelFit[];
  at_bounds: BoundShare[];
  profile: VsProfile | null;
  curve: FitCurve | null;
  figures: FigureName[];
}

export interface ChainTraces {
  parameter: string;
  step: number;
  chains: number[][];
}

export interface Marginal {
  parameter: string;
  low: number;
  high: number;
  counts: number[][];
}

export interface Chains {
  traces: ChainTraces[];
  marginals: Marginal[];
}

export interface SoilColumn {
  soils: string[];
  thicknesses_m: number[];
  ns: number[];
  water_table_m: number | null;
}

export interface PetroCard extends Card {
  model: string | null;
  column: SoilColumn | null;
  curve: FitCurve | null;
  fit: ModelFit | null;
  gaps: string[];
}
