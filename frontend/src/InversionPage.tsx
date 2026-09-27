import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { API } from "./api";
import {
  ArrowRightIcon,
  CrosshairIcon,
  DepthIcon,
  PlusIcon,
} from "./components/icons";
import {
  Callout,
  Card,
  Fields,
  NumberField,
  Page,
  Segmented,
} from "./components/kit";
import { LayerTable } from "./components/LayerTable";
import {
  added,
  type Layers,
  type ThicknessLayer,
  type VsLayer,
} from "./components/layers";
import {
  PositionRail,
  RailLegend,
  type RailCell,
} from "./components/PositionRail";
import { judged, useStageStates, xmidKey } from "./components/railStates";
import { num } from "./components/viz/format";
import { RunPanel } from "./components/RunPanel";
import { runningJob } from "./components/jobs";
import { readStored, useStoredState, writeStored } from "./components/stored";
import { RunSelect } from "./components/RunSelect";

// Seismic inversion: which picked curves to invert, the layers (chosen by the data within
// bounds, or given in a table), the Vs drop allowed and the chains' effort, then the run. Its
// results are read in Visualization.

type Layering = "free" | "fixed";

// When the data choose the layers: each bound left out (null) comes from the curves.
interface FreeLayers {
  vs_min: number | null;
  vs_max: number | null;
  depth_max: number | null;
  max_layers: number;
}

// sigpipe's inversion parameters (GET /inversion/defaults): the form's starting values, and
// those of a layer it adds.
interface InversionParameters {
  layering?: Layering; // older backends: the layers given
  free?: FreeLayers;
  n_layers: number;
  vs_layers: VsLayer[];
  thickness_layers: ThicknessLayer[];
  max_vs_drop?: number;
  n_iterations: number;
  n_burnin_iterations: number;
  n_chains: number;
}

interface InversionDefaults {
  parameters: InversionParameters;
  vs_layer: VsLayer;
  thickness_layer: ThicknessLayer;
  half_space_layer?: VsLayer; // older backends: the layers'
}

// A value, or how to change the previous one.
type Update<T> = T | ((previous: T) => T);

// A run's modes to invert and its positions, as the page keeps them.
interface Selection {
  labels?: string[];
  positions?: number[];
}

// The parameters, as the page keeps them.
interface InversionSettings {
  layering: Layering;
  free: FreeLayers;
  maxVsDrop: number; // %
  nLayers: number;
  vsLayers: VsLayer[];
  thicknessLayers: ThicknessLayer[];
  nIterations: number;
  nBurninIterations: number;
  nChains: number;
}

// Kept under a name of its own since the layers can be chosen by the data: the earlier
// settings' table is taken over, their effort (the sampler's before 2026-09-27) is not.
const SETTINGS_KEY = "pac.inversion.settings.2";
const DROP_TIP =
  "A layer's Vs below the one above, at most\nFarther, a stiff layer over a soft one fits slow picks the ground never gave\n0: Vs only increases with depth";
const EARLIER_KEY = "pac.inversion.settings";
const FREE: FreeLayers = {
  vs_min: null,
  vs_max: null,
  depth_max: null,
  max_layers: 8,
};

// A bound left empty (NaN in the form) is sent as none: the curves set it.
const bound = (value: number | null) =>
  value !== null && Number.isFinite(value) ? value : null;

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export default function InversionPage() {
  // The run, kept when the page is left.
  const [folder, setFolder] = useStoredState("pac.inversion.folder", "");
  // Back on the page while its job runs: its run, so that its run bar shows the job.
  useEffect(() => {
    let cancelled = false;
    runningJob("inversion").then((job) => {
      if (!cancelled && job) setFolder(job.target);
    });
    return () => {
      cancelled = true;
    };
  }, [setFolder]);
  const [xmids, setXmids] = useState<number[]>([]);
  const [labelCounts, setLabelCounts] = useState<Record<string, number>>({});
  const [positionPicks, setPositionPicks] = useState<
    { xmid: number; labels: string[] }[]
  >([]);
  // Each run's modes and positions, kept when the page is left: every mode, to begin with.
  const [selections, setSelections] = useStoredState<Record<string, Selection>>(
    "pac.inversion.selections",
    {},
  );
  const selectedLabels = selections[folder]?.labels ?? Object.keys(labelCounts);
  const selectedPositions: Record<number, boolean> = Object.fromEntries(
    (selections[folder]?.positions ?? []).map((xmid) => [xmid, true]),
  );
  const setSelectedLabels = (update: Update<string[]>) =>
    setSelections((all) => {
      const previous = all[folder]?.labels ?? Object.keys(labelCounts);
      return {
        ...all,
        [folder]: {
          ...all[folder],
          labels: typeof update === "function" ? update(previous) : update,
        },
      };
    });
  const setSelectedPositions = (update: Update<Record<number, boolean>>) =>
    setSelections((all) => {
      const previous = Object.fromEntries(
        (all[folder]?.positions ?? []).map((xmid) => [xmid, true]),
      );
      const next = typeof update === "function" ? update(previous) : update;
      const positions = Object.keys(next)
        .map(Number)
        .filter((xmid) => next[xmid]);
      return { ...all, [folder]: { ...all[folder], positions } };
    });

  // The parameters, kept as one when the page is left; sigpipe's defaults when none were.
  const kept =
    readStored<InversionSettings>(SETTINGS_KEY) ??
    (readStored<InversionSettings>(EARLIER_KEY)
      ? {
          ...readStored<InversionSettings>(EARLIER_KEY)!,
          layering: undefined,
          nIterations: undefined,
          nBurninIterations: undefined,
          nChains: undefined,
        }
      : undefined);
  const [defaults, setDefaults] = useState<InversionDefaults | null>(null);
  const [layering, setLayering] = useState<Layering>(kept?.layering ?? "free");
  const [free, setFree] = useState<FreeLayers>(kept?.free ?? FREE);
  const [maxVsDrop, setMaxVsDrop] = useState(kept?.maxVsDrop ?? 20);
  const [nLayers, setNLayers] = useState(kept?.nLayers ?? 0);
  const [vsLayers, setVsLayers] = useState<VsLayer[]>(kept?.vsLayers ?? []);
  const [thicknessLayers, setThicknessLayers] = useState<ThicknessLayer[]>(
    kept?.thicknessLayers ?? [],
  );
  const [nIterations, setNIterations] = useState(kept?.nIterations ?? 0);
  const [nBurninIterations, setNBurninIterations] = useState(
    kept?.nBurninIterations ?? 0,
  );
  const [nChains, setNChains] = useState(kept?.nChains ?? 0);
  const [nWorkers, setNWorkers] = useStoredState("pac.inversion.workers", 1);
  useEffect(() => {
    if (nLayers > 0 && nIterations > 0) {
      writeStored(SETTINGS_KEY, {
        layering,
        free,
        maxVsDrop,
        nLayers,
        vsLayers,
        thicknessLayers,
        nIterations,
        nBurninIterations,
        nChains,
      });
    }
  }, [
    layering,
    free,
    maxVsDrop,
    nLayers,
    vsLayers,
    thicknessLayers,
    nIterations,
    nBurninIterations,
    nChains,
  ]);
  const [running, setRunning] = useState(false); // the settings frozen while their job runs

  const [error, setError] = useState<string | null>(null);
  const [loadingLabels, setLoadingLabels] = useState(true);
  // Which positions are inverted, and their checks, as Visualization shows them; again after a run.
  const [runs, setRuns] = useState(0);
  const { states, loading: statesLoading } = useStageStates(
    folder,
    "inversion",
    runs,
  );
  const nCpus = navigator.hardwareConcurrency || 1;

  useEffect(() => {
    fetch(`${API}/inversion/defaults`)
      .then((res) => json<InversionDefaults>(res))
      .then((data) => {
        const parameters = data.parameters;
        setDefaults(data);
        if (readStored(SETTINGS_KEY) !== undefined) return; // the page's own, kept
        if (parameters.free) setFree(parameters.free);
        if (parameters.max_vs_drop !== undefined)
          setMaxVsDrop(Math.round(parameters.max_vs_drop * 100));
        setNIterations(parameters.n_iterations);
        setNBurninIterations(parameters.n_burnin_iterations);
        setNChains(parameters.n_chains);
        if (readStored(EARLIER_KEY) !== undefined) return; // the earlier table, kept
        setNLayers(parameters.n_layers);
        setVsLayers(parameters.vs_layers);
        setThicknessLayers(parameters.thickness_layers);
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }, []);

  const refresh = useCallback((folderName: string) => {
    Promise.resolve().then(() => setLoadingLabels(true));
    fetch(`${API}/xmids/${encodeURIComponent(folderName)}`)
      .then((res) => json<number[]>(res))
      .then(setXmids)
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
    fetch(`${API}/dispersion_image_labels/${encodeURIComponent(folderName)}`)
      .then((res) => json<Record<string, number>>(res))
      .then(setLabelCounts)
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      )
      .finally(() => setLoadingLabels(false));
    fetch(
      `${API}/dispersion_picks_by_position/${encodeURIComponent(folderName)}`,
    )
      .then((res) => json<{ xmid: number; labels: string[] }[]>(res))
      .then(setPositionPicks)
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }, []);

  useEffect(() => {
    Promise.resolve().then(() => {
      setXmids([]);
      setLabelCounts({});
      setPositionPicks([]);
      setError(null);
    });
    if (folder) refresh(folder);
  }, [folder, refresh]);

  // The layers, as the table edits them.
  const layers: Layers = { vs: vsLayers, thickness: thicknessLayers };
  function setLayers(next: Layers) {
    setNLayers(next.vs.length);
    setVsLayers(next.vs);
    setThicknessLayers(next.thickness);
  }

  function toggleLabel(label: string) {
    setSelectedLabels((prev) =>
      prev.includes(label)
        ? prev.filter((one) => one !== label)
        : [...prev, label],
    );
  }

  const allLabels = Object.keys(labelCounts);
  const eligible = xmids.filter((xmid) => {
    const picks = positionPicks.find((one) => one.xmid === xmid);
    return (
      !!picks && picks.labels.some((label) => selectedLabels.includes(label))
    );
  });
  const selectedXmids = eligible.filter((xmid) => selectedPositions[xmid]);
  const maxWorkers =
    selectedXmids.length > 0 ? Math.min(nCpus, selectedXmids.length) : nCpus;
  const cells: RailCell[] = xmids.map((xmid) => {
    const picks = positionPicks.find((one) => one.xmid === xmid);
    const can = eligible.includes(xmid);
    const state = states.get(xmidKey(xmid));
    const verdict = judged(state);
    const lines = !can
      ? [`xmid ${num(xmid, 4)} m`, "no curve of the modes chosen"]
      : state && verdict
        ? state.hover
        : [
            `xmid ${num(xmid, 4)} m`,
            picks?.labels.join(", ") ?? "",
            "not inverted",
          ];
    return {
      xmid,
      tone: !can ? "off" : (verdict ?? "none"),
      title: lines.join("\n"),
    };
  });

  const config = {
    folder,
    positions: selectedXmids,
    labels: selectedLabels,
    parameters: {
      layering,
      ...(layering === "free"
        ? {
            free: {
              vs_min: bound(free.vs_min),
              vs_max: bound(free.vs_max),
              depth_max: bound(free.depth_max),
              max_layers: free.max_layers,
            },
          }
        : {
            n_layers: nLayers,
            // A value fixed only when one is.
            vs_layers: vsLayers.map(({ vs_fixed, ...rest }) =>
              vs_fixed == null ? rest : { ...rest, vs_fixed },
            ),
            thickness_layers: thicknessLayers.map(
              ({ thickness_fixed, ...rest }) =>
                thickness_fixed == null ? rest : { ...rest, thickness_fixed },
            ),
          }),
      max_vs_drop: maxVsDrop / 100,
      n_iterations: nIterations,
      n_burnin_iterations: nBurninIterations,
      n_chains: nChains,
    },
    n_workers: nWorkers,
  };

  const missing: string[] = [];
  if (!defaults) missing.push("the defaults (loading)");
  if (selectedLabels.length === 0) missing.push("a mode");
  if (selectedXmids.length === 0) missing.push("a position");
  const fixedCount =
    vsLayers.filter((layer) => layer.vs_fixed != null).length +
    thicknessLayers.filter((layer) => layer.thickness_fixed != null).length;
  if (layering === "fixed" && nLayers > 0 && fixedCount >= 2 * nLayers - 1)
    missing.push("a value left free");

  return (
    <Page
      icon={<DepthIcon size={24} />}
      title="Seismic inversion"
      subtitle="Invert dispersion curves into Vs profiles"
      art="inversion"
      actions={
        <RunSelect folder={folder} onChange={setFolder} disabled={running} />
      }
    >
      {error && <Callout tone="error">{error}</Callout>}
      {folder && !loadingLabels && allLabels.length === 0 && (
        <Callout tone="warn" title="No curve picked in this run">
          <Link to="/dispersion_picking">Pick its curves first.</Link>
        </Callout>
      )}

      {folder && allLabels.length > 0 && (
        <div className="stack">
          <fieldset className="frozen" disabled={running}>
            <Card
              step={1}
              title="Curves to invert"
              hint="The modes to invert, and where along the line."
              aside={
                <RailLegend
                  groups={[
                    {
                      title: "Inverted",
                      items: [
                        ["pass", "passed"],
                        ["warn", "flagged"],
                        ["fail", "rejected"],
                      ],
                    },
                    {
                      items: [
                        ["none", "Not inverted"],
                        ["off", "No curve"],
                      ],
                    },
                  ]}
                />
              }
            >
              <div className="chips" role="group" aria-label="Modes">
                {allLabels.map((label) => (
                  <label
                    key={label}
                    className={`chip${selectedLabels.includes(label) ? " on" : ""}`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedLabels.includes(label)}
                      onChange={() => toggleLabel(label)}
                    />
                    <CrosshairIcon size={14} />
                    {label}
                    <small>{labelCounts[label]} positions</small>
                  </label>
                ))}
              </div>
              <div className="rail-head">
                <span className="muted">
                  <b>{selectedXmids.length}</b> of {eligible.length} positions
                </span>
                <span className="rail-actions">
                  <button
                    type="button"
                    className="ghost"
                    onClick={() =>
                      setSelectedPositions(
                        Object.fromEntries(
                          eligible.map((xmid) => [xmid, true]),
                        ),
                      )
                    }
                  >
                    All
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => setSelectedPositions({})}
                  >
                    None
                  </button>
                </span>
              </div>
              {!statesLoading && (
                <PositionRail
                  cells={cells}
                  isActive={(xmid) =>
                    !!selectedPositions[xmid] && eligible.includes(xmid)
                  }
                  onPaint={(xmid, on) =>
                    setSelectedPositions((prev) => ({ ...prev, [xmid]: on }))
                  }
                />
              )}
            </Card>

            <Card
              step={2}
              title="Layers"
              hint={
                layering === "free"
                  ? "Chosen by the data: how many, how thick, how fast."
                  : "Given: each one's range, or a value fixed."
              }
              aside={
                <span className="layers-aside">
                  {layering === "fixed" && (
                    <span className="stepper">
                      <b>{nLayers}</b> layers
                      <button
                        type="button"
                        className="secondary icon"
                        aria-label="Add a layer"
                        data-tip={
                          "Add a layer\nA copy of the one above the half-space"
                        }
                        disabled={!defaults || vsLayers.length === 0}
                        onClick={() =>
                          defaults &&
                          setLayers(
                            added(layers, {
                              vs: defaults.vs_layer,
                              thickness: defaults.thickness_layer,
                            }),
                          )
                        }
                      >
                        <PlusIcon size={15} />
                      </button>
                    </span>
                  )}
                  <Segmented
                    label="Layers"
                    size="sm"
                    value={layering}
                    onChange={setLayering}
                    options={[
                      {
                        value: "free",
                        label: "By the data",
                        title:
                          "Chosen by the data\nHow many layers, how thick: the inversion samples them\nWithin the bounds below",
                      },
                      {
                        value: "fixed",
                        label: "Fixed",
                        title:
                          "Given\nThe layers of the table, each one's range or value",
                      },
                    ]}
                  />
                </span>
              }
            >
              {layering === "free" ? (
                <Fields>
                  <NumberField
                    label="Vs from"
                    unit="m/s"
                    value={free.vs_min ?? Number.NaN}
                    optional="auto"
                    min={1}
                    step={10}
                    title={
                      "Every layer's least Vs\nEmpty: half the slowest pick"
                    }
                    onChange={(value) => setFree({ ...free, vs_min: value })}
                  />
                  <NumberField
                    label="Vs to"
                    unit="m/s"
                    value={free.vs_max ?? Number.NaN}
                    optional="auto"
                    min={1}
                    step={10}
                    title={
                      "Every layer's greatest Vs\nEmpty: three times the fastest pick"
                    }
                    onChange={(value) => setFree({ ...free, vs_max: value })}
                  />
                  <NumberField
                    label="Deepest interface"
                    unit="m"
                    value={free.depth_max ?? Number.NaN}
                    optional="auto"
                    min={0.1}
                    step={0.5}
                    title={
                      "Where the half-space may start, at the deepest\nEmpty: half the longest picked wavelength"
                    }
                    onChange={(value) => setFree({ ...free, depth_max: value })}
                  />
                  <NumberField
                    label="Layers at most"
                    value={free.max_layers}
                    min={1}
                    max={20}
                    step={1}
                    title={"The half-space included\nThe data choose how many"}
                    onChange={(value) =>
                      setFree({
                        ...free,
                        max_layers: Math.max(
                          1,
                          Math.min(20, Math.round(value) || 1),
                        ),
                      })
                    }
                  />
                  <NumberField
                    label="Vs drop"
                    unit="%"
                    value={maxVsDrop}
                    min={0}
                    max={100}
                    step={5}
                    title={DROP_TIP}
                    onChange={setMaxVsDrop}
                  />
                </Fields>
              ) : (
                <>
                  <LayerTable layers={layers} onChange={setLayers} />
                  <div className="layers-drop">
                    <Fields>
                      <NumberField
                        label="Vs drop"
                        unit="%"
                        value={maxVsDrop}
                        min={0}
                        max={100}
                        step={5}
                        title={DROP_TIP}
                        onChange={setMaxVsDrop}
                      />
                    </Fields>
                  </div>
                </>
              )}
            </Card>

            <Card
              step={3}
              title="Sampling"
              hint="The Markov chains' length and number."
            >
              <Fields>
                <NumberField
                  label="Iterations"
                  value={nIterations}
                  min={1}
                  step={100}
                  onChange={setNIterations}
                />
                <NumberField
                  label="Burn-in"
                  value={nBurninIterations}
                  min={1}
                  step={100}
                  onChange={setNBurninIterations}
                />
                <NumberField
                  label="Chains"
                  value={nChains}
                  min={1}
                  step={1}
                  title={"Compared to judge them\n2 at least"}
                  onChange={setNChains}
                />
              </Fields>
            </Card>
          </fieldset>

          <RunPanel
            onRunning={setRunning}
            config={config}
            runUrl="/inversion/run"
            itemLabel="positions"
            itemLabelSingular="position"
            label="Invert"
            missing={missing}
            onDone={() => setRuns((n) => n + 1)}
            summary={
              <>
                <span>
                  <b>{selectedXmids.length}</b> positions to invert
                </span>
                <label
                  className="workers"
                  data-tip="Positions inverted at once"
                >
                  <input
                    type="number"
                    min={1}
                    max={maxWorkers}
                    value={nWorkers}
                    onChange={(e) =>
                      setNWorkers(
                        Math.max(
                          1,
                          Math.min(Number(e.target.value), maxWorkers),
                        ),
                      )
                    }
                  />
                  <span>workers</span>
                </label>
              </>
            }
            after={() => (
              <span className="run-next">
                <Link
                  to={`/visualization?run=${encodeURIComponent(folder)}&tab=inversion`}
                >
                  See the models <ArrowRightIcon size={13} />
                </Link>
              </span>
            )}
          />
        </div>
      )}
    </Page>
  );
}
