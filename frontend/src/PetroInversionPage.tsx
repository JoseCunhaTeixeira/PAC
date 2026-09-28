import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { API } from "./api";
import { ArrowRightIcon, CpuIcon, OutcropIcon } from "./components/icons";
import {
  Callout,
  Card,
  Page,
  SelectField,
} from "./components/kit";
import {
  PositionRail,
  RailLegend,
  type RailCell,
} from "./components/PositionRail";
import { judged, useReceivers, useStageStates, xmidKey } from "./components/railStates";
import { num } from "./components/viz/format";
import { WorkersField } from "./components/computing";
import { RunPanel } from "./components/RunPanel";
import { runningJob } from "./components/jobs";
import { useStoredState } from "./components/stored";
import { RunSelect } from "./components/RunSelect";

// Petrophysical inversion: the fundamental mode's curves, a Silex model, then the run. Its
// results are read in Visualization.

// Picking labels aren't tied to a wave-type letter in this app (e.g. "M0",
// "L0", "R0" are all used) -- a pick counts as the fundamental mode when its
// trailing digits parse to mode number 0, regardless of the letter prefix.
function isFundamentalModeLabel(label: string): boolean {
  const match = /(\d+)$/.exec(label);
  return match !== null && Number(match[1]) === 0;
}

// A value, or how to change the previous one.
type Update<T> = T | ((previous: T) => T);

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export default function PetroInversionPage() {
  // The run, kept when the page is left.
  const [folder, setFolder] = useStoredState("pac.petro.folder", "");
  // Back on the page while its job runs: its run, so that its run bar shows the job.
  useEffect(() => {
    let cancelled = false;
    runningJob("petro_inversion").then((job) => {
      if (!cancelled && job) setFolder(job.target);
    });
    return () => {
      cancelled = true;
    };
  }, [setFolder]);
  const [xmids, setXmids] = useState<number[]>([]);
  const [positionPicks, setPositionPicks] = useState<
    { xmid: number; labels: string[] }[]
  >([]);
  // Each run's positions, the model and the workers, kept when the page is left.
  const [selections, setSelections] = useStoredState<Record<string, number[]>>(
    "pac.petro.selections",
    {},
  );
  const selectedPositions: Record<number, boolean> = Object.fromEntries(
    (selections[folder] ?? []).map((xmid) => [xmid, true]),
  );
  const setSelectedPositions = (update: Update<Record<number, boolean>>) =>
    setSelections((all) => {
      const previous = Object.fromEntries(
        (all[folder] ?? []).map((xmid) => [xmid, true]),
      );
      const next = typeof update === "function" ? update(previous) : update;
      return {
        ...all,
        [folder]: Object.keys(next)
          .map(Number)
          .filter((xmid) => next[xmid]),
      };
    });
  const [models, setModels] = useState<string[]>([]);
  const [modelName, setModelName] = useStoredState("pac.petro.model", "");
  const [nWorkers, setNWorkers] = useStoredState("pac.petro.workers", 1);
  const [running, setRunning] = useState(false); // the settings frozen while their job runs
  const [error, setError] = useState<string | null>(null);
  // Which positions are inverted, and their checks, as Visualization shows them; again after a run.
  const [runs, setRuns] = useState(0);
  const { states, loading: statesLoading } = useStageStates(
    folder,
    "petro",
    runs,
  );
  // The line's receivers: a window's cell is their spacing wide, whatever the step.
  const { receivers, loading: receiversLoading } = useReceivers(folder);
  // Starts true so the first render after picking a folder doesn't flash
  // "No picked dispersion data found" before the effect below runs.
  const [loadingPicks, setLoadingPicks] = useState(true);
  const nCpus = navigator.hardwareConcurrency || 1;

  useEffect(() => {
    fetch(`${API}/petro_inversion/models`)
      .then((res) => json<string[]>(res))
      .then((data) => {
        setModels(data);
        setModelName(
          (current) => current || (data.length === 1 ? data[0] : ""),
        );
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }, [setModelName]);

  const refresh = useCallback((folderName: string) => {
    Promise.resolve().then(() => setLoadingPicks(true));
    fetch(`${API}/xmids/${encodeURIComponent(folderName)}`)
      .then((res) => json<number[]>(res))
      .then(setXmids)
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
    fetch(
      `${API}/dispersion_picks_by_position/${encodeURIComponent(folderName)}`,
    )
      .then((res) => json<{ xmid: number; labels: string[] }[]>(res))
      .then(setPositionPicks)
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      )
      .finally(() => setLoadingPicks(false));
  }, []);

  useEffect(() => {
    Promise.resolve().then(() => {
      setXmids([]);
      setPositionPicks([]);
      setError(null);
    });
    if (folder) refresh(folder);
  }, [folder, refresh]);

  const eligible = xmids.filter((xmid) => {
    const picks = positionPicks.find((one) => one.xmid === xmid);
    return !!picks && picks.labels.some(isFundamentalModeLabel);
  });
  const selectedXmids = eligible.filter((xmid) => selectedPositions[xmid]);
  const maxWorkers =
    selectedXmids.length > 0 ? Math.min(nCpus, selectedXmids.length) : nCpus;
  // Fewer positions than workers: as many workers, for good (more positions later do not raise
  // them).
  if (nWorkers > maxWorkers) setNWorkers(maxWorkers);
  const cells: RailCell[] = xmids.map((xmid) => {
    const can = eligible.includes(xmid);
    const state = states.get(xmidKey(xmid));
    const verdict = judged(state);
    const lines = !can
      ? [`xmid ${num(xmid, 4)} m`, "no fundamental mode picked"]
      : state && verdict
        ? state.hover
        : [
            `xmid ${num(xmid, 4)} m`,
            ...(state?.hover.slice(1) ?? ["not inverted"]),
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
    model_name: modelName,
    n_workers: nWorkers,
  };
  const missing: string[] = [];
  if (!modelName) missing.push("a Silex model");
  if (selectedXmids.length === 0) missing.push("a window");

  return (
    <Page
      icon={<OutcropIcon size={24} />}
      title="Petrophysical inversion"
      subtitle="Invert the fundamental mode into soil types"
      art="petro"
      actions={
        <RunSelect folder={folder} onChange={setFolder} disabled={running} />
      }
    >
      {error && <Callout tone="error">{error}</Callout>}
      {folder && !loadingPicks && eligible.length === 0 && (
        <Callout tone="warn" title="No fundamental mode picked in this run">
          <Link to="/dispersion_picking">Pick M0 first.</Link>
        </Callout>
      )}

      {folder && eligible.length > 0 && (
        <div className="stack">
          <fieldset className="frozen" disabled={running}>
            <Card
              step={1}
              title="Curves to invert"
              hint="Fundamental-mode curves (M0, R0…) only."
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
              <div className="rail-head" style={{ marginTop: 0 }}>
                <span className="muted">
                  <b>{selectedXmids.length}</b> of {eligible.length} windows
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
              {!statesLoading && !receiversLoading && (
                <PositionRail
                  cells={cells}
                  receivers={receivers}
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
              title="Silex model"
              hint="The deep-learning model."
            >
              {models.length === 0 ? (
                <Callout tone="warn">No Silex model found.</Callout>
              ) : (
                <div style={{ maxWidth: 360 }}>
                  <SelectField
                    value={modelName}
                    onChange={setModelName}
                    icon={<CpuIcon size={15} />}
                  >
                    <option value="">Choose a model…</option>
                    {models.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </SelectField>
                </div>
              )}
            </Card>
          </fieldset>

          <RunPanel
            onRunning={setRunning}
            config={config}
            runUrl="/petro_inversion/run"
            itemLabel="windows"
            itemLabelSingular="window"
            label="Invert"
            missing={missing}
            onDone={() => setRuns((n) => n + 1)}
            summary={
              <>
                <span>
                  <b>{selectedXmids.length}</b>{" "}
                  {selectedXmids.length === 1 ? "window" : "windows"} to
                  invert
                </span>
                <WorkersField
                  workers={nWorkers}
                  setWorkers={setNWorkers}
                  maxWorkers={maxWorkers}
                  tip="Windows inverted in parallel"
                />
              </>
            }
            after={() => (
              <span className="run-next">
                <Link
                  to={`/visualization?run=${encodeURIComponent(folder)}&tab=petro`}
                >
                  See the sections <ArrowRightIcon size={13} />
                </Link>
              </span>
            )}
          />
        </div>
      )}
    </Page>
  );
}
