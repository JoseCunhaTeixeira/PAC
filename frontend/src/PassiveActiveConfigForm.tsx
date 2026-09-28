import { useState } from "react";
import { type Dispersion, type Masw } from "./api";
import {
  buildFilteringParams,
  buildMaswParams,
  buildMutingParams,
  buildStackingParams,
  buildTriggerParams,
} from "./builders";
import {
  DispersionRows,
  FilteringRow,
  MutingRow,
  NextSteps,
  StackingRow,
  WindowsCard,
  WorkersField,
  type FormProps,
} from "./components/computing";
import { Callout, Card } from "./components/kit";
import { RunPanel } from "./components/RunPanel";
import { useStoredState } from "./components/stored";
import {
  type FilteringState,
  type MutingState,
  type PresetDefaults,
  type StackingState,
  stage,
  triggerDefault,
  usePreset,
} from "./presets";

export function ConfigForm({ acquisition, profile, onRunning }: FormProps) {
  const { preset, error } = usePreset("passive-active", profile);
  if (error)
    return (
      <Callout tone="error" title="The settings could not load">
        {error}
      </Callout>
    );
  if (!preset) return <p className="muted">Loading the settings…</p>;
  return (
    <Form
      acquisition={acquisition}
      profile={profile}
      preset={preset}
      onRunning={onRunning}
    />
  );
}

function Form({
  acquisition,
  profile,
  preset,
  onRunning,
}: FormProps & { preset: PresetDefaults }) {
  // The settings, kept for this profile when the page is left; frozen while their job runs.
  const kept = `pac.form.passive-active.${profile}`;
  const [running, setRunning] = useState(false);
  const nyquist = (acquisition.sampling_frequencies[0] ?? 0) / 2;
  const nCpus = navigator.hardwareConcurrency || 1;

  const [masw, setMasw] = useStoredState(
    `${kept}.masw`,
    stage<Masw>(preset, "masw"),
  );
  // @2: since the bounds may be empty and the trigger the files' (2026-09-28), an earlier
  // session's stand-in values are not kept.
  const [trigger, setTrigger] = useStoredState(
    `${kept}.trigger@2`,
    triggerDefault(preset, acquisition.triggers),
  );
  const [muting, setMuting] = useStoredState(
    `${kept}.muting@2`,
    stage<MutingState>(preset, "muting"),
  );
  const [filtering, setFiltering] = useStoredState(
    `${kept}.filtering`,
    stage<FilteringState>(preset, "filtering"),
  );
  const [dispersion, setDispersion] = useStoredState(
    `${kept}.dispersion`,
    stage<Dispersion>(preset, "dispersion"),
  );
  const [stacking, setStacking] = useStoredState(
    `${kept}.stacking`,
    stage<StackingState>(preset, "stacking"),
  );
  const [workers, setWorkers] = useStoredState(`${kept}.workers`, 1);
  const [nPositions, setNPositions] = useState(0);

  // At most one worker a window.
  const maxWorkers = nPositions > 0 ? Math.min(nCpus, nPositions) : nCpus;
  // Fewer windows than workers: as many workers, for good (more windows later do not raise them).
  if (workers > maxWorkers) setWorkers(maxWorkers);
  const config = {
    profile,
    mode: "passive-active",
    overrides: {
      masw: buildMaswParams(masw),
      trigger: buildTriggerParams(trigger, muting),
      muting: buildMutingParams(muting),
      filtering: buildFilteringParams(filtering),
      stacking: buildStackingParams(stacking),
      dispersion,
    },
    workers,
  };

  return (
    <>
      <fieldset className="frozen" disabled={running}>
        <WindowsCard
          acquisition={acquisition}
          masw={masw}
          setMasw={setMasw}
          onCount={setNPositions}
          showSources
        />

        <Card
          step={2}
          title="Preprocessing"
          hint="Applied to each shot first."
        >
          <div className="rows">
            <MutingRow
              acquisition={acquisition}
              muting={muting}
              setMuting={setMuting}
              trigger={{ t0: trigger.t0, setT0: (t0) => setTrigger({ t0 }) }}
            />
            <FilteringRow
              filtering={filtering}
              setFiltering={setFiltering}
              nyquist={nyquist}
            />
          </div>
        </Card>

        <Card
          step={3}
          title="Interferometry"
          hint="Each shot correlated with its nearest receiver, then stacked into a virtual shot."
        >
          <div className="rows">
            <StackingRow
              title="Correlation stacking"
              hint="How the shots' correlations add up."
              stacking={stacking}
              setStacking={setStacking}
            />
          </div>
        </Card>

        <Card
          step={4}
          title="Dispersion image"
          hint="The stacked virtual shot's dispersion image."
        >
          <div className="rows">
            <DispersionRows
              dispersion={dispersion}
              setDispersion={setDispersion}
              nyquist={nyquist}
              kept={filtering.method === "iir" ? [filtering.fmin, filtering.fmax] : undefined}
            />
          </div>
        </Card>
      </fieldset>

      <RunPanel
        onRunning={(now) => {
          setRunning(now);
          onRunning?.(now);
        }}
        config={config}
        missing={nPositions === 0 ? ["a window with a shot"] : []}
        summary={
          <>
            <span>
              <b>{nPositions}</b> windows to compute
            </span>
            <WorkersField
              workers={workers}
              setWorkers={setWorkers}
              maxWorkers={maxWorkers}
            />
          </>
        }
        after={(job) => <NextSteps job={job} />}
      />
    </>
  );
}
