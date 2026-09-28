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
  const { preset, error } = usePreset("active", profile);
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
  const kept = `pac.form.active.${profile}`;
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
    `${kept}.trigger@3`,
    triggerDefault(preset),
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
  // Older backends have no image stacking stage: linear, as they stack.
  const [imageStacking, setImageStacking] = useStoredState(
    `${kept}.imageStacking`,
    preset.values.image_stacking
      ? stage<StackingState>(preset, "image_stacking")
      : { method: "linear", nu: 2, n: 2 },
  );
  const [workers, setWorkers] = useStoredState(`${kept}.workers`, 1);
  const [nPositions, setNPositions] = useState(0);

  // At most one worker a window.
  const maxWorkers = nPositions > 0 ? Math.min(nCpus, nPositions) : nCpus;
  // Fewer windows than workers: as many workers, for good (more windows later do not raise them).
  if (workers > maxWorkers) setWorkers(maxWorkers);
  const config = {
    profile,
    mode: "active",
    overrides: {
      masw: buildMaswParams(masw),
      trigger: buildTriggerParams(trigger, muting),
      muting: buildMutingParams(muting),
      filtering: buildFilteringParams(filtering),
      dispersion,
      ...(preset.values.image_stacking
        ? { image_stacking: buildStackingParams(imageStacking) }
        : {}),
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
          title="Dispersion image"
          hint="Each shot's dispersion image, then their stack."
        >
          <div className="rows">
            <DispersionRows
              dispersion={dispersion}
              setDispersion={setDispersion}
              nyquist={nyquist}
              kept={filtering.method === "iir" ? [filtering.fmin, filtering.fmax] : undefined}
            />
            {preset.values.image_stacking && (
              <StackingRow
                title="Dispersion image stacking"
                hint={"How the shots' dispersion images add up\nLinear: every shot counts\nRoot: what most shots share"}
                stacking={imageStacking}
                setStacking={setImageStacking}
                phaseWeighted={false}
              />
            )}
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
