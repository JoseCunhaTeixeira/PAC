import { useState } from "react";
import { type Dispersion, type Masw } from "./api";
import {
  buildFilteringParams,
  buildMutingParams,
  buildStackingParams,
  buildWindowParams,
} from "./builders";
import {
  DispersionRows,
  FilteringRow,
  MutingRow,
  NextSteps,
  Row,
  StackingRow,
  TriggerRow,
  WindowsCard,
  WorkersField,
  type FormProps,
} from "./components/computing";
import { ScissorsIcon } from "./components/icons";
import {
  Callout,
  Card,
  Fields,
  NumberField,
  Segmented,
} from "./components/kit";
import { above } from "./components/numbers";
import { RunPanel } from "./components/RunPanel";
import { useStoredState } from "./components/stored";
import {
  type FilteringState,
  type MutingState,
  type PresetDefaults,
  type StackingState,
  type WindowState,
  stage,
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
  const maxTime = Number(acquisition.durations[0]?.toFixed(2) ?? 0);
  const nyquist = (acquisition.sampling_frequencies[0] ?? 0) / 2;
  const nCpus = navigator.hardwareConcurrency || 1;

  const [masw, setMasw] = useStoredState(
    `${kept}.masw`,
    stage<Masw>(preset, "masw"),
  );
  const [trigger, setTrigger] = useStoredState(
    `${kept}.trigger`,
    stage<{ t0: number }>(preset, "trigger"),
  );
  const [muting, setMuting] = useStoredState(
    `${kept}.muting`,
    stage<MutingState>(preset, "muting"),
  );
  const [filtering, setFiltering] = useStoredState(
    `${kept}.filtering`,
    stage<FilteringState>(preset, "filtering"),
  );
  const [surfaceWaves, setSurfaceWaves] = useStoredState(
    `${kept}.surfaceWaves`,
    stage<WindowState>(preset, "correlation_window"),
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
      masw,
      trigger,
      muting: buildMutingParams(muting),
      filtering: buildFilteringParams(filtering),
      correlation_window: buildWindowParams(surfaceWaves),
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
            <TriggerRow t0={trigger.t0} setT0={(t0) => setTrigger({ t0 })} />
            <MutingRow
              acquisition={acquisition}
              muting={muting}
              setMuting={setMuting}
              maxTime={maxTime}
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
            <Row
              icon={<ScissorsIcon size={16} />}
              title="Surface-wave muting"
              hint="Keeps the arrivals between two velocities, before correlating."
              control={
                <Segmented
                  size="sm"
                  label="Surface-wave window"
                  value={surfaceWaves.method}
                  onChange={(method) =>
                    setSurfaceWaves({ ...surfaceWaves, method })
                  }
                  options={[
                    { value: "none", label: "Off" },
                    { value: "mute", label: "Mute" },
                  ]}
                />
              }
            >
              {surfaceWaves.method === "mute" && (
                <Fields>
                  <NumberField
                    label="Slowest"
                    unit="m/s"
                    value={surfaceWaves.vmin}
                    onChange={(v) =>
                      setSurfaceWaves({ ...surfaceWaves, vmin: v })
                    }
                    min={0}
                  />
                  <NumberField
                    label="Fastest"
                    unit="m/s"
                    value={surfaceWaves.vmax}
                    onChange={(v) =>
                      setSurfaceWaves({ ...surfaceWaves, vmax: v })
                    }
                    min={0}
                    check={above(surfaceWaves.vmin)}
                  />
                  <NumberField
                    label="Taper"
                    unit="samples"
                    value={surfaceWaves.taper}
                    onChange={(v) =>
                      setSurfaceWaves({ ...surfaceWaves, taper: v })
                    }
                    min={0}
                  />
                </Fields>
              )}
            </Row>
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
