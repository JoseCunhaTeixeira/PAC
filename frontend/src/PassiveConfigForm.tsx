import { useState } from "react";
import { type Dispersion, type Masw } from "./api";
import {
  buildFilteringParams,
  buildMutingParams,
  buildNormalizationParams,
  buildSelectionParams,
  buildStackingParams,
  buildWhiteningParams,
} from "./builders";
import {
  DispersionRows,
  FilteringRow,
  MutingRow,
  NextSteps,
  Row,
  StackingRow,
  WindowsCard,
  WorkersField,
  type FormProps,
} from "./components/computing";
import {
  ClockIcon,
  PulseIcon,
  SlidersIcon,
  SpectrumIcon,
} from "./components/icons";
import {
  Callout,
  Card,
  Fields,
  NumberField,
  Segmented,
} from "./components/kit";
import { above, below, upTo } from "./components/numbers";
import { RunPanel } from "./components/RunPanel";
import { useStoredState } from "./components/stored";
import {
  type FilteringState,
  type MutingState,
  type PresetDefaults,
  type SelectionState,
  type StackingState,
  type WhiteningState,
  stage,
  usePreset,
} from "./presets";

export function ConfigForm({ acquisition, profile, onRunning }: FormProps) {
  const { preset, error } = usePreset("passive", profile);
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
  const kept = `pac.form.passive.${profile}`;
  const [running, setRunning] = useState(false);
  const maxTime = Number(Math.max(...acquisition.durations).toFixed(2));
  const nyquist = (acquisition.sampling_frequencies[0] ?? 0) / 2;
  const nCpus = navigator.hardwareConcurrency || 1;

  const [masw, setMasw] = useStoredState(
    `${kept}.masw`,
    stage<Masw>(preset, "masw"),
  );
  const [muting, setMuting] = useStoredState(
    `${kept}.muting`,
    stage<MutingState>(preset, "muting"),
  );
  const [filtering, setFiltering] = useStoredState(
    `${kept}.filtering`,
    stage<FilteringState>(preset, "filtering"),
  );
  const [slicing, setSlicing] = useState(() =>
    stage<{ segment_duration: number; segment_step: number }>(
      preset,
      "slicing",
    ),
  );
  const [selection, setSelection] = useStoredState(
    `${kept}.selection`,
    stage<SelectionState>(preset, "selection"),
  );
  const [whitening, setWhitening] = useStoredState(
    `${kept}.whitening`,
    stage<WhiteningState>(preset, "whitening"),
  );
  const [normalization, setNormalization] = useStoredState(
    `${kept}.normalization`,
    stage<{ method: string }>(preset, "normalization"),
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
    mode: "passive",
    overrides: {
      masw,
      muting: buildMutingParams(muting),
      filtering: buildFilteringParams(filtering),
      slicing,
      selection: buildSelectionParams(selection),
      whitening: buildWhiteningParams(whitening),
      normalization: buildNormalizationParams(normalization),
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
          showSources={false}
          unit="records"
        />

        <Card
          step={2}
          title="Preprocessing"
          hint="Applied to each record first."
        >
          <div className="rows">
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
          hint="Short segments, cleaned, correlated, then stacked into virtual shots."
        >
          <div className="rows">
            <Row
              icon={<ClockIcon size={16} />}
              title="Slicing"
              hint="The segments each record is cut into."
            >
              <Fields>
                <NumberField
                  label="Segment"
                  unit="s"
                  value={slicing.segment_duration}
                  onChange={(v) =>
                    setSlicing({ ...slicing, segment_duration: v })
                  }
                  min={0.1}
                  max={maxTime}
                  step={0.05}
                />
                <NumberField
                  label="Step"
                  unit="s"
                  value={slicing.segment_step}
                  onChange={(v) => setSlicing({ ...slicing, segment_step: v })}
                  min={0.01}
                  max={maxTime}
                  check={upTo(slicing.segment_duration)}
                  step={0.01}
                />
              </Fields>
            </Row>
            <Row
              icon={<SlidersIcon size={16} />}
              title="Segment selection"
              hint="Keeps the segments whose energy travels along the line."
              control={
                <Segmented
                  size="sm"
                  label="Selection"
                  value={selection.method}
                  onChange={(method) => setSelection({ ...selection, method })}
                  options={[
                    { value: "none", label: "Off" },
                    { value: "fk", label: "FK" },
                  ]}
                />
              }
            >
              {selection.method === "fk" && (
                <Fields>
                  <NumberField
                    label="Threshold"
                    value={selection.threshold}
                    onChange={(v) =>
                      setSelection({ ...selection, threshold: v })
                    }
                    min={0}
                    max={1}
                    step={0.1}
                  />
                  <NumberField
                    label="Slowest"
                    unit="m/s"
                    value={selection.vmin}
                    onChange={(v) => setSelection({ ...selection, vmin: v })}
                    min={0}
                  />
                  <NumberField
                    label="Fastest"
                    unit="m/s"
                    value={selection.vmax}
                    onChange={(v) => setSelection({ ...selection, vmax: v })}
                    min={0}
                    check={above(selection.vmin)}
                  />
                </Fields>
              )}
            </Row>
            <Row
              icon={<SpectrumIcon size={16} />}
              title="Spectral whitening"
              hint="Flattens each segment's spectrum."
              control={
                <Segmented
                  size="sm"
                  label="Whitening"
                  value={whitening.method}
                  onChange={(method) => setWhitening({ ...whitening, method })}
                  options={[
                    { value: "none", label: "Off" },
                    { value: "onebit", label: "One-bit" },
                    { value: "onebit_apod", label: "One-bit, tapered" },
                  ]}
                />
              }
            >
              {whitening.method === "onebit_apod" && (
                <Fields>
                  <NumberField
                    label="From"
                    unit="Hz"
                    value={whitening.fmin}
                    onChange={(v) => setWhitening({ ...whitening, fmin: v })}
                    min={0}
                    max={nyquist}
                    step={5}
                  />
                  <NumberField
                    label="To"
                    unit="Hz"
                    value={whitening.fmax}
                    onChange={(v) => setWhitening({ ...whitening, fmax: v })}
                    min={0}
                    max={nyquist}
                    check={above(whitening.fmin)}
                    step={5}
                  />
                  <NumberField
                    label="Taper"
                    unit="Hz"
                    value={whitening.taper_width_Hz}
                    onChange={(v) =>
                      setWhitening({ ...whitening, taper_width_Hz: v })
                    }
                    min={0}
                    max={nyquist / 4}
                    // Narrower than the band, when there is one (its "To" says otherwise).
                    check={
                      whitening.fmax > whitening.fmin
                        ? below(whitening.fmax - whitening.fmin)
                        : undefined
                    }
                    step={1}
                  />
                </Fields>
              )}
            </Row>
            <Row
              icon={<PulseIcon size={16} />}
              title="Temporal normalization"
              hint="Evens out loud and quiet moments."
              control={
                <Segmented
                  size="sm"
                  label="Normalization"
                  value={normalization.method}
                  onChange={(method) =>
                    setNormalization({ ...normalization, method })
                  }
                  options={[
                    { value: "none", label: "Off" },
                    { value: "onebit", label: "One-bit" },
                  ]}
                />
              }
            />
            <StackingRow
              title="Correlation stacking"
              hint="How the segments' correlations add up."
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
        missing={nPositions === 0 ? ["a window"] : []}
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
