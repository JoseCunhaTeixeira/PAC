import { useState } from "react";
import { type Dispersion, type Masw } from "./api";
import {
  bound,
  buildFilteringParams,
  buildMaswParams,
  buildNormalizationParams,
  buildSelectionParams,
  buildStackingParams,
  buildWhiteningParams,
} from "./builders";
import {
  DispersionRows,
  FilteringRow,
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
import { written } from "./components/numbers";
import { shortestRecord } from "./components/records";
import { RunPanel } from "./components/RunPanel";
import { useStoredState } from "./components/stored";
import {
  type FilteringState,
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
  const nyquist = (acquisition.sampling_frequencies[0] ?? 0) / 2;
  const nCpus = navigator.hardwareConcurrency || 1;

  const [masw, setMasw] = useStoredState(
    `${kept}.masw`,
    stage<Masw>(preset, "masw"),
  );
  const [filtering, setFiltering] = useStoredState(
    `${kept}.filtering`,
    stage<FilteringState>(preset, "filtering"),
  );
  // The step, empty: the segment's length, segments end to end.
  const [slicing, setSlicing] = useState(() => {
    const given = stage<{ segment_duration: number; segment_step: number }>(preset, "slicing");
    return {
      segment_duration: given.segment_duration,
      segment_step: given.segment_step === given.segment_duration ? null : given.segment_step,
    } as { segment_duration: number; segment_step: number | null };
  });
  // What the records hold (see records.ts): a segment within the shortest record; its
  // frequency step, 1/segment.
  const data = shortestRecord(acquisition);
  const segment = bound(slicing.segment_duration);
  // @3: the selection always on, an earlier session's stored "off" is not kept.
  const [selection, setSelection] = useStoredState(
    `${kept}.selection@3`,
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
  // The band the filter and the whitening keep: the image overlaps it (sigpipe checks it).
  const filtered = filtering.method === "iir" ? [filtering.fmin, filtering.fmax] : [0, nyquist];
  const whitened = whitening.method === "onebit_apod" ? [whitening.fmin, whitening.fmax] : [0, nyquist];
  const keptBand: [number, number] = [Math.max(filtered[0], whitened[0]), Math.min(filtered[1], whitened[1])];
  const [dispersion, setDispersion] = useStoredState(
    `${kept}.dispersion`,
    stage<Dispersion>(preset, "dispersion"),
  );
  // @2: phase-weighted by default, an earlier session's stored stack is not kept.
  const [stacking, setStacking] = useStoredState(
    `${kept}.stacking@2`,
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
      // No shot: no distances to them, no velocities from them.
      masw: buildMaswParams({ ...masw, distance_min: null, distance_max: null }),
      filtering: buildFilteringParams(filtering),
      slicing: {
        segment_duration: slicing.segment_duration,
        segment_step: bound(slicing.segment_step) ?? slicing.segment_duration,
      },
      selection: buildSelectionParams({ ...selection, method: "fk" }),
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
            <FilteringRow
              filtering={filtering}
              setFiltering={setFiltering}
              nyquist={nyquist}
              acquisition={acquisition}
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
                  max={data}
                  step={0.05}
                />
                <NumberField
                  label="Step"
                  unit="s"
                  value={slicing.segment_step ?? Number.NaN}
                  optional={segment !== null ? written(segment) : "segment"}
                  onChange={(v) => setSlicing({ ...slicing, segment_step: v })}
                  min={0.01}
                  max={segment ?? data}
                  step={0.01}
                />
              </Fields>
            </Row>
            <Row
              icon={<SlidersIcon size={16} />}
              title="Segment selection"
              hint="Keeps the segments whose energy travels along the line (FK, always on)."
            >
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
                {/* The band's velocities: empty, the band open on that side (0, ∞). */}
                <NumberField
                  label="Slowest"
                  unit="m/s"
                  value={selection.vmin ?? Number.NaN}
                  optional="0"
                  onChange={(v) => setSelection({ ...selection, vmin: v })}
                  min={0}
                />
                <NumberField
                  label="Fastest"
                  unit="m/s"
                  value={selection.vmax ?? Number.NaN}
                  optional="∞"
                  onChange={(v) => setSelection({ ...selection, vmax: v })}
                  gt={bound(selection.vmin) ?? 0}
                />
              </Fields>
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
                  {/* Two of a segment's frequency steps at least: sigpipe whitens no narrower. */}
                  <NumberField
                    label="To"
                    unit="Hz"
                    value={whitening.fmax}
                    onChange={(v) => setWhitening({ ...whitening, fmax: v })}
                    min={segment !== null ? whitening.fmin + 2 / segment : undefined}
                    gt={segment !== null ? undefined : whitening.fmin}
                    max={nyquist}
                    step={5}
                  />
                  {/* Narrower than the band, when there is one (its "To" says otherwise). */}
                  <NumberField
                    label="Taper"
                    unit="Hz"
                    value={whitening.taper_width_Hz}
                    onChange={(v) =>
                      setWhitening({ ...whitening, taper_width_Hz: v })
                    }
                    gt={0}
                    lt={whitening.fmax > whitening.fmin ? whitening.fmax - whitening.fmin : undefined}
                    max={nyquist / 4}
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
              kept={keptBand}
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
