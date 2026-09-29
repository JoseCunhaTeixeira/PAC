"""The quality of a run's records: sigpipe's measures of each preprocessed record
(sigpipe.masw.quality.signal: dead, clipped and NaN traces, the SNR and usable band from a
surface-wave window against a noise window, the neighbouring traces' coherence, the trigger the
first breaks point to), the records and traces the run left out, and for a run the assistant
made, its signal check's verdicts (G1) and the preprocessing's attempts. Shown along the line at
each record's shot: the records strip of Visualization, and the selected record's card."""

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from masw.io.paths import workspace
from masw.io.quality.done import record_ranges, record_settings, with_spreads
from masw.io.quality.files import folder_path, line_geometry, read_manifest
from masw.io.quality.log import (
    AttemptSummary,
    GateResult,
    Metric,
    QCLog,
    config_section,
    line_receivers,
    read_log,
)
from masw.io.quality.runs import stage_text
from masw.io.quality.sources import stacking_windows
from masw.io.quality.view import (
    Card,
    Cell,
    Overview,
    Sentence,
    Status,
    Track,
    flag_text,
    gate_view,
    measured_status,
    number,
    plural,
    span,
    verdict_sentence,
    verdict_status,
    warnings,
)
from sigpipe.base.stream import Stream
from sigpipe.dataio.signal_plotting import load_spectra
from sigpipe.dataio.stream.loading import load_stream
from sigpipe.masw.pipelines import shot_time_s, unmuted_record
from sigpipe.masw.pipelines.common import PREPROCESSED
from sigpipe.masw.presets import Preset
from sigpipe.masw.profiles import Profile, ProfileError, load_profile
from sigpipe.masw.quality.signal import (
    Windows,
    dead_clipped_nan,
    first_breaks,
    lateral_coherence,
    signal_windows,
    snr_db,
    trigger_shift,
    usable_band,
)
from sigpipe.masw.quality.spectra import save_record_spectra
from sigpipe.masw.runs import RunManifest, xmid_of

STAGE = "preprocessing"
GATE = "G1"
logger = logging.getLogger(__name__)

MEASURES_FILE = "SignalMeasures_0000.json"  # beside a record's preprocessed file, PAC's job's
GATHER_SAMPLES = 1_200  # the most samples a trace keeps for the wiggle plot


class SignalThresholds(BaseModel):
    """G1's windows and limits: the assistant's defaults, or those the run's checks used."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    vg_min: float = 80.0  # m/s, the slowest surface wave: the window's end
    vg_max: float = 1_500.0  # m/s, the fastest arrival: its start
    pad_s: float = 0.05
    dead_ratio: float = 0.01
    clip_share: float = 0.005
    min_snr_db: float = 6.0
    band_db: float = 6.0
    peak_db: float = 20.0
    min_coherence: float = 0.5
    max_trigger_shift_s: float = 0.01
    max_trigger_scatter_s: float = 0.05
    first_break_ratio: float = 5.0


class SignalMeasures(BaseModel):
    """A preprocessed record's measures against G1's limits and its usable band, saved when PAC's
    job processed it: Visualization reads them and never measures."""

    model_config = ConfigDict(frozen=True)

    metrics: tuple[Metric, ...]
    band_hz: tuple[float, float] | None


def measure_records(run_folder: Path) -> None:
    """Save the measures of every record run `run_folder` preprocessed, beside it (a run PAC made:
    the assistant's own are in its QC log), and its spectra's figure (the preprocessed record's):
    measured as G1 measures it, its noise before its muting (before_muting)."""
    manifest = read_manifest(run_folder)
    if manifest is None:
        return
    thresholds = thresholds_of(run_folder)
    active = manifest.profile.kind == "active"
    try:
        profile: Profile | None = load_profile(manifest.profile.name, workspace())
    except ProfileError:
        profile = None
    for record in manifest.records:
        path = run_folder / record.folder / PREPROCESSED
        if record.status != "succeeded" or not path.exists():
            continue
        stream = load_stream([path])[0]
        unmuted = before_muting(manifest.preset, profile, record.name)
        metrics, band = signal_metrics(stream, thresholds, active, unmuted)
        measures = SignalMeasures(metrics=metrics, band_hz=band)
        (path.parent / MEASURES_FILE).write_text(measures.model_dump_json(indent=2))
        try:
            save_record_spectra(stream, path.parent, band)
        except Exception:  # a figure must not lose the measures
            logger.exception("Could not draw the spectra of %s", path.parent)


def before_muting(
    preset: Preset, profile: Profile | None, name: str
) -> tuple[Stream, float] | None:
    """Record `name` before its muting, and where its shot is on it (s): what G1 measures its
    noise on (a muting zeroes the noise window after the slowest arrival; a trigger's shift,
    part of the muting, drops the one before the trigger), preprocessed as `preset` has it from
    its input file (sigpipe's unmuted_record: neither its trigger shifted nor muted, its shot
    at shot_time_s). None when it is not muted (its saved record is the same), or its input is
    not at hand."""
    values: dict[str, Any] = preset.model_dump()
    muting: dict[str, Any] = values.get("muting") or {}
    records = profile.records if profile is not None else ()
    found = next((one for one in records if one.path.name == name), None)
    if (
        profile is None
        or found is None
        or muting.get("method", "none") == "none"
        or not found.path.exists()
    ):
        return None
    return unmuted_record(preset, found, profile), shot_time_s(preset, found)


class RecordCard(Card):
    """A record's card: its measures and verdict, what the windows took of it."""

    attempts: tuple[AttemptSummary, ...] = ()
    x: float | None  # its shot along the line; None: no source (passive)
    windows: tuple[str, ...]  # the windows that stack it
    excluded_traces: tuple[int, ...]


def thresholds_of(run_folder: Path) -> SignalThresholds:
    return SignalThresholds.model_validate(config_section(run_folder, "signal"))


def records_overview(folder: str) -> Overview:
    """Each record at its shot: its signal check's verdict, or its measures against the check's
    limits, and its median SNR."""
    run_folder = folder_path(folder)
    manifest = _manifest(run_folder, folder)
    log = read_log(run_folder)
    thresholds = thresholds_of(run_folder)
    sources = line_geometry(run_folder, manifest).sources
    excluded = set(manifest.exclusions.records)
    cells: list[Cell] = []
    line_out, _ = line_receivers(log)
    for record in manifest.records:
        g1, metrics, band = _judged(run_folder, manifest, log, record.name)
        snr = next((one.value for one in metrics if one.name == "snr_db"), None)
        # The record's own, the receivers left out of every window apart.
        left = [
            trace
            for trace in manifest.exclusions.traces.get(record.name, ())
            if trace not in line_out
        ]
        status: Status
        if record.status == "failed" or record.name in excluded:
            status = "fail"
        else:
            status = verdict_status(g1) if g1 is not None else measured_status(metrics)
        hover = [
            record.name
            + (f" · shot at {number(sources[record.name], 4)} m" if record.name in sources else ""),
            "failed to preprocess"
            if record.status == "failed"
            else "left out of every window"
            if record.name in excluded
            else f"median SNR {number(snr)} dB"
            if snr is not None
            else "not measured",
        ]
        if band is not None:
            hover.append(f"usable band {span(*band, 'Hz')}")
        if left:
            hover.append(f"{plural(len(left), 'trace')} left out")
        if g1 is not None and g1.flags:
            hover.append("flags: " + ", ".join(flag_text(flag.name) for flag in g1.flags))
        cells.append(
            Cell(
                key=record.name,
                x=sources.get(record.name),
                status=status,
                hover=tuple(hover),
                value=snr,
            )
        )
    failed = {record.name for record in manifest.records if record.status == "failed"}
    preprocessed = len(manifest.records) - len(failed)
    used = len(manifest.records) - len(excluded | failed)
    traces = sum(len(one) for one in manifest.exclusions.traces.values())
    summary = f"{preprocessed} of {plural(len(manifest.records), 'record')} preprocessed" + (
        " and used" if used == preprocessed else f", {used} used"
    )
    if excluded:
        summary += f" · {len(excluded)} left out by the signal check"
    if traces:
        summary += f" · {plural(traces, 'trace')} left out"
    paco = log is not None
    return Overview(
        paco=paco,
        summary=summary,
        legend={
            "pass": "passed the signal check (G1)" if paco else "measures within the limits",
            "warn": "used, still flagged" if paco else "a measure beyond its limit",
            "fail": "left out of every window" if paco else "failed to preprocess",
            "none": "not measured",
        },
        cells=tuple(cells),
        track=Track(
            label="Median SNR (dB)",
            short="SNR",
            kind="value",
            limit=thresholds.min_snr_db,
            bound="min",
        ),
    )


class RecordGather(BaseModel):
    """A record as the windows used it (preprocessed), for the wiggle plot along the line."""

    model_config = ConfigDict(frozen=True)

    name: str
    positions: tuple[float, ...]  # m along the line, each trace's receiver
    source: float | None  # the shot's; None: none (passive)
    dt: float  # s between the samples kept
    traces: tuple[tuple[float, ...], ...]  # each normalized to its largest value, or the record's
    excluded: tuple[int, ...]  # the traces left out of the windows


def record_gather(folder: str, name: str, norm: str = "trace") -> RecordGather:
    """Record `name` preprocessed, every trace at most GATHER_SAMPLES samples (kept one in
    several: the display does not need the rest), normalized by trace or over the record."""
    run_folder = folder_path(folder)
    manifest = _manifest(run_folder, folder)
    record = next((one for one in manifest.records if one.name == name), None)
    if record is None or record.status != "succeeded":
        raise ValueError(f"No preprocessed record {name} in folder={folder}")
    path = run_folder / record.folder / PREPROCESSED
    if not path.exists():
        raise ValueError(f"No preprocessed file for record {name} in folder={folder}")
    stream = load_stream([path])[0]
    return gather_of(
        stream,
        name,
        norm,
        with_source=manifest.profile.kind != "passive",
        excluded=manifest.exclusions.traces.get(name, ()),
    )


def gather_of(
    stream: Stream,
    name: str,
    norm: str = "trace",
    with_source: bool = True,
    excluded: tuple[int, ...] = (),
) -> RecordGather:
    """`stream` as the wiggle plot takes it, the whole of it: every trace at most GATHER_SAMPLES
    samples (kept one in several: the display does not need the rest), normalized by trace or
    over the gather; its source's position when `with_source`."""
    xt = np.nan_to_num(np.asarray(stream.xt, dtype=float))
    stride = max(1, -(-xt.shape[1] // GATHER_SAMPLES))
    kept = xt[:, ::stride]
    scale = (
        np.abs(kept).max(axis=1, keepdims=True)
        if norm == "trace"
        else np.full((kept.shape[0], 1), np.abs(kept).max())
    )
    kept = kept / np.where(scale > 0, scale, 1.0)
    source = stream.acquisition.source
    ts = np.asarray(stream.ts, dtype=float)
    return RecordGather(
        name=name,
        positions=tuple(round(float(receiver.x), 3) for receiver in stream.acquisition.receivers),
        source=round(float(source.x), 3) if with_source else None,
        dt=float(ts[1] - ts[0]) * stride,
        traces=tuple(tuple(round(float(value), 2) for value in trace) for trace in kept),
        excluded=excluded,
    )


class SavedSpectra(BaseModel):
    """A stream's spectra as its job saved them beside it (sigpipe's save_spectra), for the
    spectrum view: each trace's amplitude spectrum at its receiver along the line."""

    model_config = ConfigDict(frozen=True)

    positions: tuple[float, ...]  # m
    freqs: tuple[float, ...]  # Hz, 0 to Nyquist
    amplitude: tuple[tuple[float, ...], ...]  # traces x freqs, 0 to 1 of each trace's largest
    band_hz: tuple[float, float] | None  # the band drawn over them: the record's usable band


def saved_spectra(folder: Path, what: str) -> SavedSpectra:
    """The spectra saved in `folder` (a record's, a window's stacked correlations'), as they
    were saved; a ValueError when none were (a run from before they were)."""
    found = load_spectra(folder)
    if found is None:
        raise ValueError(f"No spectra saved for {what}")
    return SavedSpectra(
        positions=tuple(round(float(x), 3) for x in found.positions),
        freqs=tuple(round(float(f), 3) for f in found.freqs),
        amplitude=tuple(tuple(round(float(v), 3) for v in trace) for trace in found.amplitude),
        band_hz=found.band,
    )


def record_spectra(folder: str, name: str) -> SavedSpectra:
    """Record `name`'s spectra, preprocessed, as its job saved them."""
    run_folder = folder_path(folder)
    manifest = _manifest(run_folder, folder)
    record = next((one for one in manifest.records if one.name == name), None)
    if record is None or record.status != "succeeded":
        raise ValueError(f"No preprocessed record {name} in folder={folder}")
    return saved_spectra(run_folder / record.folder, f"record {name} in folder={folder}")


def record_card(folder: str, name: str) -> RecordCard:
    run_folder = folder_path(folder)
    manifest = _manifest(run_folder, folder)
    record = next((one for one in manifest.records if one.name == name), None)
    if record is None:
        raise ValueError(f"No record {name} in folder={folder}")
    log = read_log(run_folder)
    thresholds = thresholds_of(run_folder)
    g1, metrics, band = _judged(run_folder, manifest, log, name)
    sources = line_geometry(run_folder, manifest).sources
    windows = stacking_windows(run_folder).get(name, ())
    excluded_traces = manifest.exclusions.traces.get(name, ())
    line_out, why_out = line_receivers(log)
    own = tuple(trace for trace in excluded_traces if trace not in line_out)
    said: list[Sentence] = []
    verdict = verdict_sentence("record", g1)
    if record.status == "failed":
        said.append(Sentence(mark="fail", text=f"Failed to preprocess: {record.error}."))
    elif name in manifest.exclusions.records and (g1 is None or g1.verdict != "reject"):
        said.append(
            Sentence(mark="fail", text="Left out of every window by the signal check (G1).")
        )
    said += _signal(metrics, band, thresholds)
    if own:
        said.append(_traces(log, name, own, run_folder, manifest))
    if len(own) < len(excluded_traces):
        said.append(Sentence(mark="info", text=why_out))
    said += _redone(log, name)
    said += warnings(g1)
    if windows:
        xmids = [xmid_of(unit) for unit in windows]
        said.append(
            Sentence(
                mark="info",
                text=f"Stacked by {plural(len(windows), 'window')}, xmid "
                f"{span(min(xmids), max(xmids), 'm', digits=4)}.",
            )
        )
    elif record.status == "succeeded" and name not in manifest.exclusions.records:
        said.append(Sentence(mark="info", text="No window stacks it: its shot is out of reach."))
    x = sources.get(name)
    status: Status = (
        "fail"
        if record.status == "failed" or name in manifest.exclusions.records
        else verdict_status(g1)
        if g1 is not None
        else measured_status(metrics)
    )
    return RecordCard(
        key=name,
        status=status,
        title=name + (f" · shot at {number(x, 4)} m" if x is not None else ""),
        verdict=verdict,
        sentences=tuple(said),
        gates=(gate_view(GATE, g1, metrics),),
        attempts=log.summaries(name, STAGE) if log is not None else (),
        x=x,
        windows=windows,
        excluded_traces=excluded_traces,
        settings=with_spreads(
            name,
            {one.name: record_settings(manifest, log, one.name) for one in manifest.records},
            "record",
            ranges=record_ranges(manifest, log),
        ),
    )


def _judged(
    run_folder: Path,
    manifest: RunManifest,
    log: QCLog | None,
    name: str,
) -> tuple[GateResult | None, tuple[Metric, ...], tuple[float, float] | None]:
    """G1's result on the record, its metrics and usable band; for a run PAC made, the measures
    of its preprocessed file against G1's limits PAC's job saved."""
    g1 = log.result(name, STAGE, GATE) if log is not None else None
    if g1 is not None:
        return g1, g1.metrics, g1.kept.band_hz
    record = next((one for one in manifest.records if one.name == name), None)
    if record is None or record.status != "succeeded":
        return None, (), None
    metrics, band = _measured(run_folder / record.folder / PREPROCESSED)
    return None, metrics, band


def _signal(
    metrics: tuple[Metric, ...], band: tuple[float, float] | None, thresholds: SignalThresholds
) -> list[Sentence]:
    """The record's signal in two sentences: its SNR and band, its traces' state and trigger."""
    values = {metric.name: metric for metric in metrics}
    said: list[Sentence] = []
    snr = values.get("snr_db")
    coherence = values.get("lateral_coherence")
    if snr is not None and snr.value is not None:
        parts = [f"median SNR {number(snr.value)} dB (at least {number(thresholds.min_snr_db)})"]
        if band is not None:
            parts.append(f"usable band {span(*band, 'Hz')}")
        if coherence is not None and coherence.value is not None:
            parts.append(
                f"neighbouring traces' coherence {number(coherence.value, 2)} "
                f"(at least {number(thresholds.min_coherence, 2)})"
            )
        good = snr.passed and (coherence is None or coherence.passed)
        said.append(Sentence(mark="pass" if good else "warn", text=_sentence(", ".join(parts))))
    bad = {
        name: int(metric.value)
        for name in ("dead_traces", "clipped_traces", "nan_traces", "rms_outliers")
        if (metric := values.get(name)) is not None and metric.value
    }
    if bad:
        said.append(
            Sentence(
                mark="warn",
                text=_sentence(
                    ", ".join(f"{count} {flag_text(name)}" for name, count in bad.items())
                ),
            )
        )
    shift = values.get("trigger_shift_s")
    if shift is not None and shift.value is not None and not shift.passed:
        said.append(
            Sentence(
                mark="warn",
                text=f"The first breaks put the trigger {number(shift.value * 1000)} ms off.",
            )
        )
    return said


def _traces(
    log: QCLog | None,
    name: str,
    traces: tuple[int, ...],
    run_folder: Path,
    manifest: RunManifest,
) -> Sentence:
    receivers = line_geometry(run_folder, manifest).receivers
    where = ", ".join(
        f"{number(receivers[trace], 4)} m" if trace < len(receivers) else f"#{trace}"
        for trace in traces
    )
    reasons = dict.fromkeys(
        flag_text(flag.name)
        for flag in (log.raised(name, STAGE) if log is not None else ())
        if flag.action.kind == "exclude_traces"
    )
    return Sentence(
        mark="info",
        text=f"Its {'trace' if len(traces) == 1 else 'traces'} at {where} left out of the windows"
        + (f" ({', '.join(reasons)})" if reasons else "")
        + ".",
    )


def _redone(log: QCLog | None, name: str) -> list[Sentence]:
    """Each time the signal check had the record preprocessed again, and with what."""
    if log is None:
        return []
    said: list[Sentence] = []
    for attempt in log.of(name, STAGE)[1:]:
        flag = attempt.triggered_by.split(":", 1)[-1]
        changed = "; ".join(
            stage_text(stage, cast(dict[str, object], values))
            for stage, values in attempt.parameters.items()
            if isinstance(values, dict)
        )
        said.append(
            Sentence(
                mark="info",
                text=f"Preprocessed again ({flag_text(flag)})"
                + (f": {changed}" if changed else "")
                + ".",
            )
        )
    return said


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] + "."


def signal_metrics(
    stream: Stream,
    thresholds: SignalThresholds,
    active: bool,
    unmuted: tuple[Stream, float] | None = None,
) -> tuple[tuple[Metric, ...], tuple[float, float] | None]:
    """sigpipe's measures of preprocessed record `stream` against G1's limits, as G1 names and
    measures them, over every trace (G1 leaves out those beyond the line's reach, and the traces
    already excluded): the bad traces; for a shot, the median SNR in the surface-wave window and
    the usable band, their noise on the record before its muting (`unmuted`: it and where its
    shot is on it, before_muting; none, not muted), the median coherence of neighbouring traces
    and the trigger the first breaks point to. Returns the metrics and the usable band itself."""
    xt = np.asarray(stream.xt, dtype=float)
    dead, clipped, nan = dead_clipped_nan(xt, thresholds.dead_ratio, thresholds.clip_share)
    metrics = [
        Metric(name=name, value=int(mask.sum()), threshold=0, bound="max", passed=not mask.any())
        for name, mask in (("dead_traces", dead), ("clipped_traces", clipped), ("nan_traces", nan))
    ]
    offsets = np.asarray(stream.acquisition.offsets, dtype=float)
    ts = np.asarray(stream.ts, dtype=float)
    windows = signal_windows(offsets, ts, thresholds.vg_min, thresholds.vg_max, thresholds.pad_s)
    if not active or windows is None:
        return tuple(metrics), None
    usable = ~(dead | clipped | nan)
    finite = np.nan_to_num(xt)
    noisy, noise = _noise(unmuted, finite, windows, offsets, thresholds)
    snr = snr_db(noisy, noise)
    median_snr = float(np.median(snr[usable])) if usable.any() else None
    snr_ok = median_snr is not None and median_snr >= thresholds.min_snr_db
    band = usable_band(
        noisy[usable],
        stream.sampling_freq,
        _rows(noise, usable),
        thresholds.band_db,
        thresholds.peak_db,
    )
    max_lag_s = (
        float(np.diff(np.sort(offsets)).min()) / thresholds.vg_min if offsets.size > 1 else 0
    )
    coherence, _ = lateral_coherence(finite, windows, stream.sampling_freq, max_lag_s)
    pairs = usable[:-1] & usable[1:]
    median_coherence = float(np.median(coherence[pairs])) if pairs.any() else None
    breaks = first_breaks(finite, ts, windows, thresholds.first_break_ratio)
    breaks[~usable | (snr < thresholds.min_snr_db)] = np.nan
    fit = trigger_shift(breaks, offsets) if snr_ok else None
    metrics += [
        Metric(
            name="snr_db",
            value=_rounded(median_snr),
            threshold=thresholds.min_snr_db,
            bound="min",
            passed=snr_ok,
            unit="dB",
        ),
        Metric(
            name="usable_band_hz",
            value=None if band is None else round(band[1] - band[0], 2),
            threshold=0,
            bound="min",
            passed=band is not None,
            unit="Hz",
        ),
        Metric(
            name="lateral_coherence",
            value=_rounded(median_coherence),
            threshold=thresholds.min_coherence,
            bound="min",
            passed=median_coherence is not None and median_coherence >= thresholds.min_coherence,
        ),
        Metric(
            name="trigger_shift_s",
            value=None if fit is None else _rounded(fit[0]),
            threshold=thresholds.max_trigger_shift_s,
            bound="max",
            passed=fit is None or abs(fit[0]) <= thresholds.max_trigger_shift_s,
            unit="s",
        ),
        Metric(
            name="trigger_scatter_s",
            value=None if fit is None else _rounded(fit[2]),
            threshold=thresholds.max_trigger_scatter_s,
            bound="max",
            passed=fit is None or fit[2] <= thresholds.max_trigger_scatter_s,
            unit="s",
        ),
    ]
    rounded = None if band is None else (round(band[0], 2), round(band[1], 2))
    return tuple(metrics), rounded


def _noise(
    unmuted: tuple[Stream, float] | None,
    finite: np.ndarray,
    windows: Windows,
    offsets: np.ndarray,
    thresholds: SignalThresholds,
) -> tuple[np.ndarray, Windows]:
    """The traces and windows G1 measures the noise on (its _before_muting): the record before
    its muting's, its times from its shot; the preprocessed record's (`finite`, `windows`) when
    not muted, or when the record before its muting leaves no room for a noise window."""
    if unmuted is None:
        return finite, windows
    stream, shot_s = unmuted
    found = signal_windows(
        offsets,
        np.asarray(stream.ts, dtype=float) - shot_s,
        thresholds.vg_min,
        thresholds.vg_max,
        thresholds.pad_s,
    )
    if found is None or stream.xt.shape != finite.shape:
        return finite, windows
    return np.nan_to_num(np.asarray(stream.xt, dtype=float)), found


def _measured(path: Path) -> tuple[tuple[Metric, ...], tuple[float, float] | None]:
    """The measures of the preprocessed record in `path` as PAC's job saved them; none when it
    saved none as new as the record (a run processed before they were saved)."""
    saved = path.parent / MEASURES_FILE
    if not path.exists() or not saved.exists() or saved.stat().st_mtime < path.stat().st_mtime:
        return (), None
    measures = SignalMeasures.model_validate_json(saved.read_text())
    return measures.metrics, measures.band_hz


def _rows(windows: Windows, rows: np.ndarray) -> Windows:
    return Windows(windows.signal[rows], windows.noise[rows], windows.where)


def _rounded(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 4)


def _manifest(run_folder: Path, folder: str) -> RunManifest:
    manifest = read_manifest(run_folder)
    if manifest is None:
        raise ValueError(f"No run manifest in folder={folder}: a folder of the older layout")
    return manifest
