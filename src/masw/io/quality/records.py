"""The quality of a run's records: sigpipe's measures of each preprocessed record
(sigpipe.masw.quality.signal: dead, clipped and NaN traces, the SNR and usable band from a
surface-wave window against a noise window, the neighbouring traces' coherence, the trigger the
first breaks point to), the records and traces the run left out, and for a run the agent made,
its signal check's verdicts (G1) and the preprocessing's attempts. Shown along the line at each
record's shot: the records strip of Visualization, and the selected record's card."""

import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from masw.io.paths import workspace
from masw.io.quality.done import attempt_steps, record_ranges, record_settings, with_spreads
from masw.io.quality.files import folder_path, line_geometry, read_manifest
from masw.io.quality.log import (
    GateResult,
    Metric,
    QCLog,
    config_section,
    line_receivers,
    read_log,
)
from masw.io.quality.sources import stacking_windows
from masw.io.quality.view import (
    Card,
    Cell,
    Overview,
    Sentence,
    States,
    Status,
    Track,
    by_whom,
    flag_text,
    gate_view,
    line_gate,
    measured_status,
    measured_warnings,
    number,
    plural,
    span,
    state_lines,
    verdict_sentence,
    verdict_status,
    warnings,
)
from sigpipe.base.stream import Stream
from sigpipe.dataio.signal_plotting import TraceSpectra, load_spectra, stream_spectra
from sigpipe.dataio.stream.loading import load_stream
from sigpipe.masw.pipelines import shot_time_s, trigger_shift_s, unmuted_record
from sigpipe.masw.pipelines.common import PREPROCESSED
from sigpipe.masw.presets import Preset
from sigpipe.masw.profiles import Profile, ProfileError, Record, load_profile
from sigpipe.masw.quality.measures import SignalLimits, line_reach, measure_signal
from sigpipe.masw.quality.spectra import save_record_spectra
from sigpipe.masw.runs import RunManifest, xmid_of

STAGE = "preprocessing"
GATE = "G1"
logger = logging.getLogger(__name__)

MEASURES_FILE = "SignalMeasures_0000.json"  # beside a record's preprocessed file, PAC's job's
GATHER_SAMPLES = 1_200  # the most samples a trace keeps for the wiggle plot


class SignalMeasures(BaseModel):
    """A preprocessed record's measures against G1's limits and its usable band, saved when PAC's
    job processed it: Visualization reads them and never measures."""

    model_config = ConfigDict(frozen=True)

    metrics: tuple[Metric, ...]
    band_hz: tuple[float, float] | None


def measure_records(run_folder: Path) -> None:
    """Save the measures of every record run `run_folder` preprocessed, beside it (a run PAC made:
    the assistant's own are in its QC log), and its spectra's figure (the preprocessed record's):
    measured as G1 measures it, its noise before its muting (before_muting), within the line's
    reach as G1's (sigpipe's line_reach: where the traces' SNR, by distance from the shots,
    falls under its limit; an active line's)."""
    manifest = read_manifest(run_folder)
    if manifest is None:
        return
    limits = thresholds_of(run_folder)
    active = manifest.profile.kind == "active"
    try:
        profile: Profile | None = load_profile(manifest.profile.name, workspace())
    except ProfileError:
        profile = None
    records = {one.path.name: one for one in profile.records} if profile is not None else {}
    done = [
        (record, path)
        for record in manifest.records
        if record.status == "succeeded"
        and (path := run_folder / record.folder / PREPROCESSED).exists()
    ]
    # Each record before its muting and where its shot is on it: its noise as G1 measures it.
    unmuted = {
        record.name: single_precision(found)
        for record, _ in done
        if (found := before_muting(manifest.preset, profile, record.name)) is not None
    }
    reach_m = (
        line_reach(
            (
                unmuted.get(record.name)
                or (
                    load_stream([path])[0],
                    shot_context(manifest.preset, records.get(record.name))[0],
                )
                for record, path in done
            ),
            limits,
            _span(profile),
        )
        if active and profile is not None
        else None
    )
    for record, path in done:
        stream = load_stream([path])[0]
        shot_s, applied_s = shot_context(manifest.preset, records.get(record.name))
        metrics, band = signal_metrics(
            stream,
            limits,
            active=active,
            shot_s=shot_s,
            applied_s=applied_s,
            excluded=manifest.exclusions.traces.get(record.name, ()),
            reach_m=reach_m,
            unmuted=unmuted.get(record.name),
            # On every line, as G1's: the same measures for every line.
            spectra=True,
            image_band=image_band(manifest.preset),
        )
        measures = SignalMeasures(metrics=metrics, band_hz=band)
        (path.parent / MEASURES_FILE).write_text(measures.model_dump_json(indent=2))
        try:
            save_record_spectra(stream, path.parent, band)
        except Exception:  # a figure must not lose the measures
            logger.exception("Could not draw the spectra of %s", path.parent)


def single_precision(found: tuple[Stream, float]) -> tuple[Stream, float]:
    """A record before its muting in single precision: a line's records held at once."""
    stream, shot_s = found
    return replace(stream, xt=np.asarray(stream.xt, dtype=np.float32)), shot_s


def _span(profile: Profile) -> float:
    positions = [receiver.x for receiver in profile.receivers]
    return max(positions) - min(positions) if positions else 0.0


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

    x: float | None  # its shot along the line; None: no source (passive)
    windows: tuple[str, ...]  # the windows that stack it
    excluded_traces: tuple[int, ...]


def thresholds_of(run_folder: Path) -> SignalLimits:
    return SignalLimits.model_validate(config_section(run_folder, "signal"))


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
    # A record's red: not used, rejected by the signal check, or failed (left out) otherwise.
    states = States(none="not measured", failed="not used")
    # Who preprocessed each record: the agent when its log has the record's run, else by hand.
    makers: Counter[str] = Counter()
    for record in manifest.records:
        makers[_maker(log, record.name)] += 1
        g1, metrics, band = _judged(run_folder, manifest, log, record.name)
        snr = next((one.value for one in metrics if one.name == "snr_db"), None)
        # The record's own, the receivers left out of every window apart.
        left = [
            trace
            for trace in manifest.exclusions.traces.get(record.name, ())
            if trace not in line_out
        ]
        if record.status == "failed":
            status = states.add("fail", judged=False)
            said = ["Failed to preprocess"]
        elif record.name in excluded and g1 is None:
            status = states.add("fail", judged=False)
            said = ["Left out by the user"]
        else:
            judged = g1 is not None
            status = states.add(
                "fail"
                if record.name in excluded
                else verdict_status(g1)
                if judged
                else measured_status(metrics),
                judged=judged,
            )
            said = state_lines(status, g1, metrics=metrics)
        hover = [
            record.name
            + (f" · shot at {number(sources[record.name], 4)} m" if record.name in sources else ""),
            *said,
        ]
        if snr is not None:
            hover.append(f"Median SNR {number(snr)} dB")
        if band is not None:
            hover.append(f"Usable band {span(*band, 'Hz')}")
        if left:
            hover.append(f"{plural(len(left), 'trace')} left out")
        cells.append(
            Cell(
                key=record.name,
                x=sources.get(record.name),
                status=status,
                hover=tuple(hover),
                value=snr,
            )
        )
    traces = sum(len(one) for one in manifest.exclusions.traces.values())
    paco = log is not None
    said = [
        plural(len(manifest.records), "record"),
        *(f"{count} preprocessed {by}" for by, count in makers.items()),
        *states.counts(),
    ]
    if traces:
        said.append(f"{plural(traces, 'trace')} left out")
    return Overview(
        paco=paco,
        summary=" · ".join(said),
        legend=states.legend(),
        cells=tuple(cells),
        gates=line_gate(log, "preprocessing", "G1"),
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
    return _as_view(found)


def _as_view(found: TraceSpectra) -> SavedSpectra:
    return SavedSpectra(
        positions=tuple(round(float(x), 3) for x in found.positions),
        freqs=tuple(round(float(f), 3) for f in found.freqs),
        amplitude=tuple(tuple(round(float(v), 3) for v in trace) for trace in found.amplitude),
        band_hz=found.band,
    )


def record_spectra(folder: str, name: str) -> SavedSpectra:
    """Record `name`'s spectra, preprocessed, as its job saved them; computed from its saved
    record when it saved none (a run from before they were saved), with the usable band its
    measures give."""
    run_folder = folder_path(folder)
    manifest = _manifest(run_folder, folder)
    record = next((one for one in manifest.records if one.name == name), None)
    if record is None or record.status != "succeeded":
        raise ValueError(f"No preprocessed record {name} in folder={folder}")
    found = load_spectra(run_folder / record.folder)
    if found is None:
        path = run_folder / record.folder / PREPROCESSED
        if not path.exists():
            raise ValueError(f"No preprocessed file for record {name} in folder={folder}")
        _, _, band = _judged(run_folder, manifest, read_log(run_folder), name)
        found = stream_spectra(load_stream([path])[0], band)
    return _as_view(found)


def record_card(folder: str, name: str) -> RecordCard:
    run_folder = folder_path(folder)
    manifest = _manifest(run_folder, folder)
    record = next((one for one in manifest.records if one.name == name), None)
    if record is None:
        raise ValueError(f"No record {name} in folder={folder}")
    log = read_log(run_folder)
    g1, metrics, _ = _judged(run_folder, manifest, log, name)
    sources = line_geometry(run_folder, manifest).sources
    windows = stacking_windows(run_folder).get(name, ())
    excluded_traces = manifest.exclusions.traces.get(name, ())
    line_out, why_out = line_receivers(log)
    own = tuple(trace for trace in excluded_traces if trace not in line_out)
    said: list[Sentence] = []
    verdict = verdict_sentence(g1)
    if record.status != "failed":
        said.append(Sentence(mark="info", text=f"Preprocessed {_maker(log, name)}."))
    if record.status == "failed":
        said.append(Sentence(mark="fail", text=f"Failed to preprocess: {record.error}."))
    elif name in manifest.exclusions.records and (g1 is None or g1.verdict != "reject"):
        said.append(Sentence(mark="fail", text="Left out of every window by the signal check."))
    if own:
        said.append(_traces(log, name, own, run_folder, manifest))
    if len(own) < len(excluded_traces):
        said.append(Sentence(mark="info", text=why_out))
    said += warnings(g1) if g1 is not None else measured_warnings(metrics)
    if windows:
        xmids = [xmid_of(unit) for unit in windows]
        said.append(
            Sentence(
                mark="info",
                text=f"Used by the {plural(len(windows), 'window')} centred from "
                f"{number(min(xmids), 4)} to {number(max(xmids), 4)} m."
                if len(windows) > 1
                else f"Used by the window centred at {number(xmids[0], 4)} m.",
            )
        )
    elif record.status == "succeeded" and name not in manifest.exclusions.records:
        said.append(
            Sentence(mark="info", text="Used by no window: its shot is beyond the shots used.")
        )
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
        steps=attempt_steps(log, name, (STAGE,), manifest.profile.receiver_spacing_m),
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


def _maker(log: QCLog | None, name: str) -> str:
    """Who preprocessed record `name`: the run's agent, when its log holds the record's
    preprocessing; else the user."""
    return by_whom(log, log is not None and bool(log.of(name, STAGE)))


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
    metrics, band = saved_measures(run_folder / record.folder / PREPROCESSED)
    return None, metrics, band


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
        + (f": {', '.join(reasons)}" if reasons else "")
        + ".",
    )


def signal_metrics(
    stream: Stream,
    limits: SignalLimits,
    *,
    active: bool,
    shot_s: float = 0.0,
    applied_s: float | None = None,
    excluded: tuple[int, ...] = (),
    reach_m: float | None = None,
    unmuted: tuple[Stream, float] | None = None,
    spectra: bool = False,
    image_band: tuple[float, float] | None = None,
) -> tuple[tuple[Metric, ...], tuple[float, float] | None]:
    """Preprocessed record `stream`'s measures as G1 measures it (sigpipe's measure_signal, the
    same definition and inputs): its traces `excluded` left out, those within `reach_m` of the
    shot (the line's reach) measured, its times from the shot at `shot_s` (`applied_s`: the
    shift its muting applied), its noise on the record before its muting (`unmuted`), its
    traces against their neighbours' spectra with `spectra`, its SNR and coherence in the part
    of its usable band the dispersion images use (`image_band`). Returns the metrics, each
    with what it covers, and the usable band itself."""
    report = measure_signal(
        stream,
        limits,
        source="shot" if active else "none",
        shot_s=shot_s,
        applied_s=applied_s,
        excluded=excluded,
        reach_m=reach_m,
        before_muting=unmuted,
        spectra=spectra,
        image_band=image_band,
    )
    band = report.band
    rounded = None if band is None else (round(band[0], 2), round(band[1], 2))
    return tuple(Metric(**one.model_dump()) for one in report.measures), rounded


def image_band(preset: Preset) -> tuple[float, float] | None:
    """The band the run's dispersion images use (its dispersion's fmin and fmax), what G1 and
    G2 measure a signal's SNR and coherence in (PACo's image_band); None when either is not
    set."""
    dispersion: dict[str, Any] = preset.model_dump().get("dispersion") or {}
    fmin, fmax = dispersion.get("fmin"), dispersion.get("fmax")
    if isinstance(fmin, int | float) and isinstance(fmax, int | float):
        return float(fmin), float(fmax)
    return None


def shot_context(preset: Preset, found: Record | None) -> tuple[float, float | None]:
    """Where record `found`'s shot is on its preprocessed file, and the shift its muting applied
    (None: off), as G1 takes them (PACo's trigger_context): with its muting on, at 0 once moved
    by the trigger; off, where its file puts it (0 when it says nothing)."""
    if found is None:
        return 0.0, None
    shift = trigger_shift_s(preset, found)
    muting: dict[str, Any] = preset.model_dump().get("muting") or {}
    muted = muting.get("method", "none") != "none" and "trigger" in type(preset).model_fields
    return shot_time_s(preset, found) - shift, shift if muted else None


def saved_measures(path: Path) -> tuple[tuple[Metric, ...], tuple[float, float] | None]:
    """The measures of the preprocessed record in `path` as PAC's job saved them; none when it
    saved none as new as the record (a run processed before they were saved)."""
    saved = path.parent / MEASURES_FILE
    if not path.exists() or not saved.exists() or saved.stat().st_mtime < path.stat().st_mtime:
        return (), None
    measures = SignalMeasures.model_validate_json(saved.read_text())
    return measures.metrics, measures.band_hz


def _manifest(run_folder: Path, folder: str) -> RunManifest:
    manifest = read_manifest(run_folder)
    if manifest is None:
        raise ValueError(f"No run manifest in folder={folder}: a folder of the older layout")
    return manifest
