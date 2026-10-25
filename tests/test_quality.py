"""Visualization's API on a synthetic run (tests/synthetic.py): what the files of a run PAC made
alone give, stage by stage (the run's card and its line, the shots each window stacks, records,
images and picks, seismic and petrophysical inversions), then the same run as the assistant
judged it, its QC log written here as PACo writes it: the gates' verdicts, metrics, flags and
attempts, and what each attempt changed."""

import json
import os
import shutil
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from paco.qc.g1_signal import SignalThresholds, judge_signal

from masw.api.main import app
from masw.io.paths import OUTPUT_DIR, workspace
from masw.io.quality.dispersion import curve_metrics
from masw.io.quality.done import in_common, record_settings, records_in_common
from masw.io.quality.files import read_manifest
from masw.io.quality.history import CHECKS, register_check, unregister_check
from masw.io.quality.journal import JournalEntry, journal_of, search_of, study_of
from masw.io.quality.log import LINE, Attempt, GateResult, Metric, QCLog
from masw.io.quality.records import before_muting, signal_metrics, thresholds_of
from masw.io.quality.runs import USER_REASON, near_field_used, processing_settings
from masw.io.quality.tried import tried_tables
from masw.io.quality.view import (
    DASH,
    Sentence,
    Setting,
    acted_on,
    measured_status,
    state_lines,
    verdict_sentence,
    warnings,
)
from sigpipe.base import (
    Coordinate,
    DispersionCurve,
    DispersionImage,
    LinearAcquisition,
    Mode,
    Stream,
    VelocityType,
)
from sigpipe.dataio.stream.loading import load_stream
from sigpipe.masw.inversion.measuring import USEFUL_REFERENCE, InversionMeasures
from sigpipe.masw.inversion.window import (
    VS_SPREAD_FILE,
)
from sigpipe.masw.presets import make_preset, resolve_preset
from sigpipe.masw.profiles import load_profile
from sigpipe.masw.quality.curve import CurveLimits
from sigpipe.masw.quality.measures import SignalLimits, line_reach
from sigpipe.masw.quality.signal import signal_windows

client = TestClient(app)

BOX = {"fmin": 10, "fmax": 60, "vmin": 100, "vmax": 400, "label": "M0"}
PARAMETERS = {
    "n_layers": 2,
    "vs_layers": [{"vs_min": 100, "vs_max": 400, "vs_perturb_std": 20}] * 2,
    "thickness_layers": [{"thickness_min": 1, "thickness_max": 5, "thickness_perturb_std": 1}],
    "n_iterations": 1_000,
    "n_burnin_iterations": 100,
    "n_chains": 2,
}


def _wait(job: dict[str, Any]) -> dict[str, Any]:
    for _ in range(600):
        job = client.get(f"/jobs/{job['id']}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.5)
    raise AssertionError("the job did not end")


def _texts(card: dict[str, Any]) -> list[str]:
    return [sentence["text"] for sentence in card["sentences"]]


def _how(step: dict[str, Any]) -> str | None:
    """How a step was done, as its badge says it."""
    return step["how"]["label"] if step.get("how") else None


def _done(card: dict[str, Any]) -> list[tuple[str, str, str | None, list[str]]]:
    """A card's "What was done": each step's title, who did it and how (its badges), its lines."""
    return [(step["title"], step["by"], _how(step), step["lines"]) for step in card["steps"]]


@pytest.fixture(scope="module")
def run() -> str:
    """shots processed in three windows, M0 picked in two, both inverted, seismically and with
    the bundled Silex model; its output folder."""
    request = {
        "profile": "shots",
        "mode": "active",
        "overrides": {"masw": {"length": 6, "step": 3}},
        "workers": 2,
    }
    job = _wait(client.post("/run", json=request).json())
    assert job["state"] == "succeeded"
    folder: str = job["run"]
    for xmid in (2.5, 5.5):
        assert (
            client.post(f"/dispersion_images/{folder}/{xmid}/pick/box", json=BOX).status_code == 200
        )
    config = {
        "folder": folder,
        "positions": [2.5, 5.5],
        "labels": ["M0"],
        "parameters": PARAMETERS,
        "n_workers": 2,
    }
    assert _wait(client.post("/inversion/run", json=config).json())["state"] == "succeeded"
    (model,) = client.get("/petro_inversion/models").json()
    petro = {"folder": folder, "positions": [2.5, 5.5], "model_name": model, "n_workers": 2}
    assert _wait(client.post("/petro_inversion/run", json=petro).json())["state"] == "succeeded"
    return folder


def test_the_runs_are_listed_by_profile(run: str) -> None:
    profiles = {one["profile"]: one for one in client.get("/quality/runs").json()}

    # Every input profile, with its runs or none.
    assert set(profiles) >= {"shots", "noise"}
    assert profiles["noise"]["runs"] == [] and profiles["noise"]["records"] is True
    entry = next(one for one in profiles["shots"]["runs"] if one["folder"] == run)
    assert entry["made_by"] == "manual" and entry["mode"] == "active" and entry["windows"] == 3
    assert entry["window_length"] == 6  # receivers, as the run's form set them
    started = [one["started_at"] for one in profiles["shots"]["runs"]]
    assert started == sorted(started, reverse=True)  # the newest first


def test_a_pac_run_card_says_its_windows_came_from_pacs_form(run: str) -> None:
    card = client.get(f"/quality/run/{run}").json()

    assert (card["profile"], card["made_by"], card["records"]) == ("shots", "manual", True)
    settings = {one["key"]: one for one in card["settings"]}
    # In the order its records went through them: the windows, the records, the shots the windows
    # stack, then each step on (the mute and the filter off, as by default: not said), the curves
    # and the Vs profiles last.
    assert list(settings) == [
        "windows",
        "records",
        "reach",
        "band",
        "image_stacking",
        "curves",
        "profiles",
    ]
    windows = settings["windows"]
    assert (windows["value"], windows["detail"]) == (
        "6 receivers (5 m)",
        "every 3 receivers (3 m) · 3 windows",
    )
    assert windows["origin"] == "you" and windows["why"].startswith("set by the user")
    # The phase shift as the preset has it: the preset's default, as the windows' cards say.
    assert (settings["band"]["label"], settings["band"]["origin"]) == (
        "Dispersion image",
        "default",
    )
    assert (settings["band"]["value"], settings["band"]["detail"]) == (
        f"0{DASH}100 Hz",
        f"1{DASH}1000 m/s",
    )
    assert (settings["reach"]["label"], settings["reach"]["value"]) == (
        "Shots used",
        "any distance",
    )
    assert (settings["records"]["value"], settings["records"]["detail"]) == ("2 of 2", "all used")
    curves = settings["curves"]
    assert (curves["value"], curves["detail"], curves["origin"]) == (
        "2 of 3",
        "windows picked",
        "you",
    )
    # No check judged its models: how many windows were inverted.
    profiles = settings["profiles"]
    assert (profiles["label"], profiles["value"], profiles["detail"]) == (
        "Vs profiles",
        "2 of 3",
        "windows inverted",
    )
    # Nothing compared, nothing to say why: its settings, the user's, said by its tiles; what
    # was done, a step each, its facts a line each: the records, the windows imaged, picked and
    # inverted.
    assert card["tried"] == [] and card["reasons"] == []
    assert [
        (step["kind"], step["by"], _how(step), step["title"], step["lines"])
        for step in card["history"]
    ] == [
        ("preprocessing", "you", "active", "Preprocessed the records", ["2 of 2 records"]),
        (
            "phase_shift",
            "you",
            "active",
            "Processed the whole line",
            ["3 of 3 windows", "Windows of 6 receivers (5 m), one every 3 receivers (3 m)"],
        ),
        (
            "picking",
            "you",
            "box",
            "Picked the curves",
            ["2 of 3 windows", "xmid 2.5 m, xmid 5.5 m"],
        ),
        (
            "inversion",
            "you",
            "MCMC",
            "Inverted the curves into Vs profiles",
            ["2 of 3 windows", "xmid 2.5 m, xmid 5.5 m"],
        ),
        (
            "petro_inversion",
            "you",
            "Silex",
            "Inverted the curves into soil columns",
            ["2 of 3 windows", "xmid 2.5 m, xmid 5.5 m"],
        ),
    ]
    # The line: every receiver, both shots, each window's first and last receivers.
    assert card["receivers"] == [float(x) for x in range(12)]
    assert card["sources"] == {"1.mseed": -2.0, "2.mseed": 13.0}
    assert [(one["xmid"], one["first"], one["last"]) for one in card["windows"]] == [
        (2.5, 0.0, 5.0),
        (5.5, 3.0, 8.0),
        (8.5, 6.0, 11.0),
    ]
    stages = {one["key"]: (one["done"], one["total"]) for one in card["stages"]}
    assert stages == {"records": (2, 2), "dispersion": (2, 3), "inversion": (2, 3), "petro": (2, 3)}


def test_a_window_says_which_shots_it_stacks(run: str) -> None:
    sources = client.get(f"/quality/sources/{run}/5.5").json()

    assert (sources["first"], sources["last"], sources["receivers"]) == (3.0, 8.0, 6)
    assert sources["passive"] is False and sources["stacked"] == 2
    assert [(one["name"], one["use"]) for one in sources["shots"]] == [
        ("1.mseed", "used"),
        ("2.mseed", "used"),
    ]
    assert _texts(sources) == [
        "Its dispersion image stacks the images of 2 shots, 7.5 m from its middle, "
        "1 on its left, 1 on its right."
    ]


def test_a_window_says_why_it_leaves_shots_out(run: str) -> None:
    folder = f"{run}-reach"
    target = OUTPUT_DIR / folder
    shutil.copytree(OUTPUT_DIR / run, target)
    manifest = json.loads((target / "run.json").read_text())
    manifest["preset"]["masw"]["distance_max"] = 10.0
    manifest["exclusions"] = {"records": ["1.mseed"], "traces": {}}
    (target / "run.json").write_text(json.dumps(manifest))
    window = json.loads((target / "xmid_2.50" / "window.json").read_text())
    for key in ("selected_files", "acquisitions"):
        window[key] = []
    (target / "xmid_2.50" / "window.json").write_text(json.dumps(window))

    sources = client.get(f"/quality/sources/{folder}/2.5").json()

    shots = {one["name"]: one for one in sources["shots"]}
    assert shots["1.mseed"]["use"] == "excluded"
    assert shots["1.mseed"]["why"] == "left out of every window by the signal check"
    assert shots["2.mseed"]["use"] == "far"
    assert shots["2.mseed"]["why"] == "10.5 m from its middle: beyond the farthest shot used, 10 m"
    # Each reason in full, in its sentence: no hover to say it again.
    assert _texts(sources) == [
        "Shots not used: 1 farther than 10 m, 1 left out by the signal check."
    ]
    assert sources["sentences"][0]["detail"] == ""
    card = client.get(f"/quality/run/{folder}").json()
    reach = next(one for one in card["settings"] if one["key"] == "reach")
    assert reach["value"] == "up to 10 m"


def test_a_passive_window_stacks_every_record() -> None:
    job = _wait(
        client.post(
            "/run",
            json={
                "profile": "noise",
                "mode": "passive",
                "overrides": {
                    "masw": {"length": 6, "step": 6},
                    "selection": {"method": "fk", "threshold": 0.0},
                },
                "workers": 1,
            },
        ).json()
    )
    assert job["state"] == "succeeded"
    card = client.get(f"/quality/run/{job['run']}").json()
    assert card["sources"] == {} and card["reach"] is None
    # Its settings in a passive line's order: no shots; the filter off, as by default, not said.
    assert [one["key"] for one in card["settings"]] == [
        "windows",
        "records",
        "slicing",
        "selection",
        "whitening",
        "normalization",
        "stacking",
        "band",
        "curves",
        "profiles",
    ]
    said = {one["key"]: (one["label"], one["value"], one["detail"]) for one in card["settings"]}
    assert said["slicing"] == ("Segments", "1 s", "every 1 s")
    assert said["selection"] == ("Segment selection", "threshold 0", "")
    assert said["stacking"] == ("Stacking", "phase-weighted", "")
    xmid = card["windows"][0]["xmid"]

    sources = client.get(f"/quality/sources/{job['run']}/{xmid}").json()

    assert sources["passive"] is True and sources["shots"] == []
    assert _texts(sources)[1] == "On a passive line, every record serves every window."
    # Its stacked correlations' spectra, as the job saved them: each trace's, 0 to 1.
    spectra = client.get(f"/quality/dispersion/spectrum/{job['run']}/{xmid}").json()
    assert len(spectra["amplitude"]) == len(spectra["positions"]) > 1
    assert {len(one) for one in spectra["amplitude"]} == {len(spectra["freqs"])}
    assert max(max(one) for one in spectra["amplitude"]) == 1.0 and spectra["band_hz"] is None
    # Its fk selection, as the job saved it: every segment's ratio, kept or not; measured on its
    # card, in its table, not said again in a sentence.
    selection = client.get(f"/quality/dispersion/selection/{job['run']}/{xmid}").json()
    assert selection["threshold"] == 0.0 and selection["segments"] == len(selection["ratios"]) > 0
    assert selection["kept_count"] == sum(selection["kept"])
    window = client.get(f"/quality/dispersion/card/{job['run']}/{xmid}").json()
    assert not any("fk selection" in text for text in _texts(window))
    metrics = {one["name"]: one for gate in window["gates"] for one in gate["metrics"]}
    assert metrics["fk_segments"]["value"] == selection["segments"]
    assert metrics["fk_kept"]["value"] == round(selection["kept_share"], 4)
    # Judged as G2 judges them: the share kept against its limit; the stacked correlations
    # measured as a record is, from the virtual source, as the job saved them.
    assert metrics["fk_kept"]["threshold"] == 0.01 and metrics["fk_kept"]["of"] == "selection"
    snr = metrics["snr_db"]
    assert snr["of"] == "signal" and snr["threshold"] == 6.0
    assert "the virtual source's own aside" in snr["over"]  # in the log, not on the page
    assert {"lateral_coherence", "usable_band_hz", "dead_traces"} <= set(metrics)


def test_a_pac_run_shows_its_records_measures_without_verdicts(run: str) -> None:
    overview = client.get(f"/quality/records/overview/{run}").json()

    assert overview["paco"] is False
    assert overview["summary"] == (
        "2 records · 2 preprocessed by the user · 2 within the agent's limits"
    )
    # Measured by the job, beside each preprocessed record.
    assert len(list((OUTPUT_DIR / run).glob("records/*/SignalMeasures_0000.json"))) == 2
    cells = overview["cells"]
    assert [(one["key"], one["x"]) for one in cells] == [("1.mseed", -2.0), ("2.mseed", 13.0)]
    # The synthetic wavelet stands far above its noise.
    assert all(one["status"] == "pass" and one["value"] > 6 for one in cells)
    assert overview["track"]["limit"] == 6.0

    card = client.get(f"/quality/records/card/{run}/1.mseed").json()
    assert card["title"] == "1.mseed · shot at -2 m" and card["status"] == "pass"
    ((gate),) = card["gates"]
    assert gate["gate"] == "G1" and gate["verdict"] is None
    names = [metric["name"] for metric in gate["metrics"]]
    # Its spectra measured as a passive record's: every line alike.
    assert names[:5] == [
        "dead_traces",
        "clipped_traces",
        "nan_traces",
        "spectral_outliers",
        "rms_outliers",
    ]
    assert card["windows"] == ["xmid_2.50", "xmid_5.50", "xmid_8.50"]
    # What was done: preprocessed by the user, in PAC.
    assert _done(card)[0] == ("Preprocessed", "you", "active", [])
    # Its measures in their tables, by object; its sentences, what they do not say.
    assert {metric["of"] for metric in gate["metrics"]} == {"signal", "spectrum"}
    assert _texts(card) == [
        "Preprocessed by the user.",
        "Used by the 3 windows centred from 2.5 to 8.5 m.",
    ]
    assert client.get(f"/quality/records/card/{run}/nope.mseed").status_code == 404
    # Its spectra, preprocessed, as the job saved them: each trace's, its usable band drawn.
    spectra = client.get(f"/quality/records/spectrum/{run}/1.mseed").json()
    assert len(spectra["positions"]) == len(spectra["amplitude"]) == 12
    assert spectra["freqs"][0] == 0.0 and spectra["band_hz"] is not None
    assert client.get(f"/quality/records/spectrum/{run}/nope.mseed").status_code == 404
    # A run from before the spectra were saved: computed from the saved record, alike.
    saved = OUTPUT_DIR / run / "records" / "2" / "Spectrum_0000.npz"
    kept = saved.read_bytes()
    stored = client.get(f"/quality/records/spectrum/{run}/2.mseed").json()
    saved.unlink()
    try:
        computed = client.get(f"/quality/records/spectrum/{run}/2.mseed").json()
    finally:
        saved.write_bytes(kept)
    assert computed["freqs"] == stored["freqs"] and computed["band_hz"] == stored["band_hz"]
    assert all(
        abs(a - b) <= 0.005
        for one, other in zip(computed["amplitude"], stored["amplitude"], strict=True)
        for a, b in zip(one, other, strict=True)
    )
    assert client.get(f"/quality/dispersion/spectrum/{run}/2.5").status_code == 404  # active


def test_pac_measures_a_record_as_the_assistants_g1_does(run: str) -> None:
    # One definition (sigpipe's measure_signal): the job's saved measures are G1's, value for
    # value, each saying what it covers, within the line's reach as G1's (line_reach); G1
    # alone judges.
    folder = OUTPUT_DIR / run / "records" / "1"
    saved = json.loads((folder / "SignalMeasures_0000.json").read_text())["metrics"]
    streams = [
        load_stream([path])[0]
        for path in sorted((OUTPUT_DIR / run).glob("records/*/Stream_0000.hdf5"))
    ]
    positions = [receiver.x for receiver in load_profile("shots", workspace()).receivers]
    reach = line_reach(
        ((one, 0.0) for one in streams), SignalLimits(), max(positions) - min(positions)
    )
    dispersion = json.loads((OUTPUT_DIR / run / "run.json").read_text())["preset"]["dispersion"]
    g1 = judge_signal(
        "1.mseed",
        load_stream([folder / "Stream_0000.hdf5"])[0],
        SignalThresholds(),
        reach_m=reach,
        spectra=True,
        image_band=(dispersion["fmin"], dispersion["fmax"]),
    )

    measured = [(one["name"], one["value"], one["passed"], one["of"], one["over"]) for one in saved]
    assert measured == [(m.name, m.value, m.passed, m.of, m.over) for m in g1.metrics]
    snr = next(one for one in saved if one["name"] == "snr_db")
    assert snr["of"] == "signal" and "; in its usable band within the images', " in snr["over"]


def test_a_muted_records_noise_is_measured_before_its_muting(run: str) -> None:
    (path,) = (OUTPUT_DIR / run).glob("records/1/Stream_0000.hdf5")
    stream = load_stream([path])[0]
    thresholds = thresholds_of(OUTPUT_DIR / run)
    # As a muting leaves it: nothing after the slowest arrival, where its noise is measured.
    offsets = np.asarray(stream.acquisition.offsets, dtype=float)
    windows = signal_windows(
        offsets, np.asarray(stream.ts), thresholds.vg_min, thresholds.vg_max, thresholds.pad_s
    )
    assert windows is not None and windows.where.startswith("after the slowest arrival")
    muted = replace(stream, xt=np.where(windows.noise, 0.0, stream.xt))

    def snr(unmuted: tuple[Stream, float] | None = None) -> float | None:
        metrics, _ = signal_metrics(muted, thresholds, active=True, unmuted=unmuted)
        return next(one.value for one in metrics if one.name == "snr_db")

    own, _ = signal_metrics(stream, thresholds, active=True)
    # Its noise zeroed, the SNR reads far over the record's own.
    own_snr = next(one.value for one in own if one.name == "snr_db") or 0.0
    assert (snr() or 0.0) > own_snr + 20
    assert snr((stream, 0.0)) == next(one.value for one in own if one.name == "snr_db")
    # The record before its muting, from its input, for a muted preset only.
    profile = load_profile("shots", workspace())
    windowed = {"masw": {"length": 6, "step": 3}}
    muting = {"muting": {"method": "mute", "vmin": 200.0, "vmax": 1500.0}}
    on = resolve_preset(make_preset("active", windowed | muting), profile)
    found = before_muting(on, profile, "1.mseed")
    assert found is not None and found[1] == 0.0  # no trigger in its file
    assert found[0].xt.shape == stream.xt.shape
    assert (
        before_muting(resolve_preset(make_preset("active", windowed), profile), profile, "1.mseed")
        is None
    )


def test_a_pac_run_shows_its_images_and_picks(run: str) -> None:
    overview = client.get(f"/quality/dispersion/overview/{run}").json()

    assert overview["paco"] is False
    assert overview["summary"] == (
        "3 windows · 2 picked by the user · 2 within the agent's limits · 1 not picked"
    )
    cells = {one["x"]: one for one in overview["cells"]}
    assert cells[8.5]["status"] == "none" and cells[8.5]["value"] is None
    # A curve picked by hand is the user's, as it is; its image against the agent's limits, as
    # every image no check judged: beyond those the agent's check only warns about, still
    # within them, the warning a line of its own, as the agent's own images say it.
    assert cells[2.5]["status"] == "pass" and cells[2.5]["value"] > 0
    image_lines = cells[2.5]["hover"][1:3]
    assert image_lines[0] == "Image: within the agent's limits"
    assert image_lines[1].startswith("Warning: ")
    assert cells[2.5]["hover"][3] == "Curve: picked by hand"
    assert (
        cells[2.5]["hover"][4].startswith("M0: ") and "Curve: not picked" in (cells[8.5]["hover"])
    )
    assert cells[2.5]["hover"][5].startswith("λmax / 2 = ")
    # Its modes, a dot each under its cell; none without a curve.
    assert cells[2.5]["modes"] == ["M0"] and cells[8.5]["modes"] == []
    # Its image and its curve apart, as on every run, as their legends say.
    assert [cells[x]["parts"] for x in (2.5, 8.5)] == [["pass", "hand"], ["pass", "none"]]
    image, curve = overview["parts"]
    assert image["title"] == "Image" and image["legend"] == {"pass": "within the agent's limits"}
    assert curve["title"] == "Curve" and curve["legend"]["hand"] == "picked by hand"
    assert curve["legend"]["none"] == "not picked" and "fail" not in curve["legend"]
    card = client.get(f"/quality/dispersion/card/{run}/2.5").json()
    assert card["picked_by"] == "hand" and [one["label"] for one in card["curves"]] == ["M0"]
    # Who picked it and how: in what was done, not among its settings.
    assert "picking" not in {one["key"] for one in card["settings"]}
    low, high = card["band_hz"]
    assert low < 20 < high
    gates = {gate["gate"]: gate for gate in card["gates"]}
    assert gates["G2"]["verdict"] is None and not gates["G2"]["by_hand"]
    # The curve's check passes the user's curve; the check along the line leaves it out.
    assert [(gates[one]["verdict"], gates[one]["by_hand"]) for one in ("G3", "G4")] == [
        ("pass", True),
        (None, True),
    ]
    # Measured as G3 measures a pick, not judged; nothing of it compared along the line.
    curve = {metric["name"]: metric for metric in gates["G3"]["metrics"]}
    assert {"sharpness", "on_data", "wavelength_ratio"} <= set(curve)
    assert {metric["of"] for metric in curve.values()} == {"curve"}
    assert gates["G4"]["metrics"] == []
    # No check over the line: nothing in the stage's menu.
    assert overview["gates"] == []
    assert {metric["name"] for metric in gates["G2"]["metrics"]} >= {
        "coherent_columns",
        "competing_ridges",
    }
    assert card["status"] == "pass" and card["verdict"] is None
    assert card["parts"] == [
        {"label": "image", "state": "pass", "judged": False},
        {"label": "curve", "state": "hand", "judged": False},
    ]
    assert any(text.startswith("M0 picked by the user: ") for text in _texts(card))
    # Its image's warnings, as the agent's check keeps them: a sentence each.
    warned = [one["text"] for one in card["sentences"] if one["mark"] == "warn"]
    assert warned and all(text.endswith(".") for text in warned)
    # Its curve the user's, taken as it is: no test of it.
    assert _done(card)[2:] == [("M0 picked", "you", "box", [])]

    unpicked = client.get(f"/quality/dispersion/card/{run}/8.5").json()
    assert unpicked["status"] == "none" and "Not picked." in _texts(unpicked)


def test_a_pac_inversion_saves_its_measures_as_the_assistant_does(run: str) -> None:
    # The job saved them where the assistant saves its own; Visualization only reads them.
    path = OUTPUT_DIR / run / "xmid_2.50" / "SeismicInversion_Measures_0000.json"
    measures = InversionMeasures.model_validate_json(path.read_text())
    assert measures.samples_per_chain > 0 and measures.useful_reference == USEFUL_REFERENCE
    saved = path.stat().st_mtime_ns

    overview = client.get(f"/quality/inversion/overview/{run}").json()

    assert overview["paco"] is False
    assert overview["summary"].startswith("3 windows · 2 inverted by the user · ")
    cells = {one["x"]: one for one in overview["cells"]}
    assert cells[8.5]["status"] == "none" and cells[8.5]["hover"][-1] == "Not inverted"
    assert 0 <= cells[2.5]["value"] <= cells[2.5]["total"]
    assert overview["track"]["kind"] == "depth"
    # The window's own settings; both windows ran with the same: no spread.
    card = client.get(f"/quality/inversion/card/{run}/2.5").json()
    settings = {one["key"]: one for one in card["settings"]}
    assert (settings["layers"]["value"], settings["layers"]["detail"]) == ("1", "over a half-space")
    assert settings["depth"]["value"] == "5 m at most"  # the sum of the thickness_max as run
    assert all(one["origin"] == "you" and not one["spread"] for one in settings.values())
    assert path.stat().st_mtime_ns == saved


def test_an_inversion_newer_than_its_measures_shows_none_of_them(run: str) -> None:
    window = OUTPUT_DIR / run / "xmid_5.50"
    path = window / "SeismicInversion_Measures_0000.json"
    saved = path.read_text()
    samples = window / "SeismicInversion_Samples_0000.npz"
    older = samples.stat().st_mtime_ns - 10**9
    os.utime(path, ns=(older, older))

    overview = client.get(f"/quality/inversion/overview/{run}").json()
    card = client.get(f"/quality/inversion/card/{run}/5.5").json()

    cell = next(one for one in overview["cells"] if one["x"] == 5.5)
    assert (cell["status"], cell["hover"][-1]) == (
        "none",
        "Inverted before its measures were saved",
    )
    assert card["inverted"] is False and _texts(card) == [
        "Inverted before its measures were saved with it: invert it again to see them."
    ]
    assert path.read_text() == saved and path.stat().st_mtime_ns == older  # never measured here
    path.write_text(saved)


def test_what_an_older_inversion_lacks_is_neither_shown_nor_made(run: str) -> None:
    window = OUTPUT_DIR / run / "xmid_5.50"
    path = window / "SeismicInversion_Measures_0000.json"
    saved, band = path.read_text(), (window / VS_SPREAD_FILE).read_text()
    # As an older inversion saved its measures: the depth read against a prior, neither the
    # interfaces nor the band of the models' Vs.
    older = json.dumps(
        json.loads(saved) | {"useful_reference": "curve", "useful_depth_m": 0.123, "interfaces": []}
    )
    path.write_text(older)
    (window / VS_SPREAD_FILE).unlink()

    card = client.get(f"/quality/inversion/card/{run}/5.5").json()
    section = client.get(f"/inversion/velocity_section/{run}").json()

    # Visualization only reads: none of them shown, none made.
    profile = card["profile"]
    assert profile["informed"] is None and profile["interfaces"] == []
    assert profile["spread_depths"] == [] and profile["uncertainty"] == []
    assert not any(text.startswith("The data inform") for text in _texts(card))
    column = next(one for one in section["windows"] if one["x"] == 5.5)
    assert column["informed"] is None
    assert not (window / VS_SPREAD_FILE).exists() and path.read_text() == older
    path.write_text(saved)
    (window / VS_SPREAD_FILE).write_text(band)


def test_the_figures_a_run_saved_are_listed_and_served_as_they_are(run: str) -> None:
    names = client.get(f"/quality/run_figures/{run}").json()

    # The jobs' sections, each smoothed along the line too, then their pseudo-section
    # comparisons, each by wavelength too.
    assert {
        "SeismicInversion_VelocitySection_0000.png",
        "SeismicInversion_VelocitySection_0000_lateralsmooth.png",
        "SeismicInversion_PseudoSectionComparison_0000_M0.png",
        "SeismicInversion_PseudoSectionComparison_0000_M0_wavelength.png",
        "PetroInversion_Section_0000.png",
        "PetroInversion_Section_0000_lateralsmooth.png",
        # The line at a glance.
        "SeismicInversion_LineSummary_0000.png",
    } <= set(names)
    sections = names.index("SeismicInversion_VelocitySection_0000.png")
    assert sections < names.index("SeismicInversion_PseudoSectionComparison_0000_M0.png")
    served = client.get(f"/quality/run_figure/{run}/{names[0]}")
    assert served.status_code == 200 and served.headers["content-type"] == "image/png"
    # Nothing else of the run's folder.
    assert client.get(f"/quality/run_figure/{run}/run.json").status_code == 404


def test_a_windows_and_a_records_figures_are_listed_and_served(run: str) -> None:
    manifest = json.loads((OUTPUT_DIR / run / "run.json").read_text())
    record = manifest["records"][0]["name"]

    # A record's preprocessed traces and its spectrum; a window's processing figures, not its
    # inversion's.
    assert client.get(f"/quality/run_figures/{run}?record={record}").json() == [
        "Spectrum_0000.png",
        "Stream_0000.png",
    ]
    served = client.get(f"/quality/run_figure/{run}/Stream_0000.png?record={record}")
    assert served.status_code == 200 and served.headers["content-type"] == "image/png"
    assert client.get(f"/quality/run_figures/{run}?xmid=2.5").json() == [
        "DispersionImage_0000.png",
        "PetroInversion_Window_0000.png",
    ]
    assert client.get(f"/quality/run_figures/{run}?record=nothing.dat").status_code == 404
    # An active window stacks images, not correlations: no gather of them.
    assert client.get(f"/quality/dispersion/gather/{run}/2.5").status_code == 404


def test_every_card_says_what_was_done_and_why(run: str) -> None:
    manifest = json.loads((OUTPUT_DIR / run / "run.json").read_text())
    record = manifest["records"][0]["name"]

    cards = {
        "record": client.get(f"/quality/records/card/{run}/{record}").json(),
        "window": client.get(f"/quality/dispersion/card/{run}/2.5").json(),
        "inversion": client.get(f"/quality/inversion/card/{run}/2.5").json(),
        "petro": client.get(f"/quality/petro/card/{run}/2.5").json(),
    }

    labels = {kind: [one["label"] for one in card["settings"]] for kind, card in cards.items()}
    # In the run card's words; a step every run does (the detrend), or off by default and left
    # off (the mute, the filter, the trigger of a record not muted), not said.
    assert labels["record"] == []
    # A window's: its receivers, its records' preprocessing, then how its image was made; who
    # picked its curves and how, in what was done.
    assert labels["window"] == ["Window", "Dispersion image", "Image stacking"]
    assert labels["inversion"] == [
        "Layers",
        "Vs bounds",
        "Thinnest layer",
        "Deepest interface",
        "Vs drop",
        "Sampling",
    ]
    assert labels["petro"] == ["Silex model"]
    # PAC's own run, its form's settings or the preset's defaults: none a check changed.
    origins = {one["origin"] for card in cards.values() for one in card["settings"]}
    assert origins <= {"you", "default"}


def test_an_inversion_card_shows_the_model_its_fit_and_its_chains(run: str) -> None:
    card = client.get(f"/quality/inversion/card/{run}/2.5").json()

    assert card["inverted"] is True and card["title"] == "xmid 2.5 m · 1 layer over a half-space"
    assert card["parameters"]["n_layers"] == 2
    assert _done(card)[0] == ("Inverted", "you", "MCMC", [])
    # No trial runs: the chains' moves follow the posterior.
    assert card["tuning"] == [] and card["step_factor"] is None
    assert len(card["acceptance"]) == PARAMETERS["n_chains"]
    # The layers given: no moves of a layer count, nor tempered copies.
    assert card["moves"] == {} and card["move_steps"] == {} and card["exchanges"] is None
    (gate,) = card["gates"]
    assert gate["gate"] == "G5" and gate["verdict"] is None
    # The acceptance among the measures, the chains' median: reported, the layers given.
    acceptance = [metric for metric in gate["metrics"] if metric["name"] == "acceptance"]
    assert [(one["threshold"], one["unit"]) for one in acceptance] == [(None, "%")]
    profile = card["profile"]
    # The ensemble by default: each depth's median of the kept models; its fit named.
    assert profile["model"] == "ensemble" and profile["tops"][0] == 0
    assert any(
        text.startswith(("The median of the ensemble fits", "The median of the ensemble misfits"))
        for text in _texts(card)
    )
    assert len(profile["tops"]) == len(profile["vs"]) <= 401
    # The kept models' 10th and 90th percentiles at each depth (sigpipe's band, every 5 cm), as
    # the curve's band.
    depths = profile["spread_depths"]
    assert 0 < len(depths) == len(profile["spread_low"]) == len(profile["spread_high"]) <= 400
    assert depths[0] == 0.025 and depths[-1] < profile["bottom"]
    assert all(
        low <= high for low, high in zip(profile["spread_low"], profile["spread_high"], strict=True)
    )
    # Their relative uncertainty U at the same depths, %.
    assert len(profile["uncertainty"]) == len(depths)
    assert all(one >= 0 for one in profile["uncertainty"])
    # Where they place interfaces, % of them per 0.5 m from the surface down.
    assert profile["interface_dz"] == 0.5 and profile["interfaces"]
    assert all(0 <= one <= 100 for one in profile["interfaces"])
    assert profile["deepest_top"] == 5.0 and profile["bottom"] > 0
    curve = card["curve"]
    assert curve["label"] == "M0" and len(curve["observed_fs"]) == len(curve["observed_vs"]) > 0
    assert len(curve["predicted_fs"]) > 0
    # The kept models' spread around it, as the saved density figure draws it: their 10th and
    # 90th percentiles at the picked frequencies.
    assert 0 < len(curve["spread_fs"]) == len(curve["spread_low"]) == len(curve["spread_high"])
    assert all(
        low <= high for low, high in zip(curve["spread_low"], curve["spread_high"], strict=True)
    )
    rows = [row["parameter"] for row in card["convergence"]]
    # Vs at the depths the curve resolves first, where the chains are judged; then the layers'
    # own values, the noise factor.
    watched = [name for name in rows if name.startswith("vs@")]
    assert watched and rows == [*watched, "vs1", "vs2", "thick1", "noise"]
    vs1 = card["convergence"][rows.index("vs1")]
    # The resulting model beside its prior: the posterior's median within its middle 80 %.
    assert vs1["prior"] == [100, 400] and 100 <= vs1["low"] <= vs1["median"] <= vs1["high"] <= 400
    assert vs1["step"] > 0 and vs1["step_unit"] == ""  # m/s, as the layer's own
    assert card["figures"] == ["marginals", "density_curves", "dispersion_image", "chains"]
    # The median of the ensemble alone said: no layered median.
    assert not any(text.startswith("Layered median") for text in _texts(card))
    best = client.get(f"/quality/inversion/card/{run}/2.5?model=best").json()
    assert best["profile"]["model"] == "best"
    assert client.get(f"/quality/inversion/card/{run}/2.5?model=nope").status_code == 422
    unmodelled = client.get(f"/quality/inversion/card/{run}/8.5").json()
    assert unmodelled["inverted"] is False and _texts(unmodelled) == ["Not inverted."]

    chains = client.get(f"/quality/inversion/chains/{run}/2.5").json()
    assert [one["parameter"] for one in chains["traces"]] == rows
    per_chain = card["samples_per_chain"]
    for traces in chains["traces"]:
        assert len(traces["chains"]) == PARAMETERS["n_chains"]
        assert all(len(chain) == -(-per_chain // traces["step"]) for chain in traces["chains"])
        assert all(len(chain) <= 400 for chain in traces["chains"])
    vs1 = next(one for one in chains["marginals"] if one["parameter"] == "vs1")
    assert (vs1["low"], vs1["high"]) == (100, 400)
    assert all(len(counts) == 40 for counts in vs1["counts"])
    assert sum(map(sum, vs1["counts"])) == per_chain * PARAMETERS["n_chains"]
    assert client.get(f"/quality/inversion/chains/{run}/8.5").status_code == 404

    figure = client.get(f"/quality/inversion/figure/{run}/2.5/marginals")
    assert figure.status_code == 200 and figure.headers["content-type"] == "image/png"
    assert client.get(f"/quality/inversion/figure/{run}/2.5/stream").status_code == 422


def test_a_pac_petrophysical_inversion_shows_its_soil_column_and_fit(run: str) -> None:
    overview = client.get(f"/quality/petro/overview/{run}").json()

    (model,) = client.get("/petro_inversion/models").json()
    assert overview["paco"] is False and overview["summary"].startswith(
        "3 windows · 2 inverted by the user · soil model "
    )
    cells = {one["x"]: one for one in overview["cells"]}
    assert cells[8.5]["status"] == "none" and cells[2.5]["value"] > 0

    card = client.get(f"/quality/petro/card/{run}/2.5").json()
    (setting,) = card["settings"]
    assert setting["value"] == model and setting["detail"].startswith("trained on ")
    assert setting["origin"] == "you" and not setting["spread"]
    assert card["model"] == model and card["fit"]["model"] == "petro"
    column = card["column"]
    assert len(column["soils"]) == len(column["thicknesses_m"]) == len(column["ns"])
    assert column["water_table_m"] > 0
    assert card["curve"]["label"] == "M0" and len(card["curve"]["predicted_fs"]) > 0
    assert card["gates"][0]["gate"] == "G7" and card["gates"][0]["verdict"] is None
    assert any(text.startswith("Soil column, top down: ") for text in _texts(card))


def test_a_measure_rejects_or_flags_only_when_the_check_acted_on_it() -> None:
    # Left out for its SNR; its trigger off too, kept as information: the SNR rejected the
    # record, the trigger only flags it.
    record = GateResult.model_validate(
        _result(
            "G1",
            "1.mseed",
            "reject",
            [_metric("snr_db", 2.6, 6.0, "min"), _metric("trigger_error_s", -0.017, 0.01, "max")],
            [
                _flag("low_snr", "Median SNR 2.6 dB", "preprocessing", {"kind": "exclude_record"}),
                _flag("shifted_trigger", "Trigger off", "preprocessing", {"kind": "keep"}),
            ],
        )
    )
    assert acted_on(record, "reject") == ("snr_db",)
    # A retry refused once the budget was spent: the measure of the flag that asked it.
    curve = GateResult.model_validate(
        _result(
            "G3",
            "xmid_2.50",
            "reject",
            [_metric("on_data", 0.5, 0.6, "min"), _metric("near_offset", 0.75, 28.5, "min")],
            [
                _flag("on_data", "Only 50 %", "picking", {"kind": "override"}),
                _flag("near_field", "The nearest shot", "phase_shift", {"kind": "keep"}),
                _flag("budget_spent", "No retry left", "picking", {"kind": "reject"}),
            ],
        )
    )
    assert acted_on(curve, "reject") == ("on_data",)
    # Flagged: a retry asked and not made, the measure it judged; nothing rejected it.
    flagged = curve.model_copy(update={"verdict": "retry"})
    assert acted_on(flagged, "retry") == ("on_data",) and acted_on(flagged, "reject") == ()
    # Passed, only warned: no measure rejected or flagged it.
    passed = record.model_copy(update={"verdict": "pass"})
    assert acted_on(passed, "reject") == acted_on(passed, "retry") == acted_on(None, "reject") == ()


def test_a_unit_made_by_hand_is_beyond_the_agents_limits_where_its_check_would_act() -> None:
    # The coherent band at the image's top, a second ridge: the agent's check only warns.
    warned = [
        Metric.model_validate(_metric("band_at_fmax", 1.0, 0.0, "max")),
        Metric.model_validate(_metric("competing_ridges", 0.75, 0.7, "max")),
    ]
    assert measured_status(warned) == "pass"
    assert state_lines("pass", metrics=warned) == [
        "Within the agent's limits",
        "Warning: coherent band at the highest frequency, second ridge",
    ]
    # A ridge at the velocity range's top: the check would flag it.
    asked = Metric.model_validate(_metric("ridge_at_vmax", 0.5, 0.2, "max"))
    assert measured_status([*warned, asked]) == "note"


def test_a_unit_refused_its_retry_stays_flagged_its_reason_a_warning() -> None:
    # PACo's retries spent: the curve still flagged, goes on; why no retry, a note first.
    curve = GateResult.model_validate(
        _result(
            "G3",
            "xmid_2.50",
            "retry",
            [_metric("on_data", 0.5, 0.6, "min")],
            [
                _flag(
                    "budget_spent",
                    "The retry budget is spent; the last attempt raised on_data.",
                    "picking",
                    {"kind": "keep", "note": "budget spent"},
                ),
                _flag("on_data", "Only 50 % on the ridge", "picking", {"kind": "override"}),
            ],
        )
    )

    assert verdict_sentence(curve) == Sentence(
        mark="flag", text="Flagged by the curve check: points off the ridge's top."
    )
    assert [one.text for one in warnings(curve)] == ["The retry budget is spent."]
    assert state_lines("warn", curve) == [
        "Flagged: points off the ridge's top",
        "Warning: retries spent",
    ]


def test_an_unknown_folder_is_not_found() -> None:
    for path in (
        "run/nope",
        "sources/nope/2.5",
        "records/overview/nope",
        "dispersion/overview/nope",
        "dispersion/card/nope/2.5",
        "inversion/overview/nope",
        "inversion/card/nope/2.5",
        "petro/overview/nope",
        "petro/card/nope/2.5",
        "inversion/overview/..%2F..%2Fetc",
    ):
        assert client.get(f"/quality/{path}").status_code == 404, path


# The same run as the assistant judged it: its QC log, configuration and window length trials,
# as PACo writes them, stamped after PAC made the run's files.


def _metric(
    name: str, value: float | None, threshold: float | None, bound: str | None
) -> dict[str, Any]:
    passed = (
        threshold is None
        or value is None
        or (value <= threshold if bound == "max" else value >= threshold)
    )
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "bound": bound,
        "passed": passed,
        "unit": "",
    }


def _flag(name: str, message: str, stage: str, action: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "message": message, "stage": stage, "action": action, "fixable": True}


def _result(
    gate: str, unit: str, verdict: str, metrics: list[Any], flags: list[Any]
) -> dict[str, Any]:
    kept = {"band_hz": [8.0, 55.0], "wavelength_m": None, "n_points": None, "n_traces": 12}
    return {
        "gate": gate,
        "unit": unit,
        "verdict": verdict,
        "metrics": metrics,
        "flags": flags,
        "kept": kept,
        "shrank": False,
    }


def _attempt(
    unit: str,
    stage: str,
    attempt: int,
    triggered_by: str,
    results: dict[str, Any],
    parameters: dict[str, Any] | None = None,
    notes: tuple[str, ...] = (),
    at: str = "",
) -> str:
    return json.dumps(
        {
            "unit": unit,
            "stage": stage,
            "attempt": attempt,
            "parameters": parameters or {},
            "triggered_by": triggered_by,
            "started_at": at,
            "finished_at": at,
            "status": "succeeded",
            "error": None,
            "notes": list(notes),
            "results": results,
        }
    )


@pytest.fixture(scope="module")
def judged(run: str) -> str:
    """The run copied, with the QC log PACo would have written for it."""
    folder = f"{run}-judged"
    target = OUTPUT_DIR / folder
    shutil.copytree(OUTPUT_DIR / run, target)
    now = datetime.now(UTC).isoformat()
    shifted = _flag(
        "shifted_trigger",
        "The first breaks put the trigger at 12 ms, the same on every trace.",
        "preprocessing",
        {"kind": "override", "stage": "preprocessing", "overrides": {"trigger": {"t0": 0.012}}},
    )
    g1_retry = _result("G1", "1.mseed", "retry", [_metric("snr_db", 9.0, 6.0, "min")], [shifted])
    g1_pass = _result("G1", "1.mseed", "pass", [_metric("snr_db", 9.5, 6.0, "min")], [])
    g2 = _result("G2", "xmid_2.50", "pass", [_metric("coherent_columns", 0.8, 0.5, "min")], [])
    g3 = _result(
        "G3",
        "xmid_2.50",
        "pass",
        [_metric("sharpness", 1.1, 0.8, "min"), _metric("prominence", 0.9, 0.5, "min")],
        [],
    )
    g4 = _result("G4", "xmid_2.50", "pass", [_metric("misfit", 0.05, 0.15, "max")], [])
    gaps = _flag(
        "gaps",
        "No curve at xmid 8.50 (1): the section has gaps there.",
        "picking",
        {"kind": "keep", "note": "the gaps stay in the report"},
    )
    g4_line = _result("G4", "line", "pass", [_metric("curves", 2, 1, "min")], [gaps])
    converged = _flag(
        "not_converged",
        "The chains do not agree: sample longer, 2000 iterations.",
        "inversion",
        {"kind": "override", "stage": "inversion", "overrides": {"n_iterations": 2000}},
    )
    g5_retry = _result("G5", "xmid_2.50", "retry", [_metric("rhat", 1.3, 1.1, "max")], [converged])
    g5_pass = _result("G5", "xmid_2.50", "pass", [_metric("rhat", 1.02, 1.1, "max")], [])
    g6 = _result("G6", "xmid_2.50", "pass", [_metric("misfit", 0.05, 0.15, "max")], [])
    g6_line = _result("G6", "line", "pass", [_metric("models", 2, 1, "min")], [])
    model = client.get("/petro_inversion/models").json()[0]
    g7 = _result("G7", "xmid_2.50", "pass", [_metric("misfit_short", 0.8, 2.0, "max")], [])
    g8 = _result("G8", "xmid_2.50", "pass", [_metric("water_table_jump", 0.5, 1.0, "max")], [])
    # The window inverted as the log says: its parameters saved as the workflow's last.
    saved_path = target / "xmid_2.50" / "SeismicInversion_Parameters_0000.json"
    saved = json.loads(saved_path.read_text())
    first = saved["parameters"]
    longer = {**first, "n_iterations": 2000, "n_burnin_iterations": 200}
    saved_path.write_text(json.dumps({**saved, "parameters": longer}))
    lines = [
        _attempt("1.mseed", "preprocessing", 1, "initial", {}, at=now),
        _attempt("1.mseed", "preprocessing", 1, "initial", {"G1": g1_retry}, at=now),
        _attempt(
            "1.mseed",
            "preprocessing",
            2,
            "G1:shifted_trigger",
            {"G1": g1_pass},
            {"trigger": {"t0": 0.012}},
            at=now,
        ),
        _attempt(
            "line",
            "phase_shift",
            1,
            "initial",
            {},
            {"masw": {"length": 6}},
            ("masw length 6 for the whole line: trial windows G3 passed: 2/2 at 6.",),
            at=now,
        ),
        _attempt("xmid_2.50", "phase_shift", 1, "initial", {"G2": g2}, at=now),
        _attempt(
            "xmid_2.50", "picking", 1, "initial", {"G3": g3, "G4": g4}, {"threshold": 0.35}, at=now
        ),
        _attempt("line", "picking", 1, "initial", {"G4": g4_line}, at=now),
        _attempt("xmid_2.50", "inversion", 1, "initial", {"G5": g5_retry}, first, at=now),
        _attempt(
            "xmid_2.50",
            "inversion",
            2,
            "G5:not_converged",
            {"G5": g5_pass, "G6": g6},
            longer,
            at=now,
        ),
        _attempt("line", "inversion", 1, "initial", {"G6": g6_line}, at=now),
        _attempt(
            "xmid_2.50",
            "petro_inversion",
            1,
            "initial",
            {"G7": g7, "G8": g8},
            {"model": model},
            at=now,
        ),
    ]
    # The last line cut short, as PACo may be writing it.
    (target / "qc_log.jsonl").write_text("\n".join(lines) + '\n{"unit": "xmid_5.50", "sta')
    (target / "qc_config.json").write_text(json.dumps({"model": {"max_rhat": 1.2, "min_ess": 300}}))
    trial = {
        "length": 6,
        "metres": 5.0,
        "xmids": [2.5, 5.5],
        "verdicts": ["pass", "pass"],
        "flags": [],
        "passed": 2,
        "windows": 3,
        "wavelengths_m": [2.0, 12.0],
        "compared": False,
    }
    choice = {"length": 6, "trials": [trial], "notes": [], "receivers": 12, "spacing_m": 1.0}
    (target / "coherence.json").write_text(json.dumps({**choice, "longest": 6}))
    return folder


def test_an_assistant_run_card_says_why_its_windows_are_what_they_are(judged: str) -> None:
    card = client.get(f"/quality/run/{judged}").json()

    assert card["made_by"] == "workflow"
    settings = {one["key"]: one for one in card["settings"]}
    windows = settings["windows"]
    # The length trial's choice in a line, its trials in their table; one every 3 receivers, not
    # the preset's 1: the request the assistant writes, the user's value or its own.
    assert windows["origin"] == "workflow"
    assert windows["why"] == (
        "the length trial kept it: 2 of its 2 test windows passed (1 length tried); one every 3 "
        "receivers (3 m): given in the request: by the user, or chosen by the workflow agent"
    )
    (lengths,) = card["tried"]
    assert (lengths["title"], lengths["by"]) == ("Window lengths tried", "workflow")
    rows = [(row["name"], row["clear"], row["depth"], row["kept"]) for row in lengths["rows"]]
    assert rows == [("Option 1", "2 of 2", "6 m", True)]
    assert (lengths["columns"], lengths["rows"][0]["values"]) == (
        ["Windows"],
        ["6 receivers (5 m)"],
    )
    assert settings["band"]["origin"] == "default"
    # A record the checks preprocessed again, said with the records, theirs.
    records = settings["records"]
    assert records["origin"] == "checks"
    assert records["why"].endswith("preprocessed again: 1 record (1 trigger off)")


def test_an_assistant_run_card_lists_what_its_line_tried(judged: str) -> None:
    target = OUTPUT_DIR / judged
    standard = {"method": "mute", "vmin": 80.0, "vmax": 1500.0, "width": 0.05}
    mutes = {
        "chosen": "none",
        "trigger": {"t0": 0.0016},
        "trials": [
            {"candidate": "none", "muting": None, "verdicts": ["pass", "pass"], "passed": 2},
            {
                "candidate": "standard",
                "muting": standard,
                "verdicts": ["pass", "reject"],
                "passed": 1,
            },
        ],
    }
    (target / "mute.json").write_text(json.dumps(mutes))
    tried = {
        "stage": "preprocessing",
        "overrides": {"muting": standard},
        "change": "muting mute 80 to 1500 m/s, width 0.05 s",
        "flag": "G2:weak_coherence",
        "asked_by": 3,
        "xmids": [2.5, 5.5],
        "before": 0,
        "after": 2,
        "kept": True,
        "note": "muting mute 80 to 1500 m/s, width 0.05 s: the line loop, ...",
    }
    # A try an older line_loop.json keeps without its change: the note says it.
    older = {
        **{key: value for key, value in tried.items() if key not in ("change", "overrides")},
        "kept": False,
        "note": "Tried for the line, not kept: trigger t0 0.003 s, asked by 90 windows "
        "(G1:shifted_trigger): 2 of 2 trial windows passing G3 against 2, under 2 more.",
    }
    (target / "line_loop.json").write_text(json.dumps({"trials": [tried, older]}))
    try:
        card = client.get(f"/quality/run/{judged}").json()
    finally:
        (target / "mute.json").unlink()
        (target / "line_loop.json").unlink()

    # Each trial of the line in its table: what was tried, how many trial windows passed, kept.
    tables = {table["title"]: table for table in card["tried"]}
    assert list(tables) == [
        "Window lengths tried",
        "Mutes tried",
        "Changes tried where curves failed",
    ]
    # As every comparison: the options side by side, the one kept first, each setting a row.
    mutes = tables["Mutes tried"]
    # Each mute with the trigger the trial set first.
    assert mutes["intro"] == [
        "Each on the same 2 sample windows, with the trigger delay 1.6 ms, where the shots' first "
        "breaks put them"
    ]
    assert mutes["columns"] == ["Mute"]
    assert [(one["name"], one["values"], one["clear"], one["kept"]) for one in mutes["rows"]] == [
        ("Option 1", ["none"], "2 of 2", True),
        ("Option 2", [f"80{DASH}1500 m/s · standard mute · signal width 0.05 s"], "1 of 2", False),
    ]
    # No dispersion images of the workflow's options: its comparisons are their tables.
    assert [(one["trial"], one["images"]) for one in mutes["rows"]] == [("", False), ("", False)]
    changes = tables["Changes tried where curves failed"]
    assert changes["columns"] == ["Mute", "Change", "Asked by"]
    line, older = changes["rows"]
    assert (line["name"], line["values"], line["clear"], line["kept"]) == (
        "Option 1",
        [f"80{DASH}1500 m/s · signal width 0.05 s", "-", "3 windows"],
        "0 → 2 of 2",
        True,
    )
    assert (older["name"], older["values"][1], older["kept"]) == (
        "Option 2",
        "trigger t0 0.003 s",
        False,
    )
    # And in what was done, with the length trial: each a step, its facts a line each.
    steps = {step["kind"]: step for step in card["history"]}
    # Each by the workflow, by trials on sample windows: a badge each.
    assert {
        kind: (steps[kind]["by"], _how(steps[kind])) for kind in ("length", "mute", "line")
    } == dict.fromkeys(("length", "mute", "line"), ("workflow", "trials"))
    assert (steps["length"]["title"], steps["length"]["lines"]) == (
        "Chose the window length",
        ["1 length tried on 2 sample windows", "Kept 6 receivers (5 m)"],
    )
    assert (steps["mute"]["title"], steps["mute"]["lines"]) == (
        "Chose the mute",
        [
            "Trigger delay 1.6 ms, where the shots' first breaks put them",
            "2 mutes tried",
            "Kept: no mute",
        ],
    )
    assert steps["line"]["lines"] == ["2 changes tried", f"Kept: mute 80{DASH}1500 m/s"]


def test_a_workflow_run_says_what_the_user_asked_it(judged: str) -> None:
    folder = f"{judged}-asked"
    shutil.copytree(OUTPUT_DIR / judged, OUTPUT_DIR / folder)
    process = {"asked": ["process"], "mode": "active"}
    calls = [
        {"at": "2026-10-24T01:02:40+00:00", "conversation": "c1", "turn": 2, "scope": process,
         "tool": "run_processing", "seconds": 300},
        {"at": "2026-10-24T01:03:08+00:00", "conversation": "c1", "turn": 3,
         "scope": {"asked": ["pick"]}, "tool": "pick", "seconds": 8},
        {"at": "2026-10-24T01:03:10+00:00", "conversation": "c1", "turn": 3,
         "scope": {"asked": ["pick"]}, "tool": "judge", "seconds": 1},
        {"at": "2026-10-24T01:04:00+00:00", "conversation": "paco-call", "turn": None,
         "scope": None, "tool": "pick", "seconds": 1},
    ]  # fmt: skip
    _lines(OUTPUT_DIR / folder / "agent_calls.jsonl", calls)

    history = client.get(f"/quality/run/{folder}").json()["history"]

    # Each turn of the chat that called a tool on the run, as PACo read the message, before the
    # stage it started; a call outside the chat left out.
    said = [(one["title"], one["by"], _how(one), one["lines"]) for one in history]
    assert [one for one in said if one[0] == "Asked"] == [
        ("Asked", "you", "chat", ["To process the records", "Method: active"]),
        ("Asked", "you", "chat", ["To pick the curves"]),
    ]
    titles = [one[0] for one in said]
    assert titles[0] == "Asked" and titles[titles.index("Picked the curves") - 1] == "Asked"


def test_the_agents_survey_of_the_records_is_measured(run: str) -> None:
    from masw.io.quality.journal import SurveyDone
    from masw.io.quality.runs import _survey_step  # pyright: ignore[reportPrivateUsage]

    manifest = read_manifest(OUTPUT_DIR / run)
    assert manifest is not None
    survey = SurveyDone(receivers=12, shots=(1, 2), usable_spread_hz=(5.0, 60.0))

    step = _survey_step(survey, (), manifest, None)

    # Its raw records measured before any setting is chosen: no method of the run's yet.
    assert (step.title, step.by, step.how.label if step.how else None) == (
        "Looked at the records",
        "agent",
        "measured",
    )
    assert step.lines[:2] == ("2 shots on 12 receivers", "Signal usable from 5 to 60 Hz")


def test_the_searchs_one_value_is_no_word_of_the_agents() -> None:
    from masw.io.quality.runs import SEARCH_SAME, _reasons  # pyright: ignore[reportPrivateUsage]

    commit = JournalEntry(
        id="L1",
        kind="commit",
        at=datetime.now(UTC),
        text="",
        data={
            "changes": {"masw": {"length": 15}, "whitening": {"method": "onebit_apod"}},
            "reasons": {"masw length": "the cleanest ridges", "whitening method": SEARCH_SAME},
        },
    )

    said = _reasons([commit], [], 1.5, 15)

    # Its why on the setting, as code says it; the agent's own words for what it chose alone.
    assert [(one.setting, one.text, one.by) for one in said] == [
        ("Windows", "The cleanest ridges", "agent")
    ]


def test_a_stage_with_nothing_made_says_no_one() -> None:
    from masw.io.quality.runs import StageCount, _profiles  # pyright: ignore[reportPrivateUsage]

    stages = [StageCount(key="inversion", done=0, total=4)]

    profiles = _profiles(stages, (), None, "autonomous")

    assert (profiles.value, profiles.origin) == ("none yet", None)


def test_a_line_loop_that_kept_no_change_says_so(tmp_path: Path) -> None:
    from masw.io.quality.runs import _line_tries  # pyright: ignore[reportPrivateUsage]

    tried = {
        "change": "selection threshold 0.15",
        "flag": "G2:few_segments_kept",
        "asked_by": 24,
        "before": 12,
        "after": 9,
        "kept": False,
    }
    (tmp_path / "line_loop.json").write_text(json.dumps({"trials": [tried]}))

    (step,) = _line_tries(tmp_path, 1.5)
    assert step.lines == ("1 change tried", "None kept")


def test_a_record_made_again_by_a_change_of_the_lines_settings_is_the_lines(run: str) -> None:
    manifest = read_manifest(OUTPUT_DIR / run)
    assert manifest is not None
    muting = {**MUTE, "width": 0.05}
    # The line loop's mute: on the line's attempt, and as what each record ran with again.
    log = QCLog(
        [
            _retried(LINE, "phase_shift", "initial", {"muting": muting}),
            _retried("1.mseed", "preprocessing", "line change", {"muting": muting}),
        ]
    )

    setting = next(one for one in record_settings(manifest, log, "1.mseed") if one.key == "muting")

    # The line's: who set it, the run's card saying why.
    assert (setting.origin, setting.why) == ("workflow", "")


def test_an_assistant_run_shows_g1_and_what_it_redid(judged: str) -> None:
    overview = client.get(f"/quality/records/overview/{judged}").json()

    assert overview["paco"] is True
    first, second = overview["cells"]
    # The attempt's last line is its current state: the second attempt passed.
    assert (first["status"], first["value"]) == ("pass", 9.5)
    assert second["status"] == "pass"  # measured: G1 said nothing of it

    card = client.get(f"/quality/records/card/{judged}/1.mseed").json()
    assert card["verdict"]["text"] == "Passed the signal check."
    # What was done: the run that stands, asked by the check, what it changed; then the check's
    # verdict on it, an entry of its own. The first run, which it replaced, left out.
    assert _done(card) == [
        (
            "Preprocessed",
            "workflow",
            "active",
            ["Asked by the signal check: trigger off", "Trigger: delay 12 ms"],
        ),
        ("Signal check: passed", "checks", "judged", []),
    ]
    # Not muted: its trigger left as recorded, whatever the check asked.
    trigger = next(one for one in card["settings"] if one["key"] == "trigger")
    assert (trigger["value"], trigger["origin"]) == ("none", "checks")
    assert trigger["why"] == "changed for this record, asked by the signal check: trigger off"


def test_an_assistant_run_shows_g2_g3_g4(judged: str) -> None:
    overview = client.get(f"/quality/dispersion/overview/{judged}").json()

    assert overview["paco"] is True
    cells = {one["x"]: one for one in overview["cells"]}
    # Its hover says its image and its curve apart, each with the gates that judged it.
    assert cells[2.5]["status"] == "pass"
    assert cells[2.5]["hover"][1:3] == ["Image: passed", "Curve: passed"]
    # Its image and its curve apart; a window without a curve says none, whatever its image.
    assert cells[2.5]["parts"] == ["pass", "pass"]
    assert cells[8.5]["parts"][1] == "none" and cells[8.5]["status"] == "none"
    assert [part["title"] for part in overview["parts"]] == ["Image", "Curve"]

    card = client.get(f"/quality/dispersion/card/{judged}/2.5").json()
    assert "picking" not in {one["key"] for one in card["settings"]}
    assert card["picked_by"] == "auto" and card["band_hz"] == [8.0, 55.0]  # G2's, as it judged it
    gates = {gate["gate"]: gate for gate in card["gates"]}
    assert [gates[gate]["verdict"] for gate in ("G2", "G3", "G4")] == ["pass"] * 3
    assert [metric["name"] for metric in gates["G3"]["metrics"]] == ["sharpness", "prominence"]
    assert card["verdict"]["text"] == (
        "Passed the dispersion image check, the curve check and the check against the "
        "neighbouring curves."
    )
    assert card["parts"] == [
        {"label": "image", "state": "pass", "judged": True},
        {"label": "curve", "state": "pass", "judged": True},
    ]
    # What was done, its own stages alone: the image the workflow made, checked by G2; the curve
    # it picked again over the user's box, how, with what its settings changed on the run's
    # picking settings, checked by G3 and G4. Its Vs profile and soil column on their cards.
    assert _done(card) == [
        ("Dispersion image made", "workflow", "active", []),
        ("Dispersion image check: passed", "checks", "judged", []),
        ("M0 picked", "workflow", "auto", ["Threshold: 0.35"]),
        ("Curve check: passed", "checks", "judged", []),
        ("Check against the neighbouring curves: passed", "checks", "judged", []),
    ]


def test_a_window_the_agent_picked_again_says_each_pick(judged: str) -> None:
    folder = f"{judged}-repicked"
    shutil.copytree(OUTPUT_DIR / judged, OUTPUT_DIR / folder)
    # The agent's pick again: its stage done afresh (a reset), the new pick attempt 1 again.
    later = (datetime.now(UTC) + timedelta(seconds=5)).isoformat()
    g3 = _result("G3", "xmid_2.50", "pass", [_metric("sharpness", 1.2, 0.8, "min")], [])
    reset = {"event": "reset", "unit": "xmid_2.50", "stages": ["picking"], "at": later}
    again = _attempt(
        "xmid_2.50",
        "picking",
        1,
        "backtrack",
        {"G3": g3},
        {"threshold": 0.35, "wavelength_step": 0.6},
        at=later,
    )
    with (OUTPUT_DIR / folder / "qc_log.jsonl").open("a") as log:
        log.write(f"{json.dumps(reset)}\n{again}\n")

    card = client.get(f"/quality/dispersion/card/{folder}/2.5").json()

    # The pick that stands: what made it, what it changed on the pick it replaced (the agent's
    # own redo, said by its tag), what the check said; the picks before it left out.
    assert _done(card)[2:] == [
        ("M0 picked", "workflow", "auto", ["Resampling: every 0.6 m of wavelength"]),
        ("Curve check: passed", "checks", "judged", []),
    ]


def test_each_stage_shows_the_check_over_its_line(judged: str) -> None:
    # G4 checked this line's curves and G6 its models; G1 not its receivers, G8 not its soil
    # columns. Each stage holds its own, in the menu a unit's measures are in: from a log
    # before the measures said what they describe, the line's.
    gates = {
        stage: client.get(f"/quality/{stage}/overview/{judged}").json()["gates"]
        for stage in ("records", "dispersion", "inversion", "petro")
    }

    assert gates["records"] == [] and gates["petro"] == []
    ((curves,),) = [gate["metrics"] for gate in gates["dispersion"]]
    assert [(gate["gate"], gate["verdict"]) for gate in gates["dispersion"]] == [("G4", "pass")]
    assert (curves["name"], curves["of"]) == ("curves", "line")
    ((models,),) = [gate["metrics"] for gate in gates["inversion"]]
    assert gates["inversion"][0]["gate"] == "G6" and models["of"] == "line"


def test_a_pick_changed_by_hand_leaves_the_assistants_curve_checks_behind(
    judged: str,
) -> None:
    folder = f"{judged}-edited"
    shutil.copytree(OUTPUT_DIR / judged, OUTPUT_DIR / folder)
    later = datetime.now(UTC).replace(year=datetime.now(UTC).year + 1)
    edits = OUTPUT_DIR / folder / "xmid_2.50" / "picks_edited.json"
    edits.write_text(json.dumps({"edited_at": later.isoformat()}))

    card = client.get(f"/quality/dispersion/card/{folder}/2.5").json()

    # The image's check stands; the curve's checks and attempts were of another pick: the pick
    # is the user's, the verdict the image check's alone.
    assert card["picked_by"] == "hand"
    assert card["verdict"]["text"] == "Passed the dispersion image check."
    assert any(text.startswith("M0 picked by the user: ") for text in _texts(card))
    gates = {gate["gate"]: (gate["verdict"], gate["by_hand"]) for gate in card["gates"]}
    assert gates == {"G2": ("pass", False), "G3": ("pass", True), "G4": (None, True)}
    # What was done: the user's curve now, by hand, taken as it is (neither checked nor
    # measured); the workflow's pick it replaced and its checks left out.
    assert _done(card) == [
        ("Dispersion image made", "workflow", "active", []),
        ("Dispersion image check: passed", "checks", "judged", []),
        ("M0 picked", "you", "by hand", []),
    ]
    cells = {
        one["key"]: one
        for one in client.get(f"/quality/dispersion/overview/{folder}").json()["cells"]
    }
    assert "G3" not in " ".join(cells["xmid_2.50"]["hover"])
    assert "Curve: picked by hand" in cells["xmid_2.50"]["hover"]
    # Its image as the assistant checked it, its curve the user's.
    assert cells["xmid_2.50"]["parts"] == ["pass", "hand"]


def test_a_mode_picked_in_pac_keeps_the_assistants_history_of_the_window(judged: str) -> None:
    folder = f"{judged}-m1"
    shutil.copytree(OUTPUT_DIR / judged, OUTPUT_DIR / folder)
    log = OUTPUT_DIR / folder / "qc_log.jsonl"
    box = {"fmin": 10, "fmax": 60, "vmin": 300, "vmax": 800, "label": "M1"}

    client.post(f"/dispersion_images/{folder}/2.5/pick/box", json=box)

    # M1 is the user's; M0, as the assistant picked it, keeps its checks and its attempts.
    card = client.get(f"/quality/dispersion/card/{folder}/2.5").json()
    assert card["picked_by"] == "auto"
    gates = {gate["gate"]: (gate["verdict"], gate["by_hand"]) for gate in card["gates"]}
    assert gates == {"G2": ("pass", False), "G3": ("pass", False), "G4": ("pass", False)}
    # Each curve its own entry: the workflow's M0 with its checks, the user's M1 after it.
    assert _done(card)[2:] == [
        ("M0 picked", "workflow", "auto", ["Threshold: 0.35"]),
        ("Curve check: passed", "checks", "judged", []),
        ("Check against the neighbouring curves: passed", "checks", "judged", []),
        ("M1 picked", "you", "box", []),
    ]

    # M0 picked again by PAC's own automatic picking: the assistant's checks were of another
    # curve, not shown; its history stays in the log.
    assert client.post(f"/dispersion_images/{folder}/2.5/pick/auto").status_code == 200
    card = client.get(f"/quality/dispersion/card/{folder}/2.5").json()
    gates = {gate["gate"]: gate["verdict"] for gate in card["gates"]}
    assert card["picked_by"] == "auto" and "G4" not in gates
    # The automatic pick now, measured as the curve check would; M1 as it was.
    (m1, m0, measured) = _done(card)[2:]
    assert m1 == ("M1 picked", "you", "box", [])
    assert m0 == ("M0 picked", "you", "auto", [])
    assert measured[1:3] == ("checks", "measured")
    picking = [line for line in log.read_text().splitlines() if '"stage": "picking"' in line]
    assert any('"unit": "xmid_2.50"' in line for line in picking)


def test_an_assistant_run_shows_g5_and_what_each_attempt_changed(judged: str) -> None:
    overview = client.get(f"/quality/inversion/overview/{judged}").json()

    assert overview["paco"] is True
    # 5.5 inverted by hand before (test_api's run): who made each model now.
    assert overview["summary"].startswith(
        "3 windows · 1 inverted by the workflow agent · 1 inverted by the user · 1 passed · "
    )
    cells = {one["x"]: one for one in overview["cells"]}
    assert cells[2.5]["status"] == "pass" and cells[2.5]["hover"][1] == "Passed"

    card = client.get(f"/quality/inversion/card/{judged}/2.5").json()
    assert card["status"] == "pass"
    # The workflow's rules; the sampling, as its retry changed it, with what asked it.
    settings = {one["key"]: one for one in card["settings"]}
    assert all(one["origin"] == "workflow" for one in settings.values())
    assert settings["effort"]["why"] == (
        "the workflow's default; changed for this window, asked by the model check: chains not "
        "converged"
    )
    assert "changed" not in settings["drop"]["why"]
    assert card["verdict"]["text"] == (
        "Passed the model check and the check against the neighbouring models."
    )
    # What was done: the workflow's retry, what asked it and what it changed, then its checks;
    # its first run and PAC's inversion before them, replaced, left out.
    assert _done(card) == [
        (
            "Inverted",
            "workflow",
            "MCMC",
            [
                "Asked by the model check: chains not converged",
                "Sampling: 2000 iterations, was 1000",
            ],
        ),
        ("Model check: passed", "checks", "judged", []),
        ("Check against the neighbouring models: passed", "checks", "judged", []),
    ]
    gates = {gate["gate"]: gate for gate in card["gates"]}
    assert gates["G5"]["verdict"] == "pass" and gates["G6"]["metrics"][0]["name"] == "misfit"
    # Its measures against the limits the check judged them by.
    rhat = next(one for one in gates["G5"]["metrics"] if one["name"] == "rhat")
    assert (rhat["value"], rhat["threshold"]) == (1.02, 1.1)


def test_a_window_inverted_again_in_pac_drops_the_assistants_verdict(judged: str) -> None:
    folder = f"{judged}-again"
    shutil.copytree(OUTPUT_DIR / judged, OUTPUT_DIR / folder)
    window = OUTPUT_DIR / folder / "xmid_2.50"
    later = time.time() + 3600
    for name in ("SeismicInversion_Samples_0000.npz", "SeismicInversion_Measures_0000.json"):
        os.utime(window / name, (later, later))

    card = client.get(f"/quality/inversion/card/{folder}/2.5").json()

    # The checks, the attempts and what each changed were of the assistant's model.
    assert card["verdict"] is None and card["gates"][0]["verdict"] is None
    assert len(card["gates"]) == 1
    assert "Inverted by the user." in _texts(card)
    # What was done: the user's model now, dated by its files, measured against the agent's
    # limits as no check judged it; the workflow's, which it replaced, left out with its checks.
    (inverted, measured) = _done(card)
    assert inverted == ("Inverted", "you", "MCMC", [])
    assert measured[1:3] == ("checks", "measured")
    overview = client.get(f"/quality/inversion/overview/{folder}").json()
    assert overview["summary"].startswith("3 windows · 2 inverted by the user")
    # What was done says it after the workflow's steps: by the user, its windows named.
    history = client.get(f"/quality/run/{folder}").json()["history"]
    assert {step["title"]: step["lines"] for step in history if step["by"] == "you"}[
        "Inverted the curves into Vs profiles"
    ] == ["2 of 3 windows", "xmid 2.5 m, xmid 5.5 m"]


def test_a_soil_column_made_again_in_pac_drops_the_assistants_checks(judged: str) -> None:
    folder = f"{judged}-soils"
    shutil.copytree(OUTPUT_DIR / judged, OUTPUT_DIR / folder)
    window = OUTPUT_DIR / folder / "xmid_2.50"
    later = time.time() + 3600
    for path in window.glob("PetroInversion_*"):
        os.utime(path, (later, later))

    card = client.get(f"/quality/petro/card/{folder}/2.5").json()

    assert card["verdict"] is None
    (soil, measured) = _done(card)
    assert soil == ("Soil column made", "you", "Silex", [])
    assert measured[1:3] == ("checks", "measured")
    assert {gate["gate"]: gate["verdict"] for gate in card["gates"]} == {"G7": None}
    assert card["column"] is not None


def test_an_assistant_run_shows_g7_and_g8(judged: str) -> None:
    card = client.get(f"/quality/petro/card/{judged}/2.5").json()

    gates = {gate["gate"]: gate for gate in card["gates"]}
    assert {gate: one["verdict"] for gate, one in gates.items()} == {"G7": "pass", "G8": "pass"}
    # Against its neighbours, in G8's measures: the sentences say what they do not.
    jump = next(one for one in gates["G8"]["metrics"] if one["name"] == "water_table_jump")
    assert (jump["value"], jump["threshold"]) == (0.5, 1.0)
    assert "Water table at 1 m." in _texts(card)
    # What was done: the workflow's soil column over PAC's, checked by G7 and G8.
    assert _done(card) == [
        ("Soil column made", "workflow", "Silex", []),
        ("Soil column check: passed", "checks", "judged", []),
        ("Check against the neighbouring soil columns: passed", "checks", "judged", []),
    ]


def test_a_run_processed_in_pac_says_what_was_done_to_each_unit(run: str) -> None:
    # Each made by PAC's processing, by the user: a record, measured against the agent's limits
    # as no check judged it.
    record = client.get(f"/quality/records/card/{run}/1.mseed").json()
    ((made, by, how, lines), (_, checks, measured, _)) = _done(record)
    assert (made, by, how, lines) == ("Preprocessed", "you", "active", [])
    assert (checks, measured) == ("checks", "measured")
    # A window's image measured alike; its curve picked in a box, the user's, taken as it is.
    window = client.get(f"/quality/dispersion/card/{run}/2.5").json()
    steps = _done(window)
    assert steps[0] == ("Dispersion image made", "you", "active", [])
    assert steps[1][1:3] == ("checks", "measured")
    assert steps[2:] == [("M0 picked", "you", "box", [])]
    # Its Vs profile and soil column made in PAC's pages, each measured as their checks would.
    for stage, title, tool in (
        ("inversion", "Inverted", "MCMC"),
        ("petro", "Soil column made", "Silex"),
    ):
        ((made, by, how, lines), (_, _, measured, _)) = _done(
            client.get(f"/quality/{stage}/card/{run}/2.5").json()
        )
        assert (made, by, how, lines) == (title, "you", tool, []) and measured == "measured"
    # A window never picked: its image alone; on the picking page, who picked each curve, how.
    unpicked = client.get(f"/quality/dispersion/card/{run}/8.5").json()
    assert [title for title, _, _, _ in _done(unpicked)][:1] == ["Dispersion image made"]
    assert all(not title.startswith("M0") for title, _, _, _ in _done(unpicked))
    (curve,) = client.get(f"/dispersion_images/{run}/2.5").json()["curves"]
    assert curve["made"] == {"by": "you", "how": {"label": "box", "tip": ""}}


def test_a_change_in_pac_replaces_what_was_done_before_and_keeps_it(run: str) -> None:
    folder = f"{run}-changed"
    shutil.copytree(OUTPUT_DIR / run, OUTPUT_DIR / folder)
    window = OUTPUT_DIR / folder / "xmid_2.50"
    before = (window / "DispersionCurves_0000.csv").read_text()

    picked = client.post(f"/dispersion_images/{folder}/2.5/pick/auto")
    assert picked.status_code == 200
    (curve,) = picked.json()["curves"]
    assert curve["made"] == {"by": "you", "how": {"label": "auto", "tip": ""}}
    card = client.get(f"/quality/dispersion/card/{folder}/2.5").json()

    # The automatic pick now, replacing the box: measured as the curve check would, as no check
    # judged it; the box left out.
    (pick, measured) = _done(card)[2:]
    assert pick == ("M0 picked", "you", "auto", [])
    assert measured[1:3] == ("checks", "measured")
    # What was made of the older curve set aside: no Vs profile, no soil column said.
    for stage in ("inversion", "petro"):
        assert client.get(f"/quality/{stage}/card/{folder}/2.5").json()["steps"] == []
    # Nothing lost: the curves it replaced, kept where its record says.
    (change,) = [
        json.loads(line)
        for line in (window / "history.jsonl").read_text().splitlines()
        if json.loads(line)["how"] == "auto"
    ]
    assert (window / change["aside"] / "DispersionCurves_0000.csv").read_text() == before
    assert change["later"] == ["inversion", "petro_inversion"]


def test_a_check_left_out_leaves_every_card_and_count(judged: str) -> None:
    def g4() -> tuple[list[str], list[str]]:
        card = client.get(f"/quality/dispersion/card/{judged}/2.5").json()
        history = client.get(f"/quality/run/{judged}").json()["history"]
        said = [title for title, _, _, _ in _done(card)]
        counted = [line for step in history for line in step["lines"]]
        name = "Check against the neighbouring curves"
        return (
            [one for one in said if one.startswith(name)],
            [one for one in counted if one.startswith(name)],
        )

    kind = CHECKS["G4"]
    unregister_check("G4")
    try:
        # Left out: none of its verdicts on the card, none counted on the run card.
        assert g4() == ([], [])
    finally:
        register_check(kind)
    assert g4() == (
        ["Check against the neighbouring curves: passed"],
        ["Check against the neighbouring curves: 1 passed"],
    )


def test_the_endpoints_write_nothing(judged: str) -> None:
    # Visualization reads what the stages saved: no file made or written again.
    folder = Path(OUTPUT_DIR / judged)
    before = {(path, path.stat().st_mtime_ns) for path in folder.rglob("*") if path.is_file()}
    for path in (
        f"quality/run/{judged}",
        f"quality/sources/{judged}/2.5",
        f"quality/records/overview/{judged}",
        f"quality/records/card/{judged}/1.mseed",
        f"quality/dispersion/overview/{judged}",
        f"quality/dispersion/card/{judged}/2.5",
        f"quality/inversion/overview/{judged}",
        f"quality/inversion/card/{judged}/2.5",
        f"quality/inversion/chains/{judged}/2.5",
        f"quality/petro/overview/{judged}",
        f"quality/petro/card/{judged}/2.5",
        f"inversion/velocity_section/{judged}",
        f"inversion/pseudo_section_comparison/{judged}/M0",
        f"petro_inversion/section/{judged}",
    ):
        assert client.get(f"/{path}").status_code == 200, path
    after = {(path, path.stat().st_mtime_ns) for path in folder.rglob("*") if path.is_file()}
    assert after == before


def test_a_receiver_the_line_left_out_is_the_windows_and_near_shots_the_near_fields(
    judged: str,
) -> None:
    folder = f"{judged}-receiver"
    target = OUTPUT_DIR / folder
    shutil.copytree(OUTPUT_DIR / judged, target)
    now = datetime.now(UTC).isoformat()
    off = _flag(
        "off_decay_receivers",
        "The receiver at 3 m (2 of 2 records) is too weak or too strong for its distance from "
        "the shot (off the amplitude decay with offset) in most of the records that reach it: a "
        "noisy or badly coupled geophone, left out of every window.",
        "preprocessing",
        {"kind": "exclude_traces", "record": "line", "traces": [3]},
    )
    line = _result("G1", "line", "pass", [_metric("off_decay_receivers", 1, 0, "max")], [off])
    near = (
        "near_field distance_m 3 m: a window stacks no shot nearer than that to its nearest "
        "receiver, half the longest wavelength its trial curves reached (6 m), where it has a "
        "farther one."
    )
    lines = (target / "qc_log.jsonl").read_text().splitlines()[:-1]
    lines += [
        _attempt("line", "preprocessing", 1, "initial", {"G1": line}, at=now),
        _attempt(
            "line",
            "phase_shift",
            1,
            "initial",
            {},
            {"masw": {"length": 6}, "near_field": {"distance_m": 3.0}},
            (near,),
            at=now,
        ),
    ]
    (target / "qc_log.jsonl").write_text("\n".join(lines) + "\n")
    manifest = json.loads((target / "run.json").read_text())
    manifest["exclusions"] = {"records": [], "traces": {"1.mseed": [3], "2.mseed": [3]}}
    (target / "run.json").write_text(json.dumps(manifest))
    # Window 2.5 (0 to 5 m): the shot at -2 m, 2 m from its first receiver, in its near field;
    # the one at 13 m, 8 m from its last, stacked without the receiver the line left out.
    window = json.loads((target / "xmid_2.50" / "window.json").read_text())
    keep = [i for i, path in enumerate(window["selected_files"]) if path.endswith("2.mseed")]
    window["selected_files"] = [window["selected_files"][i] for i in keep]
    window["acquisitions"] = [window["acquisitions"][i] for i in keep]
    window["record_receivers"] = [[0, 1, 2, 4, 5]] * len(keep)
    (target / "xmid_2.50" / "window.json").write_text(json.dumps(window))

    sources = client.get(f"/quality/sources/{folder}/2.5").json()

    shots = {one["name"]: one for one in sources["shots"]}
    assert shots["2.mseed"]["use"] == "used"
    assert shots["1.mseed"]["use"] == "near"
    assert shots["1.mseed"]["why"] == (
        "2 m from the window's nearest receiver: in its near field (nearer than 3 m, half the "
        "longest wavelength)"
    )
    texts = _texts(sources)
    assert (
        "Its receiver at 3 m is left out, as from every window: too weak or too loud in most of "
        "the line's records." in texts
    )
    assert "Shots not used: 1 nearer than 3 m to the window, in its near field." in texts
    card = client.get(f"/quality/run/{folder}").json()
    settings = {one["key"]: one for one in card["settings"]}
    assert (
        "The receiver at 3 m (2 of 2 records) is too weak or too strong"
        in settings["records"]["why"]
    )
    # No shot within 3 m of a window's receivers: beyond 5.5 m of its middle, its half 2.5 m.
    reach = settings["reach"]
    assert (reach["value"], reach["detail"]) == ("beyond 5.5 m", "from each window's middle")
    assert reach["why"].startswith(
        "nearest, 3 m from a window's receivers: a window stacks no shot nearer than that"
    )
    record = client.get(f"/quality/records/card/{folder}/1.mseed").json()
    assert any(text.startswith("The receiver at 3 m") for text in _texts(record))
    assert not any(text.startswith("Its trace at 3 m") for text in _texts(record))


def test_a_curves_points_past_the_images_lines_are_measured() -> None:
    # Under λmin (two spacings, 2 m) the aliasing zone, over λmax (three window lengths, 15 m)
    # beyond the window's reach: where G3's flags start, measured on a curve no check judged. A
    # point on each, a hair past it in float32, is within it.
    receivers = tuple(Coordinate(2.0 + k, 0.0, 0.0) for k in range(6))
    curve = DispersionCurve(
        # 200 m, 15.0000004 m, 12 m, 3.2 m, 1.9999998 m, 1.5 m
        fs=np.array([1.0, 6.666666507720947, 10, 50, 75.00000762939453, 100], dtype=np.float32),
        vs=np.array([200.0, 100, 120, 160, 150, 150], dtype=np.float32),
        mode=Mode("M", 0),
        acquisition=LinearAcquisition(source=Coordinate(0.0, 0.0, 0.0), receivers=receivers),
        type=VelocityType.PHASE,
    )

    # Its image's grid: the pick measured on it, as G3 measures one (sigpipe's measure_curve).
    fs, vs = np.arange(1.0, 101.0), np.arange(50.0, 301.0)
    image = DispersionImage(
        fv_map=np.full((fs.size, vs.size), 0.5),
        fs=fs,
        vs=vs,
        type=VelocityType.PHASE,
        acquisition=curve.acquisition,
    )

    metrics = {one.name: one for one in curve_metrics(image, curve, CurveLimits())}

    for name in ("aliased_points", "beyond_reach_points"):
        assert metrics[name].value == 0.167 and not metrics[name].passed
        assert metrics[name].of == "curve" and metrics[name].over.startswith("the curve's 6 points")


def test_an_inversion_whose_data_chose_the_layers_shows_how_its_chains_moved(run: str) -> None:
    # On a copy of the run, a window inverted again with the layers chosen by the data: each
    # move's acceptance and step, the exchanges between tempered copies, the rows' steps
    # relative to their values.
    folder = f"{run}-free"
    shutil.copytree(OUTPUT_DIR / run, OUTPUT_DIR / folder)
    config = {
        "folder": folder,
        "positions": [2.5],
        "labels": ["M0"],
        "parameters": {"n_iterations": 1_500, "n_chains": 2},
        "n_workers": 1,
    }
    assert _wait(client.post("/inversion/run", json=config).json())["state"] == "succeeded"

    card = client.get(f"/quality/inversion/card/{folder}/2.5").json()

    assert {"birth", "death", "vs", "noise"} <= set(card["moves"])
    assert set(card["move_steps"]) == {"interface", "vs", "noise", "shift", "stretch"}
    assert card["exchanges"] is not None
    rows = {row["parameter"]: row for row in card["convergence"]}
    assert all(row["step_unit"] == "%" for name, row in rows.items() if name.startswith("vs@"))
    assert rows["noise"]["step_unit"] == "%" and rows["layers"]["step"] is None
    # The chains' median acceptance: a warning outside 20 to 30 % (their steps aim at 30 %).
    (gate,) = card["gates"]
    acceptance = [metric for metric in gate["metrics"] if metric["name"] == "acceptance"]
    assert [(one["threshold"], one["bound"]) for one in acceptance] == [(20, "min"), (30, "max")]


def _retried(unit: str, stage: str, triggered_by: str, parameters: dict[str, Any]) -> Attempt:
    """An attempt of the assistant's: a redo's fresh first one, or the line's mute trial."""
    return Attempt(
        unit=unit,
        stage=stage,
        attempt=1,
        parameters=parameters,
        triggered_by=triggered_by,
        started_at=datetime.now(UTC),
        status="succeeded",
    )


MUTE = {"method": "mute", "vmin": 80.0, "vmax": 1500.0}


def test_the_models_verdicts_are_the_logs_now_not_a_jobs_end() -> None:
    from masw.io.quality.journal import JournalEntry
    from masw.io.quality.runs import _model_verdicts  # pyright: ignore[reportPrivateUsage]

    # The job's end said retry; a later job, stopped before its end, made it pass.
    ended = JournalEntry(
        id="I1",
        kind="inversion",
        at=datetime.now(UTC),
        text="I1 inverted",
        data={"g5": {"xmid_2.50": "retry"}},
    )
    again = _retried("xmid_2.50", "inversion", "G5:not_converged", {}).model_copy(
        update={
            "results": {"G5": GateResult.model_validate(_result("G5", "xmid_2.50", "pass", [], []))}
        }
    )

    assert _model_verdicts([ended], QCLog([again])) == {"xmid_2.50": "pass"}
    # A run whose log holds none: the journal's.
    assert _model_verdicts([ended], None) == {"xmid_2.50": "retry"}


def test_a_near_field_that_dropped_no_shot_is_not_said(run: str) -> None:
    def logged(distance: float) -> QCLog:
        near = {"near_field": {"distance_m": distance}}
        return QCLog([_retried(LINE, "phase_shift", "initial", near)])

    # Farther than any shot a window stacks: every window kept its near shots, none farther.
    assert near_field_used(OUTPUT_DIR / run, logged(500.0)) is None
    # Nearer than every shot of a window: it shows there.
    assert near_field_used(OUTPUT_DIR / run, logged(0.5)) == 0.5


def test_a_windows_records_muted_each_with_its_own_pulse_are_said_together(run: str) -> None:
    manifest = read_manifest(OUTPUT_DIR / run)
    assert manifest is not None
    # A redo after the image check: each record muted afresh with its own pulse's width.
    log = QCLog(
        [
            _retried(
                "1.mseed",
                "preprocessing",
                "G2:weak_coherence",
                {"muting": {**MUTE, "width": 0.035}},
            ),
            _retried(
                "2.mseed",
                "preprocessing",
                "G2:weak_coherence",
                {"muting": {**MUTE, "width": 0.094}},
            ),
        ]
    )

    settings = {one.key: one for one in records_in_common(manifest, log, ["1.mseed", "2.mseed"])}

    muting = settings["muting"]
    assert (muting.value, muting.detail) == (
        f"80{DASH}1500 m/s",
        f"signal width 0.035{DASH}0.094 s",
    )
    assert muting.origin == "checks" and not muting.spread
    assert muting.why == (
        "changed for 2 records, asked by the dispersion image check: weak coherence"
    )
    # The run's card counts a redo's fresh attempts as done again.
    card = {one.key: one for one in processing_settings(manifest, log, None)}
    assert card["records"].why.endswith("preprocessed again: 2 records (2 weak coherence)")


def test_a_window_the_agent_could_not_pick_is_marked_as_its_check_judged_it() -> None:
    from masw.io.quality.dispersion import _picks  # pyright: ignore[reportPrivateUsage]

    # Its retries spent, the curve check kept it flagged: orange, not rejected.
    flagged = GateResult.model_validate(_result("G3", "xmid_2.50", "retry", [], []))
    (said,) = _picks([], None, flagged, (), None, True, "by the workflow agent")
    assert (said.mark, said.text) == ("flag", "Not picked.")
    # No verdict of the check: none picked where the agent ran.
    (said,) = _picks([], None, None, (), None, True, "by the workflow agent")
    assert said.mark == "fail"


def test_a_window_is_judged_only_where_its_checks_made_its_state() -> None:
    from masw.io.quality.dispersion import _judged  # pyright: ignore[reportPrivateUsage]

    # Its image passed by the image check, its curve auto-picked in PAC beyond the agent's
    # limits: the window's state is the curve's measure, no verdict.
    assert not _judged("note", ("pass", True), ("note", False))
    # Its curve within the limits by its measures: not every part in its state was checked.
    assert not _judged("pass", ("pass", True), ("pass", False))
    # Its image rejected by its check, whatever its curve; both parts passed by their checks.
    assert _judged("fail", ("fail", True), ("pass", False))
    assert _judged("pass", ("pass", True), ("pass", True))


def test_a_window_asked_twice_for_one_flag_is_counted_once() -> None:
    from masw.io.quality.runs import _again  # pyright: ignore[reportPrivateUsage]

    # Picked a second and a third time for the same narrow range: one window, asked by it.
    first = _retried("xmid_2.50", "picking", "G3:narrow_span", {})
    log = QCLog([first.model_copy(update={"attempt": n}) for n in (2, 3)])

    assert _again(log, "picking", "window") == "1 window (1 narrow range of wavelengths)"


def test_records_made_again_by_a_change_of_the_lines_settings_are_said_so(run: str) -> None:
    manifest = read_manifest(OUTPUT_DIR / run)
    assert manifest is not None
    # The assistant's line loop: every record made again with the line's new settings, none of
    # its own.
    log = QCLog(
        [_retried(name, "preprocessing", "line change", {}) for name in ("1.mseed", "2.mseed")]
    )

    card = {one.key: one for one in processing_settings(manifest, log, None)}

    assert card["records"].why.endswith(
        "preprocessed again: 2 records (2 with a change of the line's settings)"
    )


def test_a_filter_is_said_between_the_mute_and_the_dispersion_image(run: str) -> None:
    manifest = read_manifest(OUTPUT_DIR / run)
    assert manifest is not None
    preset = manifest.preset.model_dump(mode="json")
    preset["muting"] = {"method": "mute", "vmin": 100.0, "vmax": 800.0}
    preset["filtering"] = {"method": "iir", "fmin": 5.0, "fmax": 50.0, "order": 4}
    filtered = manifest.model_copy(update={"preset": type(manifest.preset).model_validate(preset)})

    card = {one.key: one for one in processing_settings(filtered, None, None)}

    # The active line's order: the mute, then the filter, then the dispersion image and its stack.
    assert list(card) == [
        "windows",
        "records",
        "reach",
        "mute",
        "filtering",
        "band",
        "image_stacking",
    ]
    assert (card["mute"].value, card["filtering"].label, card["filtering"].value) == (
        f"100{DASH}800 m/s",
        "Filter",
        f"5{DASH}50 Hz",
    )


def test_a_window_says_the_muting_most_of_its_records_ran_with() -> None:
    def muted(width: float) -> Setting:
        return Setting(
            key="muting",
            label="Mute",
            value=f"80{DASH}1500 m/s",
            detail=f"signal width {width} s",
            why="changed for this record, asked by the dispersion image check: weak coherence",
            origin="workflow",
        )

    unmuted = Setting(key="muting", label="Mute", value="none", why="", origin="default")
    widths = {"r1": 0.035, "r2": 0.05, "r3": 0.094}
    per_unit = {name: [muted(width)] for name, width in widths.items()}
    per_unit |= {"r4": [unmuted], "r5": [unmuted]}
    values = {name: {"muting": {**MUTE, "width": width}} for name, width in widths.items()}
    values |= {name: {"muting": {"method": "none"}} for name in ("r4", "r5")}
    causes = {name: {"muting": ["G2 weak coherence"]} for name in widths}
    causes |= {name: {"muting": []} for name in ("r4", "r5")}

    (muting,) = in_common(per_unit, values, causes, "record")

    # Three of five muted, each its own width: muted, not the two left unmuted.
    assert (muting.value, muting.detail) == (
        f"80{DASH}1500 m/s",
        f"signal width 0.035{DASH}0.094 s",
    )
    assert muting.spread == "other records: none on r4, r5"


def test_a_records_muting_the_mute_trial_chose_comes_from_the_data(run: str) -> None:
    manifest = read_manifest(OUTPUT_DIR / run)
    assert manifest is not None
    # The line's mute trial: its muting logged with the line's images.
    log = QCLog([_retried(LINE, "phase_shift", "initial", {"muting": {**MUTE, "width": 0.05}})])

    muting = next(one for one in record_settings(manifest, log, "1.mseed") if one.key == "muting")

    # The line's: who set it, the run's card saying why.
    assert (muting.origin, muting.why) == ("workflow", "")


def _lines(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(one) + "\n" for one in records))


def _study(profile: Path, run_id: str) -> Path:
    """A study of PACo's autonomous mode whose line is run `run_id`: its journal (the
    objective, a hypothesis, an experiment testing it, the commit it led to, a conclusion) and
    its search, a sheet judged by an earlier search among those of its own."""
    study = profile / "studies" / "20261021-145900-0000"
    study.mkdir(parents=True)
    commit = {"number": 1, "run_id": run_id}
    (study / "line.json").write_text(json.dumps({"run_id": run_id, "commits": [commit]}))
    variants: list[dict[str, Any]] = [
        {"label": "current", "changes": {}, "passing": 5, "windows": 12, "depth_m": 12.0},
        {
            "label": "1",
            "changes": {"masw": {"length": 6}},
            "passing": 9,
            "windows": 12,
            "gained": [1.5, 2.5, 3.5, 4.5],
            "lost": [],
            "depth_m": 15.0,
        },
        {
            "label": "2",
            "changes": {"masw": {"length": 8}},
            "passing": 4,
            "windows": 12,
            "gained": [],
            "lost": [2.5],
            "depth_m": 16.0,
            "dominated_by": "1",
        },
    ]
    changes = {"masw": {"length": 6}, "dispersion": {"fmax": 60}}
    reasons = {"masw length": "the most windows passing in E1", "dispersion fmax": USER_REASON}
    _lines(
        study / "journal.jsonl",
        [
            {
                "id": "O1",
                "kind": "objective",
                "at": "2026-10-21T14:59:00Z",
                "text": "O1 to 20 m",
                "data": {"given": {"mode": "active", "line": {}, "inversion": {}}},
            },
            {
                "id": "H1",
                "kind": "hypothesis",
                "at": "2026-10-21T15:02:00Z",
                "text": "Longer windows sharpen the low band",
            },
            {
                "id": "E1",
                "kind": "experiment",
                "at": "2026-10-21T15:05:00Z",
                "text": "on spread:12: current 5/12; ...",
                "of": ["H1"],
                "data": {"windows": "spread:12", "variants": variants},
            },
            {
                "id": "L1",
                "kind": "commit",
                "at": "2026-10-21T15:10:00Z",
                "text": "L1 made run ...",
                "of": ["E1"],
                "data": {"run_id": run_id, "changes": changes, "reasons": reasons, "new_run": True},
            },
            {
                "id": "C1",
                "kind": "conclusion",
                "at": "2026-10-21T15:12:00Z",
                "text": "C1 Six receivers hold the low band",
                "of": ["E1", "L1"],
            },
        ],
    )
    rows = [{"label": f"g{n}", "length": 6} for n in (1, 2, 3)]
    axes = {"masw length": ["6", "8"], "masw distance_max": ["0.5", "1"]}
    search = {"made_at": "2026-10-21T15:00:00+00:00", "rows": rows, "dropped": 1, "axes": axes}
    (study / "searches").mkdir()
    (study / "searches" / "S1.json").write_text(json.dumps(search))
    knockout = {"round": 2, "sheets": [], "finalists": ["g1", "g2"], "experiment": "E1"}
    (study / "searches" / "S1_knockout.json").write_text(json.dumps(knockout))
    _lines(
        study / "searches" / "S1_judged.jsonl",
        [
            {"round": 1, "sheet": 1, "kept": ["g3"], "at": "2026-10-21T14:00:00+00:00"},
            {"round": 1, "sheet": 1, "kept": ["g1"], "at": "2026-10-21T15:01:00+00:00"},
            {"round": 1, "sheet": 2, "kept": ["g2"], "at": "2026-10-21T15:01:00+00:00"},
            {"round": 2, "sheet": 1, "kept": ["g1", "g2"], "at": "2026-10-21T15:01:30+00:00"},
        ],
    )
    return study


def test_a_search_shows_its_finalists_the_one_it_chose_first(tmp_path: Path) -> None:
    study = tmp_path / "study"
    (study / "searches" / "S1_images").mkdir(parents=True)
    rows = [
        {
            "label": f"g{n}",
            "changes": {"masw": {"length": 5 + 2 * n}},
            "card": {"covered": n, "windows": 12, "depth_m": 10.0 + n},
        }
        for n in range(1, 5)
    ]
    search = {
        "made_at": "2026-10-21T15:00:00+00:00",
        "rows": rows,
        "dropped": 0,
        "axes": {"length": ["7", "9", "11", "13"]},
        "places": [10.0 * n for n in range(1, 13)],
    }
    (study / "searches" / "S1.json").write_text(json.dumps(search))
    knockout = {"round": 3, "sheets": [], "finalists": ["g2", "g3", "g4"], "best": "g3"}
    (study / "searches" / "S1_knockout.json").write_text(json.dumps(knockout))
    (study / "searches" / "S1_images" / "g3.npz").write_bytes(b"")
    commit = JournalEntry(
        id="L1",
        kind="commit",
        at=datetime.now(UTC),
        text="",
        data={"changes": {"masw": {"length": 11}}},
    )
    defaults = make_preset("active").model_dump(mode="json")

    searched = search_of(study)
    (table,) = tried_tables(tmp_path / "run", [commit], searched, study, 1.0, defaults, None)

    # What its combinations did not all share: the window length alone.
    assert searched is not None and searched.varied == {"masw", "masw length"}

    # What the search did, then its finalists side by side, the one it chose first and the line
    # made with, each with how many of its windows a curve passed, how deep, and its images where
    # the study kept them; nothing tried after it.
    assert table.title == "Search"
    assert table.intro == (
        "4 combinations of settings compared by the agent's vision, each on 12 dispersion images "
        "spread along the line",
        "The 3 best of them:",
    )
    assert [
        (row.name, row.kept, row.trial, row.images, row.clear, row.depth) for row in table.rows
    ] == [
        ("Option 1", True, "g3", True, "3 of 12", "13 m"),
        ("Option 2", False, "g2", False, "2 of 12", "12 m"),
        ("Option 3", False, "g4", False, "4 of 12", "14 m"),
    ]


def test_a_study_is_read_for_the_runs_its_line_was(tmp_path: Path) -> None:
    profile = tmp_path / "shots"
    study = _study(profile, "20261021-151000-abcd")
    (study / "journal.jsonl").write_text((study / "journal.jsonl").read_text() + "not json\n")

    # A run of its line: the study, its journal (a line unread left out) and its search, the
    # sheets judged since it was made alone.
    assert study_of(profile, "20261021-151000-abcd") == study
    assert [one.id for one in journal_of(study)] == ["O1", "H1", "E1", "L1", "C1"]
    search = search_of(study)
    assert search is not None
    assert (search.variants, search.settings, search.dropped) == (3, 2, 1)
    assert (search.sheets, search.rounds) == (3, 2)
    assert [one.label for one in search.finalists] == ["g1", "g2"]
    assert study_of(profile, "20261021-160000-0000") is None
    assert study_of(tmp_path / "noise", "20261021-151000-abcd") is None


def test_an_autonomous_run_card_says_what_the_agent_chose_tried_and_did(judged: str) -> None:
    folder = "shots/20261021-151000-abcd"
    target = OUTPUT_DIR / folder
    shutil.copytree(OUTPUT_DIR / judged, target)
    (target / "coherence.json").unlink()
    study = _study(target.parent, target.name)
    # The line's settings as the agent's commit logs them: its own with its reason, the user's
    # as theirs.
    notes = (
        "masw length: the most windows passing in E1 (L1, E1)",
        f"dispersion fmax: {USER_REASON} (L1)",
    )
    lines = (target / "qc_log.jsonl").read_text().splitlines()[:-1]  # its last line cut short
    committed = {"masw": {"length": 6}, "dispersion": {"fmax": 60}}
    now = datetime.now(UTC).isoformat()
    lines.append(_attempt("line", "phase_shift", 1, "initial", {}, committed, notes, at=now))
    (target / "qc_log.jsonl").write_text("\n".join(lines) + "\n")
    try:
        card = client.get(f"/quality/run/{folder}").json()
        dispersion = client.get(f"/quality/dispersion/card/{folder}/2.5").json()
        runs = client.get("/quality/runs").json()
    finally:
        shutil.rmtree(target)
        shutil.rmtree(study.parent)

    assert card["made_by"] == "autonomous"
    entry = next(one for profile in runs for one in profile["runs"] if one["folder"] == folder)
    assert entry["made_by"] == "autonomous"
    # The assistant's choices said as its; the user's as theirs.
    settings = {one["key"]: one for one in card["settings"]}
    assert (settings["windows"]["origin"], settings["windows"]["why"]) == (
        "agent",
        "the most windows passing in E1; one every 3 receivers (3 m): given in the user's request",
    )
    assert (settings["band"]["origin"], settings["band"]["why"]) == ("you", USER_REASON)
    # No default in an autonomous run: its search set what its commit did not change, here the
    # same in every combination it compared.
    assert (settings["image_stacking"]["origin"], settings["image_stacking"]["why"]) == (
        "agent",
        "the same in every combination the search compared",
    )
    assert "default" not in {one["origin"] for one in card["settings"]}
    assert "default" not in {one["origin"] for one in dispersion["settings"]}
    # Who picked its curve and how: in what was done, by the agent's tag.
    assert _done(dispersion)[2] == ("M0 picked", "agent", "auto", ["Threshold: 0.35"])
    # Its comparison as a table, an option a column with every setting, the one the run applied
    # first; what the search made, dropped, judged and selected, where it ran; no journal id. The
    # line had no settings of its own then: none shown against the options.
    (experiment,) = card["tried"]
    assert (experiment["title"], experiment["by"]) == ("Search", "agent")
    assert experiment["intro"] == [
        "2 combinations of settings compared by the agent's vision on their dispersion images",
        "The 2 best of them:",
    ]
    assert experiment["columns"] == [
        "Windows",
        "Shots used",
        "Mute",
        "Filter",
        "Dispersion image",
        "Image stacking",
    ]
    every = ["any distance", "none", "none", f"0{DASH}100 Hz · 1{DASH}1000 m/s · 1000 steps"]
    rows = [
        (one["name"], one["kept"], one["values"], one["clear"], one["depth"])
        for one in experiment["rows"]
    ]
    assert rows == [
        (
            "Option 1",
            True,
            ["6 receivers (5 m) · every 1 receiver (1 m)", *every, "linear"],
            "9 of 12",
            "15 m",
        ),
        (
            "Option 2",
            False,
            ["8 receivers (7 m) · every 1 receiver (1 m)", *every, "linear"],
            "4 of 12",
            "16 m",
        ),
    ]
    # Why, in the assistant's own words, and what the user set.
    assert [(one["setting"], one["text"], one["by"]) for one in card["reasons"]] == [
        ("Windows", "The most windows passing in E1", "agent"),
        ("Dispersion image", "up to 60 Hz", "you"),
    ]
    # What was done, a step each, its facts a line each, by who did it; the assistant's
    # reasoning left to its journal.
    steps = card["history"]
    assert [(one["kind"], one["by"], _how(one), one["title"], one["lines"]) for one in steps] == [
        # What the user asked, the method they typed with it.
        ("objective", "you", "chat", "Asked", ["To 20 m", "Method: active"]),
        (
            "search",
            "agent",
            "vision",
            "Chose the processing settings",
            [
                "2 combinations of settings",
                "Dispersion images compared by the agent's vision",
                "The best of them chosen",
            ],
        ),
        (
            "commit",
            "agent",
            "active",
            "Processed the whole line",
            ["Windows: 6 receivers (5 m)", "Dispersion image: up to 60 Hz"],
        ),
        # Who made the curves, Vs profiles and soil columns that stand and how, in time: window
        # 5.5 picked and inverted in PAC, the user's; window 2.5, the agent's after the user's,
        # with what its checks said.
        ("picking", "you", "box", "Picked the curves", ["1 of 3 windows", "xmid 5.5 m"]),
        (
            "inversion",
            "you",
            "MCMC",
            "Inverted the curves into Vs profiles",
            ["1 of 3 windows", "xmid 5.5 m"],
        ),
        (
            "petro_inversion",
            "you",
            "Silex",
            "Inverted the curves into soil columns",
            ["1 of 3 windows", "xmid 5.5 m"],
        ),
        (
            "picking",
            "agent",
            "auto",
            "Picked the curves",
            [
                "1 of 3 windows",
                "Curve check: 1 passed",
                "Check against the neighbouring curves: 1 passed",
            ],
        ),
        (
            "inversion",
            "agent",
            "MCMC",
            "Inverted the curves into Vs profiles",
            [
                "1 of 3 windows",
                "Model check: 1 passed",
                "Check against the neighbouring models: 1 passed",
            ],
        ),
        (
            "petro_inversion",
            "agent",
            "Silex",
            "Inverted the curves into soil columns",
            [
                "1 of 3 windows",
                "Soil column check: 1 passed",
                "Check against the neighbouring soil columns: 1 passed",
            ],
        ),
    ]
