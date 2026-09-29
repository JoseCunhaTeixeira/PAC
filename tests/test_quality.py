"""Visualization's API on a synthetic run (tests/synthetic.py): what the files of a run PAC made
alone give, stage by stage (the run's card and its line, the shots each window stacks, records,
images and picks, seismic and petrophysical inversions), then the same run as the assistant
judged it, its QC log written here as PACo writes it: the gates' verdicts, metrics, flags and
attempts, and what each attempt changed."""

import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from masw.api.main import app
from masw.io.paths import OUTPUT_DIR
from masw.io.quality.dispersion import CurveThresholds, curve_metrics
from sigpipe.base import Coordinate, DispersionCurve, LinearAcquisition, Mode, VelocityType
from sigpipe.masw.inversion.measuring import USEFUL_REFERENCE, InversionMeasures, useful_depth
from sigpipe.masw.inversion.window import (
    SAMPLES_FILE,
    VS_SPREAD_FILE,
    load_profiles,
    load_vs_spread,
)

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
    assert entry["by"] == "pac" and entry["mode"] == "active" and entry["windows"] == 3
    assert entry["window_length"] == 6  # receivers, as the run's form set them
    started = [one["started_at"] for one in profiles["shots"]["runs"]]
    assert started == sorted(started, reverse=True)  # the newest first


def test_a_pac_run_card_says_its_windows_came_from_pacs_form(run: str) -> None:
    card = client.get(f"/quality/run/{run}").json()

    assert (card["profile"], card["by"], card["records"]) == ("shots", "pac", True)
    settings = {one["key"]: one for one in card["settings"]}
    assert list(settings) == [
        "length",
        "step",
        "windows",
        "reach",
        "records",
        "band",
        "preprocessing",
    ]
    assert (settings["length"]["value"], settings["length"]["detail"]) == ("5 m", "6 receivers")
    assert (settings["step"]["value"], settings["step"]["detail"]) == ("3 m", "3 receivers")
    assert settings["length"]["origin"] == "pac" and "set by hand" in settings["length"]["why"]
    assert settings["reach"]["value"] == "any distance"
    assert settings["records"]["value"] == "2 of 2"
    assert card["trials"] == []
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
        "Its dispersion image stacks the images of 2 shots, 7.5 m from its middle: "
        "1 on the left, 1 on the right."
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
    assert shots["1.mseed"]["why"] == "left out of every window by the signal check (G1)"
    assert shots["2.mseed"]["use"] == "far"
    assert shots["2.mseed"]["why"] == "10.5 m from the middle: beyond the 10 m the windows stack"
    # In short, the reasons in full and the shots named on hover.
    assert _texts(sources) == ["Shots left out: 1 rejected (G1), 1 too far."]
    assert sources["sentences"][0]["detail"] == (
        "Shots left out: 1 rejected by the signal check (1.mseed), 1 beyond the 10 m reach."
    )
    card = client.get(f"/quality/run/{folder}").json()
    reach = next(one for one in card["settings"] if one["key"] == "reach")
    assert reach["value"] == "within 10 m"


def test_a_passive_window_stacks_every_record() -> None:
    job = _wait(
        client.post(
            "/run",
            json={
                "profile": "noise",
                "mode": "passive",
                "overrides": {"masw": {"length": 6, "step": 6}},
                "workers": 1,
            },
        ).json()
    )
    assert job["state"] == "succeeded"
    card = client.get(f"/quality/run/{job['run']}").json()
    assert card["sources"] == {} and card["reach"] is None
    assert "reach" not in [one["key"] for one in card["settings"]]
    xmid = card["windows"][0]["xmid"]

    sources = client.get(f"/quality/sources/{job['run']}/{xmid}").json()

    assert sources["passive"] is True and sources["shots"] == []
    assert "every record of a passive line" in _texts(sources)[0]


def test_a_pac_run_shows_its_records_measures_without_verdicts(run: str) -> None:
    overview = client.get(f"/quality/records/overview/{run}").json()

    assert overview["paco"] is False and overview["summary"] == "2 of 2 records used"
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
    assert names[:4] == ["dead_traces", "clipped_traces", "nan_traces", "snr_db"]
    assert card["attempts"] == [] and card["windows"] == ["xmid_2.50", "xmid_5.50", "xmid_8.50"]
    assert _texts(card)[0].startswith("Median SNR")
    assert client.get(f"/quality/records/card/{run}/nope.mseed").status_code == 404


def test_a_pac_run_shows_its_images_and_picks(run: str) -> None:
    overview = client.get(f"/quality/dispersion/overview/{run}").json()

    assert overview["paco"] is False
    assert overview["summary"] == "2 of 3 windows picked by hand · 1 without a curve"
    cells = {one["x"]: one for one in overview["cells"]}
    assert cells[8.5]["status"] == "none" and cells[8.5]["value"] is None
    # A curve picked by hand is the user's: passed as it is, said so, nothing more.
    assert cells[2.5]["status"] == "pass" and cells[2.5]["value"] > 0
    assert cells[2.5]["hover"][:3] == ["xmid 2.5 m", "Curve: by hand", "1 mode picked: M0"]
    assert cells[2.5]["hover"][3].startswith("M0: ") and cells[8.5]["hover"][1] == "No curve"
    # Its modes, a dot each under its cell; none without a curve.
    assert cells[2.5]["modes"] == ["M0"] and cells[8.5]["modes"] == []
    # No image checked: each cell its curve alone, by hand or none, as its legend says.
    assert [cells[x]["parts"] for x in (2.5, 8.5)] == [["hand"], ["none"]]
    (curve,) = overview["parts"]
    assert curve["title"] == "Curve" and curve["legend"]["hand"] == "by hand"
    assert curve["legend"]["none"] == "no curve" and "fail" not in curve["legend"]
    (setting,) = overview["settings"]
    assert setting["origin"] == "pac"

    card = client.get(f"/quality/dispersion/card/{run}/2.5").json()
    assert card["picked_by"] == "hand" and [one["label"] for one in card["curves"]] == ["M0"]
    low, high = card["band_hz"]
    assert low < 20 < high
    gates = {gate["gate"]: gate for gate in card["gates"]}
    assert gates["G2"]["verdict"] is None and not gates["G2"]["by_hand"]
    # The curve's check passes the user's curve; the check along the line leaves it out.
    assert [(gates[one]["verdict"], gates[one]["by_hand"]) for one in ("G3", "G4")] == [
        ("pass", True),
        (None, True),
    ]
    assert gates["G3"]["metrics"] == [] and gates["G4"]["metrics"] == []
    assert {metric["name"] for metric in gates["G2"]["metrics"]} >= {
        "coherent_columns",
        "competing_ridges",
    }
    assert card["status"] == "pass" and card["verdict"]["text"] == "Picked by hand."
    assert card["parts"] == [{"label": "curve", "state": "hand"}]
    assert not any(text.startswith("M0 picked") for text in _texts(card))
    assert card["attempts"] == []

    unpicked = client.get(f"/quality/dispersion/card/{run}/8.5").json()
    assert unpicked["status"] == "none" and "No curve picked." in _texts(unpicked)


def test_a_pac_inversion_saves_its_measures_as_the_assistant_does(run: str) -> None:
    # The job saved them where the assistant saves its own; Visualization only reads them.
    path = OUTPUT_DIR / run / "xmid_2.50" / "SeismicInversion_Measures_0000.json"
    measures = InversionMeasures.model_validate_json(path.read_text())
    assert measures.samples_per_chain > 0 and measures.useful_reference == USEFUL_REFERENCE
    saved = path.stat().st_mtime_ns

    overview = client.get(f"/quality/inversion/overview/{run}").json()

    assert overview["paco"] is False
    assert overview["summary"].startswith("2 of 3 windows inverted by hand")
    cells = {one["x"]: one for one in overview["cells"]}
    assert cells[8.5]["status"] == "none" and cells[8.5]["hover"][-1] == "not inverted"
    assert 0 <= cells[2.5]["value"] <= cells[2.5]["total"]
    assert overview["track"]["kind"] == "depth"
    settings = {one["key"]: one for one in overview["settings"]}
    assert (settings["layers"]["value"], settings["layers"]["detail"]) == ("1", "over a half-space")
    assert settings["depth"]["value"] == "5 m"  # the sum of the thickness_max as run
    assert all(one["origin"] == "pac" for one in settings.values())
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
        "inverted before its measures were saved",
    )
    assert card["inverted"] is False and _texts(card) == [
        "Inverted before its measures were saved with it: invert it again to see them."
    ]
    assert path.read_text() == saved and path.stat().st_mtime_ns == older  # never measured here
    path.write_text(saved)


def test_a_depth_measured_by_an_older_rule_is_read_again_from_the_models_band(run: str) -> None:
    window = OUTPUT_DIR / run / "xmid_5.50"
    path = window / "SeismicInversion_Measures_0000.json"
    saved = path.read_text()
    # As saved before 2026-09-29: the depth read against a prior, and no band of the models' Vs.
    older = json.loads(saved) | {"useful_reference": "curve", "useful_depth_m": 0.123}
    path.write_text(json.dumps(older))
    (window / VS_SPREAD_FILE).unlink()

    card = client.get(f"/quality/inversion/card/{run}/5.5").json()
    section = client.get(f"/inversion/velocity_section/{run}").json()

    # Read again from the kept models' band, made from them once and kept beside them.
    spread = load_vs_spread(window)
    assert spread is not None
    # None: the whole model.
    expected = useful_depth(spread, load_profiles(window / SAMPLES_FILE), 0.25)
    assert card["profile"]["informed"] == expected != 0.123
    assert any("inform" in text for text in _texts(card))
    column = next(one for one in section["windows"] if one["x"] == 5.5)
    assert column["informed"] == (column["depth"] if expected is None else expected)
    assert json.loads(path.read_text())["useful_depth_m"] == 0.123  # never measured here
    path.write_text(saved)


def test_an_inversion_card_shows_the_model_its_fit_and_its_chains(run: str) -> None:
    card = client.get(f"/quality/inversion/card/{run}/2.5").json()

    assert card["inverted"] is True and card["title"] == "xmid 2.5 m · 1 layer over a half-space"
    assert card["parameters"]["n_layers"] == 2 and card["attempts"] == []
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
    assert card["figures"] == ["marginals", "density_curves", "dispersion_image"]
    # The median of the ensemble alone said (the user, 2026-09-29): no layered median.
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
    assert overview["paco"] is False and overview["summary"].startswith("2 of 3 windows inverted")
    (setting,) = overview["settings"]
    assert setting["value"] == model and setting["detail"].startswith("trained on ")
    assert setting["origin"] == "pac"
    cells = {one["x"]: one for one in overview["cells"]}
    assert cells[8.5]["status"] == "none" and cells[2.5]["value"] > 0

    card = client.get(f"/quality/petro/card/{run}/2.5").json()
    assert card["model"] == model and card["fit"]["model"] == "petro"
    column = card["column"]
    assert len(column["soils"]) == len(column["thicknesses_m"]) == len(column["ns"])
    assert column["water_table_m"] > 0
    assert card["curve"]["label"] == "M0" and len(card["curve"]["predicted_fs"]) > 0
    assert card["gates"][0]["gate"] == "G7" and card["gates"][0]["verdict"] is None
    assert any(text.startswith("Soil column, top down: ") for text in _texts(card))


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
    longer = {**PARAMETERS, "n_iterations": 2000, "n_burnin_iterations": 200, "n_layers": 2}
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
        _attempt("xmid_2.50", "inversion", 1, "initial", {"G5": g5_retry}, PARAMETERS, at=now),
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

    assert card["by"] == "assistant"
    settings = {one["key"]: one for one in card["settings"]}
    length = settings["length"]
    # The rule in a line; the trials in their table.
    assert length["origin"] == "rule" and "the shortest passing 80% of its trials" in length["why"]
    assert [(trial["length"], trial["passed"]) for trial in card["trials"]] == [(6, 2)]
    assert settings["step"]["origin"] == "given"  # 3, not the preset's 1
    assert settings["band"]["origin"] == "default"
    assert settings["preprocessing"]["why"].endswith(
        "done again for 1 record by the signal check (G1: 1 shifted trigger)"
    )
    assert card["trials"][0]["wavelengths_m"] == [2.0, 12.0]


def test_an_assistant_run_shows_g1_and_what_it_redid(judged: str) -> None:
    overview = client.get(f"/quality/records/overview/{judged}").json()

    assert overview["paco"] is True
    first, second = overview["cells"]
    # The attempt's last line is its current state: the second attempt passed.
    assert (first["status"], first["value"]) == ("pass", 9.5)
    assert second["status"] == "pass"  # measured: G1 said nothing of it

    card = client.get(f"/quality/records/card/{judged}/1.mseed").json()
    assert card["verdict"]["text"] == "The assistant's checks (G1) passed this record."
    assert "Preprocessed again (shifted trigger): trigger at 12 ms." in _texts(card)
    first_attempt, retried = card["attempts"]
    assert first_attempt["verdicts"] == {"G1": "retry"}
    assert first_attempt["flags"] == ["shifted_trigger"]
    assert retried["parameters"] == {"trigger": {"t0": 0.012}}


def test_an_assistant_run_shows_g2_g3_g4(judged: str) -> None:
    overview = client.get(f"/quality/dispersion/overview/{judged}").json()

    assert overview["paco"] is True
    cells = {one["x"]: one for one in overview["cells"]}
    # Its hover says its image and its curve apart, each with the gates that judged it.
    assert cells[2.5]["status"] == "pass"
    assert cells[2.5]["hover"][1:3] == ["Image: passed (G2)", "Curve: passed (G3, G4)"]
    # Its image and its curve apart; a window without a curve says none, whatever its image.
    assert cells[2.5]["parts"] == ["pass", "pass"]
    assert cells[8.5]["parts"][1] == "none" and cells[8.5]["status"] == "none"
    assert [part["title"] for part in overview["parts"]] == ["Image", "Curve"]
    assert overview["settings"][0]["origin"] == "rule"

    card = client.get(f"/quality/dispersion/card/{judged}/2.5").json()
    assert card["picked_by"] == "auto" and card["band_hz"] == [8.0, 55.0]  # G2's, as it judged it
    gates = {gate["gate"]: gate for gate in card["gates"]}
    assert [gates[gate]["verdict"] for gate in ("G2", "G3", "G4")] == ["pass"] * 3
    assert [metric["name"] for metric in gates["G3"]["metrics"]] == ["sharpness", "prominence"]
    assert card["verdict"]["text"] == "The assistant's checks (G2, G3, G4) passed this window."
    assert card["parts"] == [
        {"label": "image", "state": "pass"},
        {"label": "curve", "state": "pass"},
    ]
    assert len(card["attempts"]) == 2  # the phase shift's and the picking's


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
    # is the user's, passed as it is.
    assert card["picked_by"] == "hand" and card["verdict"]["text"] == "Picked by hand."
    gates = {gate["gate"]: (gate["verdict"], gate["by_hand"]) for gate in card["gates"]}
    assert gates == {"G2": ("pass", False), "G3": ("pass", True), "G4": (None, True)}
    assert [attempt["stage"] for attempt in card["attempts"]] == ["phase_shift"]
    assert not any("Picked again" in text for text in _texts(card))
    cells = {
        one["key"]: one
        for one in client.get(f"/quality/dispersion/overview/{folder}").json()["cells"]
    }
    assert "G3" not in " ".join(cells["xmid_2.50"]["hover"])
    assert "Curve: by hand" in cells["xmid_2.50"]["hover"]
    # Its image as the assistant checked it, its curve the user's.
    assert cells["xmid_2.50"]["parts"] == ["pass", "hand"]


def test_an_assistant_run_shows_g5_and_what_each_attempt_changed(judged: str) -> None:
    overview = client.get(f"/quality/inversion/overview/{judged}").json()

    assert overview["paco"] is True
    # 5.5 inverted by hand before (test_api's run): who made each model now.
    assert overview["summary"].startswith("2 of 3 windows inverted: 1 by the assistant, 1 by hand")
    cells = {one["x"]: one for one in overview["cells"]}
    assert cells[2.5]["status"] == "pass" and "G5 pass · G6 pass" in cells[2.5]["hover"]
    assert overview["settings"][0]["origin"] == "rule"

    card = client.get(f"/quality/inversion/card/{judged}/2.5").json()
    assert card["status"] == "pass"
    assert card["verdict"]["text"] == "The assistant's checks (G5, G6) passed this model."
    assert "Attempt 2: 2,000 iterations (was 1,000) (G5: not converged)." in _texts(card)
    # The run's own thresholds.
    assert any("(at most 1.2)" in text for text in _texts(card))
    gates = {gate["gate"]: gate for gate in card["gates"]}
    assert gates["G5"]["verdict"] == "pass" and gates["G6"]["metrics"][0]["name"] == "misfit"
    retried, passed = card["attempts"]
    assert (retried["triggered_by"], retried["verdicts"], retried["flags"]) == (
        "initial",
        {"G5": "retry"},
        ["not_converged"],
    )
    assert (passed["n_layers"], passed["depth_m"]) == (2, 5.0)


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
    assert card["attempts"] == [] and len(card["gates"]) == 1
    assert any(text.startswith("Inverted by hand: ") for text in _texts(card))
    assert not any(text.startswith("Attempt ") for text in _texts(card))
    overview = client.get(f"/quality/inversion/overview/{folder}").json()
    assert overview["summary"].startswith("2 of 3 windows inverted by hand")


def test_a_soil_column_made_again_in_pac_drops_the_assistants_checks(judged: str) -> None:
    folder = f"{judged}-soils"
    shutil.copytree(OUTPUT_DIR / judged, OUTPUT_DIR / folder)
    window = OUTPUT_DIR / folder / "xmid_2.50"
    later = time.time() + 3600
    for path in window.glob("PetroInversion_*"):
        os.utime(path, (later, later))

    card = client.get(f"/quality/petro/card/{folder}/2.5").json()

    assert card["verdict"] is None and card["attempts"] == []
    assert {gate["gate"]: gate["verdict"] for gate in card["gates"]} == {"G7": None}
    assert card["column"] is not None


def test_an_assistant_run_shows_g7_and_g8(judged: str) -> None:
    card = client.get(f"/quality/petro/card/{judged}/2.5").json()

    gates = {gate["gate"]: gate["verdict"] for gate in card["gates"]}
    assert gates == {"G7": "pass", "G8": "pass"}
    assert "Against its neighbours: water table 0.5 m off theirs (at most 1 m)." in _texts(card)
    assert [attempt["verdicts"] for attempt in card["attempts"]] == [{"G7": "pass", "G8": "pass"}]


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
    assert any(
        text.startswith("Its receiver at 3 m is left out, as of every window") for text in texts
    )
    assert "Shots left out: 1 too near." in texts
    card = client.get(f"/quality/run/{folder}").json()
    settings = {one["key"]: one for one in card["settings"]}
    assert (
        "The receiver at 3 m (2 of 2 records) is too weak or too strong"
        in settings["records"]["why"]
    )
    assert settings["reach"]["detail"] == "from the window's middle, out of its near field (3 m)"
    assert settings["reach"]["why"].startswith("nearest, a window stacks no shot nearer than that")
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

    metrics = {one.name: one for one in curve_metrics(curve, CurveThresholds(), 2.0, 15.0)}

    for name in ("aliased_points", "beyond_reach_points"):
        assert metrics[name].value == 0.167 and not metrics[name].passed


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
