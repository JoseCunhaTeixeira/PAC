"""The API on two synthetic profiles (tests/synthetic.py), page by page: a profile's settings,
a run, its dispersion images and picks, an inversion and its section. sigpipe's MASW layer
computes; these tests check what PAC asks of it and what the pages read back."""

import json
import shutil
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from masw.api.main import app
from masw.io.paths import output_folder
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters, pick_modes
from sigpipe.masw.inversion import InversionParameters, ThicknessLayer, VsLayer
from sigpipe.masw.runs import load_image

from .synthetic import N_RECEIVERS, SAMPLING, SOURCES

client = TestClient(app)


def _wait(job: dict[str, Any]) -> dict[str, Any]:
    for _ in range(600):
        job = client.get(f"/jobs/{job['id']}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.5)
    raise AssertionError("the job did not end")


def test_the_input_folder_lists_the_profiles() -> None:
    assert client.get("/input_folders").json() == ["noise", "shots"]


def test_a_profile_shows_its_records_and_receivers() -> None:
    shots = client.get("/acquisitions/shots").json()

    assert shots["files"] == sorted(SOURCES)
    assert shots["sampling_frequencies"] == [SAMPLING] * len(SOURCES)
    assert [x for x, _ in shots["source_positions"]] == [SOURCES[name] for name in sorted(SOURCES)]
    # The files' triggers: none said in these (MiniSEED; a SEG-2 file says its DELAY).
    assert shots["triggers"] == [None] * len(SOURCES)
    assert len(shots["receiver_positions"]) == N_RECEIVERS
    assert shots["modes"] == ["active", "passive-active"]
    noise = client.get("/acquisitions/noise").json()
    assert noise["source_positions"] == [] and noise["modes"] == ["passive"]
    assert client.get("/acquisitions/nope").status_code == 404


def test_a_form_starts_from_the_preset_fitted_to_its_profile() -> None:
    preset = client.get("/presets/passive-active", params={"profile": "shots"}).json()

    values, methods = preset["values"], preset["methods"]
    assert values["mode"] == "passive-active"
    assert values["masw"]["length"] == 5
    assert "correlation_window" not in values  # removed: the muting's velocities cut the same
    # Each method with its own values, those the profile derives filled in; a bound left out,
    # none (no stand-in value).
    assert methods["muting"]["mute"]["tmax"] is None
    assert methods["muting"]["mute"]["width"] == pytest.approx(1 / SAMPLING)  # one sample
    assert methods["filtering"]["iir"]["fmax"] == pytest.approx(0.95 * SAMPLING / 2)
    assert set(methods["stacking"]) == {"linear", "phase_weighted", "root"}
    refused = client.get("/presets/active", params={"profile": "noise"})
    assert refused.status_code == 422 and "does not fit passive profile" in refused.json()["detail"]


def test_the_windows_are_previewed_and_the_settings_checked() -> None:
    windows = client.post(
        "/windows", json={"profile": "shots", "masw": {"length": 6, "step": 3}}
    ).json()
    assert [window["xmid"] for window in windows] == [2.5, 5.5, 8.5]
    assert all(window["n_shots"] == 2 for window in windows)
    assert all(len(window["sources"]) == 2 for window in windows)

    request = {"profile": "shots", "mode": "active", "overrides": {"masw": {"length": 6}}}
    assert client.post("/config", json=request).json() == {"valid": True, "mode": "active"}
    request["overrides"] = {"masw": {"lenght": 6}}
    invalid = client.post("/config", json=request)
    assert (
        invalid.status_code == 422 and "masw.lenght: unknown parameter" in invalid.json()["detail"]
    )


@pytest.fixture(scope="module")
def run() -> str:
    """shots processed in three windows; its output folder."""
    request = {
        "profile": "shots",
        "mode": "active",
        "overrides": {"masw": {"length": 6, "step": 3}},
        "workers": 2,
    }
    started = client.post("/run", json=request)
    assert started.status_code == 202
    job = _wait(started.json())
    assert job["state"] == "succeeded" and job["errors"] == []
    assert (job["completed"], job["total"]) == (3, 3)
    assert job["run"].startswith("shots/")
    return job["run"]


def test_a_run_is_listed_first_and_holds_its_windows(run: str) -> None:
    assert client.get("/output_folders").json()[0] == run
    assert client.get(f"/xmids/{run}").json() == [2.5, 5.5, 8.5]
    # As the pages send it: encodeURIComponent turns the run's slash into %2F.
    assert client.get(f"/xmids/{quote(run, safe='')}").json() == [2.5, 5.5, 8.5]
    assert client.get("/xmids/..%2F..%2Fetc").status_code == 404
    # Nor does a pick write outside the output folder.
    box = {"fmin": 10, "fmax": 60, "vmin": 100, "vmax": 400, "label": "M0"}
    escape = client.post("/dispersion_images/..%2F..%2Ftmp/2.5/pick/box", json=box)
    assert escape.status_code == 404


def test_the_picks_replace_their_modes_curve(run: str) -> None:
    image = client.get(f"/dispersion_images/{run}/2.5").json()
    # The window's resolution limits: twice its spacing, and its length (6 receivers 1 m apart).
    assert image["curves"] == [] and (image["lambda_min"], image["lambda_max"]) == (2.0, 5.0)

    box = {"fmin": 10, "fmax": 60, "vmin": 100, "vmax": 400, "label": "M0"}
    picked = client.post(f"/dispersion_images/{run}/2.5/pick/box", json=box).json()
    (curve,) = picked["curves"]
    assert curve["label"] == "M0"
    assert 150 < sorted(curve["vs"])[len(curve["vs"]) // 2] < 250
    # Picked again with the lasso, M0 is replaced; M1 comes beside it.
    polygon = [(10.0, 120.0), (60.0, 120.0), (60.0, 300.0), (10.0, 300.0)]
    lasso = {"polygon": polygon, "label": "M0"}
    assert (
        len(client.post(f"/dispersion_images/{run}/2.5/pick/lasso", json=lasso).json()["curves"])
        == 1
    )
    box["label"], box["vmin"], box["vmax"] = "M1", 300, 800
    both = client.post(f"/dispersion_images/{run}/2.5/pick/box", json=box).json()
    assert [one["label"] for one in both["curves"]] == ["M0", "M1"]

    assert client.get(f"/dispersion_image_labels/{run}").json() == {"M0": 1, "M1": 1}
    left = client.delete(f"/dispersion_images/{run}/2.5/pick/M1").json()
    assert [one["label"] for one in left["curves"]] == ["M0"]
    assert client.delete(f"/dispersion_images/{run}/2.5/pick/M1").status_code == 404
    # Picked in PAC, by hand: the picking page says so.
    assert client.get(f"/dispersion_picks_by_position/{run}").json()[0] == {
        "xmid": 2.5,
        "labels": ["M0"],
        "picked_by": "hand",
    }
    section = client.get(f"/dispersion_pseudo_section/{run}/M0").json()
    assert section["positions"] == [2.5, 5.5, 8.5]


def test_a_windows_m0_is_picked_automatically(run: str) -> None:
    # A copy of the run, not to change the others' windows; its window 5.5 given an M0 and an M1
    # by hand.
    folder = f"{run}-auto"
    shutil.copytree(output_folder(run), output_folder(folder))
    box = {"fmin": 10, "fmax": 60, "vmin": 300, "vmax": 800, "label": "M1"}
    client.post(f"/dispersion_images/{folder}/5.5/pick/box", json=box)
    box |= {"vmin": 100, "vmax": 400, "label": "M0"}
    hand = client.post(f"/dispersion_images/{folder}/5.5/pick/box", json=box).json()

    auto = client.post(f"/dispersion_images/{folder}/5.5/pick/auto").json()

    # Its M0 replaced by PACo's (sigpipe's tracking picker, its defaults), its M1 kept; the window
    # said picked automatically.
    window = output_folder(folder) / "xmid_5.50"
    picker = pick_modes(load_image(window), PickingParameters())[0].curve
    assert picker is not None
    m0, m1 = auto["curves"]
    assert m0["label"] == "M0" and m0["fs"] == pytest.approx(picker.fs.tolist())
    assert m0["fs"] != hand["curves"][0]["fs"]
    assert m1 == hand["curves"][1]
    by_position = client.get(f"/dispersion_picks_by_position/{folder}").json()
    assert by_position[1] == {"xmid": 5.5, "labels": ["M0", "M1"], "picked_by": "auto"}
    # A curve edited by hand afterwards is the user's.
    client.delete(f"/dispersion_images/{folder}/5.5/pick/M1")
    assert client.get(f"/dispersion_picks_by_position/{folder}").json()[1]["picked_by"] == "hand"
    assert client.post(f"/dispersion_images/{folder}/9.5/pick/auto").status_code == 404


def test_the_inversion_form_starts_from_sigpipes_defaults() -> None:
    defaults = client.get("/inversion/defaults").json()

    parameters = defaults["parameters"]
    # sigpipe's, but the half-space up to 2,000 m/s, as PACo's priors start it.
    half_space = VsLayer(vs_max=2_000.0)
    # The data choosing the layers; the table's layers ready for the fixed layering.
    assert InversionParameters.model_validate(parameters) == InversionParameters(
        layering="free", vs_layers=(VsLayer(), half_space)
    )
    assert len(parameters["vs_layers"]) == parameters["n_layers"]
    assert defaults["vs_layer"] == VsLayer().model_dump()
    assert defaults["half_space_layer"] == half_space.model_dump()
    assert defaults["thickness_layer"] == ThicknessLayer().model_dump()


def test_an_inversion_gives_a_section(run: str) -> None:
    box = {"fmin": 10, "fmax": 60, "vmin": 100, "vmax": 400, "label": "M0"}
    for xmid in (2.5, 5.5):
        assert client.post(f"/dispersion_images/{run}/{xmid}/pick/box", json=box).status_code == 200
    config = {
        "folder": run,
        "positions": [2.5, 5.5],
        "labels": ["M0"],
        "parameters": {
            "n_layers": 2,
            "vs_layers": [{"vs_min": 100, "vs_max": 400, "vs_perturb_std": 20}] * 2,
            "thickness_layers": [
                {"thickness_min": 1, "thickness_max": 5, "thickness_perturb_std": 1}
            ],
            "n_iterations": 1_000,
            "n_burnin_iterations": 100,
            "n_chains": 2,
        },
        "n_workers": 2,
    }

    job = _wait(client.post("/inversion/run", json=config).json())

    assert job["state"] == "succeeded" and job["errors"] == []
    status = client.get(f"/inversion/status/{run}").json()
    assert [one["has_result"] for one in status] == [True, True, False]
    section = client.get(f"/inversion/velocity_section/{run}").json()
    assert section["positions"] == [2.5, 5.5]
    assert all(100 <= vs <= 400 for row in section["vs_grid"] for vs in row if vs is not None)
    # Each window's column: how deep its data inform its model, of the depth modelled, as the
    # job measured it (saved beside the model).
    assert [one["x"] for one in section["windows"]] == [2.5, 5.5]
    assert all(0 <= one["informed"] <= one["depth"] for one in section["windows"])
    assert (output_folder(run) / "xmid_2.50" / "SeismicInversion_Measures_0000.json").exists()
    assert len(section["informed_levels"]) == len(section["positions"])
    # The spread in % of Vs, and where the kept models put interfaces (the job counted them).
    assert all(0 <= std < 200 for row in section["vs_std_grid"] for std in row if std is not None)
    shares = [one for row in section["interface_grid"] for one in row if one is not None]
    assert shares and all(0 <= one <= 100 for one in shares)
    smoothed = client.get(f"/inversion/velocity_section/{run}", params={"lateral_smoothing": True})
    assert smoothed.status_code == 200
    # Smoothed as the section: a level per column of its finer grid.
    assert len(smoothed.json()["informed_levels"]) == len(smoothed.json()["positions"]) > 2
    curves = client.get(f"/inversion/curves/{run}/M0").json()
    assert [one["predicted_fs"] is not None for one in curves] == [True, True, False]
    comparison = client.get(f"/inversion/pseudo_section_comparison/{run}/M0").json()
    assert comparison["positions"] == [2.5, 5.5]
    # Along wavelength too, as many rows.
    lambdas = comparison["lambdas"]
    assert len(lambdas) == len(comparison["fs"]) and lambdas == sorted(lambdas)
    assert len(comparison["observed_by_wavelength_grid"][0]) == len(lambdas)


def test_a_window_done_again_by_hand_keeps_nothing_older(run: str) -> None:
    # After the inversion above: 2.5 and 5.5 inverted with M0. 2.5 as the assistant leaves a
    # window it retried: an attempt archived, its lines in the QC log, a file of its own.
    run_folder = output_folder(run)
    window = run_folder / "xmid_2.50"
    archive = window / "attempts" / "1_inversion"
    archive.mkdir(parents=True)
    (archive / "SeismicInversion_Samples_0000.npz").write_text("its first models")
    (window / "SeismicInversion_Model_0000_old.csv").write_text("an older run's model")
    line = {"parameters": {}, "started_at": "2026-09-26T18:40:12Z", "status": "succeeded"}
    lines = [
        {"unit": "xmid_2.50", "stage": "inversion", "attempt": 1, "triggered_by": "initial"},
        {"unit": "xmid_2.50", "stage": "inversion", "attempt": 2, "triggered_by": "G5:steps"},
        {"unit": "xmid_5.50", "stage": "inversion", "attempt": 1, "triggered_by": "initial"},
    ]
    log = run_folder / "qc_log.jsonl"
    log.write_text("".join(json.dumps(line | one) + "\n" for one in lines))
    config = {
        "folder": run,
        "positions": [2.5],
        "labels": ["M0"],
        "parameters": {"n_iterations": 1_000, "n_burnin_iterations": 100, "n_chains": 2},
        "n_workers": 1,
    }

    assert _wait(client.post("/inversion/run", json=config).json())["state"] == "succeeded"

    # Inverted again by hand: the window's earlier attempts gone, its lines in the log with them
    # (5.5's kept), no older file beside the new ones; its card, by hand.
    assert not (window / "attempts").exists()
    assert not (window / "SeismicInversion_Model_0000_old.csv").exists()
    assert (window / "SeismicInversion_Samples_0000.npz").exists()
    assert [json.loads(one)["unit"] for one in log.read_text().splitlines()] == ["xmid_5.50"]
    card = client.get(f"/quality/inversion/card/{run}/2.5").json()
    assert card["attempts"] == [] and any(
        one["text"].startswith("Inverted by hand") for one in card["sentences"]
    )
    section = run_folder / "SeismicInversion_VelocitySection_0000.png"
    assert section.exists()

    # Another mode picked, then deleted: the inversion, of M0, kept.
    box = {"fmin": 10, "fmax": 60, "vmin": 300, "vmax": 800, "label": "M1"}
    assert client.post(f"/dispersion_images/{run}/2.5/pick/box", json=box).status_code == 200
    assert client.delete(f"/dispersion_images/{run}/2.5/pick/M1").status_code == 200
    assert (window / "SeismicInversion_Samples_0000.npz").exists()
    # M0 picked again: the inversion made of the old curve erased, the line's section with it.
    box.update({"vmin": 100, "vmax": 400, "label": "M0"})
    assert client.post(f"/dispersion_images/{run}/2.5/pick/box", json=box).status_code == 200
    assert not any(window.glob("SeismicInversion_*")) and not section.exists()
    assert (window / "DispersionCurves_0000.csv").exists()
    status = client.get(f"/inversion/status/{run}").json()
    assert [one["has_result"] for one in status] == [False, True, False]
    log.unlink()


def test_a_petrophysical_inversion_gives_its_sections(run: str) -> None:
    box = {"fmin": 10, "fmax": 60, "vmin": 100, "vmax": 400, "label": "M0"}
    for xmid in (2.5, 5.5):
        assert client.post(f"/dispersion_images/{run}/{xmid}/pick/box", json=box).status_code == 200
    (model,) = client.get("/petro_inversion/models").json()
    config = {"folder": run, "positions": [2.5, 5.5], "model_name": model, "n_workers": 2}

    job = _wait(client.post("/petro_inversion/run", json=config).json())

    assert job["state"] == "succeeded" and job["errors"] == []
    status = client.get(f"/petro_inversion/status/{run}").json()
    assert [one["has_result"] for one in status] == [True, True, False]
    section = client.get(f"/petro_inversion/section/{run}").json()
    assert section["positions"] == [2.5, 5.5]
    soils = {soil for row in section["soil_grid"] for soil in row if soil is not None}
    assert soils and soils <= {"clay", "loam", "silt", "sand"}
    for quantity in ("shear_modulus_section", "vs_section"):
        grid = client.get(f"/petro_inversion/{quantity}/{run}").json()
        assert grid["positions"] == [2.5, 5.5]
        assert any(value is not None for row in grid["values"] for value in row)
    curves = client.get(f"/petro_inversion/curves/{run}").json()
    assert [one["predicted_fs"] is not None for one in curves] == [True, True, False]
    comparison = client.get(f"/petro_inversion/pseudo_section_comparison/{run}").json()
    assert comparison["positions"] == [2.5, 5.5]


# Stopping: at once, what finished kept, nothing half-written (sigpipe.masw.runs.stopping).


def _once(check: Callable[[dict[str, Any]], bool], job: dict[str, Any]) -> dict[str, Any]:
    """Job `job` once `check` holds of it."""
    for _ in range(1200):
        job = client.get(f"/jobs/{job['id']}").json()
        if check(job):
            return job
        time.sleep(0.05)
    raise AssertionError("the job never got there")


def test_a_stopped_run_keeps_the_windows_that_finished() -> None:
    request = {
        "profile": "shots",
        "mode": "active",
        "overrides": {"masw": {"length": 6, "step": 1}},
        "workers": 1,
    }
    job = _once(lambda one: one["completed"] >= 1, client.post("/run", json=request).json())

    asked = client.post(f"/jobs/{job['id']}/stop").json()
    job = _wait(job)

    assert asked["state"] == "stopped" or asked["stopping"]
    assert job["state"] == "stopped" and not job["stopping"]
    folder = output_folder(job["run"])
    manifest = json.loads((folder / "run.json").read_text())
    kept = {window["folder"] for window in manifest["windows"]}
    assert manifest["stopped"] and 1 <= len(kept) < job["total"]
    assert {path.name for path in folder.glob("xmid_*")} == kept
    assert not list(folder.rglob(".partial"))
    assert any(one["id"] == job["id"] for one in client.get("/jobs").json())


def test_a_job_stopped_before_it_starts_never_runs() -> None:
    request = {
        "profile": "shots",
        "mode": "active",
        "overrides": {"masw": {"length": 6, "step": 1}},
        "workers": 1,
    }
    first = client.post("/run", json=request).json()
    queued = client.post("/run", json=request).json()

    stopped = client.post(f"/jobs/{queued['id']}/stop").json()
    client.post(f"/jobs/{first['id']}/stop")

    assert (stopped["state"], stopped["elapsed"], stopped["run"]) == ("stopped", 0.0, None)
    assert _wait(first)["state"] == "stopped"
    assert client.post("/jobs/nope/stop").status_code == 404


def test_a_stopped_inversion_leaves_its_windows_as_they_were(run: str) -> None:
    box = {"fmin": 10, "fmax": 60, "vmin": 100, "vmax": 400, "label": "M0"}
    assert client.post(f"/dispersion_images/{run}/8.5/pick/box", json=box).status_code == 200
    window = output_folder(run) / "xmid_8.50"
    before = window / "SeismicInversion_Log_0000.log"
    before.write_text("an earlier inversion\n")
    files = sorted(path.name for path in window.iterdir())
    config = {
        "folder": run,
        "positions": [8.5],
        "labels": ["M0"],
        "parameters": {
            "n_layers": 2,
            "vs_layers": [{"vs_min": 100, "vs_max": 400, "vs_perturb_std": 20}] * 2,
            "thickness_layers": [
                {"thickness_min": 1, "thickness_max": 5, "thickness_perturb_std": 1}
            ],
            "n_iterations": 2_000_000,  # minutes: stopped long before its end
            "n_burnin_iterations": 1_000,
            "n_chains": 2,
        },
        "n_workers": 1,
    }
    job = client.post("/inversion/run", json=config).json()
    time.sleep(3)  # the position's worker inverting

    client.post(f"/jobs/{job['id']}/stop")
    job = _wait(job)

    assert job["state"] == "stopped" and job["elapsed"] < 60
    assert sorted(path.name for path in window.iterdir()) == files  # no staging, nothing new
    assert before.read_text() == "an earlier inversion\n"
    outcome = json.loads((output_folder(run) / "seismic_inversion_outcome.json").read_text())
    assert outcome == []
