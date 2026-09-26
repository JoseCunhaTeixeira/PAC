"""The API on two synthetic profiles (tests/synthetic.py), page by page: a profile's settings,
a run, its dispersion images and picks, an inversion and its section. sigpipe's MASW layer
computes; these tests check what PAC asks of it and what the pages read back."""

import time
from typing import Any
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from masw.api.main import app
from sigpipe.masw.inversion import InversionParameters, ThicknessLayer, VsLayer

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
    assert values["correlation_window"] == {
        "method": "mute",
        "vmin": 80.0,
        "vmax": 1500.0,
        "taper": 50,
    }
    # Each method with its own values, those the profile derives filled in.
    assert methods["muting"]["mute"]["tmax"] == pytest.approx(1.0, abs=0.01)
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
    assert client.get(f"/dispersion_picks_by_position/{run}").json()[0] == {
        "xmid": 2.5,
        "labels": ["M0"],
    }
    section = client.get(f"/dispersion_pseudo_section/{run}/M0").json()
    assert section["positions"] == [2.5, 5.5, 8.5]


def test_the_inversion_form_starts_from_sigpipes_defaults() -> None:
    defaults = client.get("/inversion/defaults").json()

    parameters = defaults["parameters"]
    assert InversionParameters.model_validate(parameters) == InversionParameters()
    assert len(parameters["vs_layers"]) == parameters["n_layers"]
    assert defaults["vs_layer"] == VsLayer().model_dump()
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
    smoothed = client.get(f"/inversion/velocity_section/{run}", params={"lateral_smoothing": True})
    assert smoothed.status_code == 200
    curves = client.get(f"/inversion/curves/{run}/M0").json()
    assert [one["predicted_fs"] is not None for one in curves] == [True, True, False]
    comparison = client.get(f"/inversion/pseudo_section_comparison/{run}/M0").json()
    assert comparison["positions"] == [2.5, 5.5]
    saved = client.post(f"/inversion/save_images/{run}", json={"labels": ["M0"]}).json()
    assert saved["errors"] == [] and len(saved["saved_paths"]) == 2


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
