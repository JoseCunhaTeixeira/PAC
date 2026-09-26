"""The inversion form's parameters are checked before any job starts."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from masw.api.main import app
from sigpipe.masw.inversion import InversionParameters

client = TestClient(app)


def _config(n_iterations: int, n_burnin_iterations: int) -> dict[str, object]:
    """A run config as the inversion form sends it, with the form's default layers."""
    return {
        "folder": "demo",
        "positions": [10.0],
        "labels": ["M0"],
        "parameters": {
            "n_layers": 2,
            "vs_layers": [{"vs_min": 100, "vs_max": 1000, "vs_perturb_std": 20}] * 2,
            "thickness_layers": [
                {"thickness_min": 1, "thickness_max": 10, "thickness_perturb_std": 1}
            ],
            "n_iterations": n_iterations,
            "n_burnin_iterations": n_burnin_iterations,
            "n_chains": 5,
        },
        "n_workers": 1,
    }


def test_a_burnin_that_leaves_nothing_to_sample_is_refused() -> None:
    # 2,000 iterations with the form's default burn-in of 10,000 leave nothing to sample.
    response = client.post("/inversion/run", json=_config(2_000, 10_000))

    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "parameters"]
    assert (
        "n_iterations (2000) must exceed n_burnin_iterations (10000) by at least 150"
        in error["msg"]
    )


def test_150_iterations_after_the_burnin_are_enough() -> None:
    parameters = _config(300, 150)["parameters"]

    InversionParameters.model_validate(parameters)
    with pytest.raises(ValidationError, match="by at least 150"):
        InversionParameters.model_validate(_config(299, 150)["parameters"])
