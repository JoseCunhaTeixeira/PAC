"""Smoke tests: the FastAPI app builds and its basic endpoints respond.

These don't exercise MASW/inversion logic (that needs real acquisition
data), just that the app wires up -- routers import, dependencies resolve,
and the server answers requests.
"""

from fastapi.testclient import TestClient

from masw.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version() -> None:
    response = client.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()


def test_openapi_schema_builds() -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
