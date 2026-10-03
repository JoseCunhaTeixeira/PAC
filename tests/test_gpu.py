"""The install step's GPU check (masw.gpu) on fake machines: what nvidia-smi and docker print,
and the sysfs tree of an AMD card."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from masw.gpu import (
    MODEL,
    Runner,
    check,
    main,
    remote_settings,
    write_env,
)

DOCKER_WITH_NVIDIA = '{"nvidia":{"path":"nvidia-container-runtime"},"runc":{"path":"runc"}}'
DOCKER_PLAIN = '{"runc":{"path":"runc"}}'


def _machine(nvidia: str | None = None, docker: str | None = DOCKER_WITH_NVIDIA) -> Runner:
    def run(command: Sequence[str]) -> str | None:
        return {"nvidia-smi": nvidia, "docker": docker}.get(command[0])

    return run


def _amd(root: Path, vram_bytes: int, kfd: bool = True) -> tuple[Path, Path]:
    """The sysfs tree of an AMD card of `vram_bytes`, and /dev."""
    drm = root / "drm"
    device = drm / "card1" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "device").write_text("0x7550\n")
    (device / "mem_info_vram_total").write_text(f"{vram_bytes}\n")
    (drm / "card1-DP-1").mkdir()  # a connector, not a card
    dev = root / "dev"
    dev.mkdir()
    if kfd:
        (dev / "kfd").touch()
    return drm, dev


def test_a_24_gb_card_runs_the_model(tmp_path: Path) -> None:
    runner = _machine("NVIDIA GeForce RTX 4090, 24564, 8.9\n")

    verdict = check("Linux", runner, drm=tmp_path, dev=tmp_path)

    assert verdict.compatible
    assert verdict.settings == {
        "PAC_EXTRAS": "agent",
        "COMPOSE_PROFILES": "agent",
        "PACO_LLM_MODEL": "Qwen/Qwen3-14B-FP8",
        "VLLM_MAX_MODEL_LEN": "16384",
    }


def test_a_card_too_small_or_without_fp8_is_not_compatible(tmp_path: Path) -> None:
    small = check("Linux", _machine("NVIDIA RTX A4000, 16376, 8.6\n"), drm=tmp_path, dev=tmp_path)
    turing = check("Linux", _machine("Quadro RTX 6000, 24576, 7.5\n"), drm=tmp_path, dev=tmp_path)

    assert not small.compatible and small.settings == {}
    assert small.reason == (
        "NVIDIA RTX A4000 (16.0 GiB): Qwen/Qwen3-14B-FP8 needs a GPU of 24 GB or more."
    )
    assert not turing.compatible and "Ampere" in turing.reason


def test_docker_must_reach_the_gpu(tmp_path: Path) -> None:
    runner = _machine("NVIDIA GeForce RTX 4090, 24564, 8.9\n", docker=DOCKER_PLAIN)
    without_toolkit = check("Linux", runner, drm=tmp_path, dev=tmp_path)
    nvidia = _machine("NVIDIA GeForce RTX 4090, 24564, 8.9\n", docker=None)
    without_docker = check("Linux", nvidia)

    assert not without_toolkit.compatible and "Container Toolkit" in without_toolkit.reason
    assert not without_docker.compatible and "Docker does not answer" in without_docker.reason
    # Docker Desktop reaches NVIDIA GPUs on Windows by itself.
    assert check("Windows", runner, drm=tmp_path, dev=tmp_path).compatible


def test_an_amd_24_gb_card_runs_the_model_with_rocms_override(tmp_path: Path) -> None:
    drm, dev = _amd(tmp_path, 25_753_026_560)

    verdict = check("Linux", _machine(docker=DOCKER_PLAIN), drm=drm, dev=dev)

    assert verdict.compatible, verdict.reason
    assert verdict.gpu is not None and verdict.gpu.vram_gib == pytest.approx(23.98, abs=0.01)
    assert verdict.settings["PACO_LLM_MODEL"] == "Qwen/Qwen3-14B-FP8"
    assert verdict.settings["COMPOSE_FILE"] == "docker-compose.yml:docker-compose.rocm.yml"
    no_rocm = check(
        "Linux",
        _machine(docker=DOCKER_PLAIN),
        drm=_amd(tmp_path / "b", 25_753_026_560, kfd=False)[0],
        dev=tmp_path / "b" / "dev",
    )
    assert not no_rocm.compatible and "/dev/kfd" in no_rocm.reason
    sixteen = check(
        "Linux", _machine(docker=DOCKER_PLAIN), drm=_amd(tmp_path / "c", 17_095_983_104)[0]
    )
    assert not sixteen.compatible and "24 GB or more" in sixteen.reason


def test_no_gpu_or_macos_is_not_compatible(tmp_path: Path) -> None:
    nvidia = _machine("NVIDIA GeForce RTX 4090, 24564, 8.9\n")
    assert "No GPU found" in check("Linux", _machine(), drm=tmp_path, dev=tmp_path).reason
    assert "macOS" in check("Darwin", nvidia).reason


def test_the_env_file_keeps_the_users_lines(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=secret\nPACO_LLM_MODEL=old\nCOMPOSE_PROFILES=agent\n")

    write_env(env, {"PAC_EXTRAS": "agent", "PACO_LLM_MODEL": MODEL})
    assert env.read_text() == f"HF_TOKEN=secret\nPAC_EXTRAS=agent\nPACO_LLM_MODEL={MODEL}\n"
    # Without the assistant, its lines go.
    write_env(env, {})
    assert env.read_text() == "HF_TOKEN=secret\n"


def test_a_remote_model_is_checked_first() -> None:
    asked: list[tuple[str, str]] = []

    def served(url: str, api_key: str) -> list[str]:
        asked.append((url, api_key))
        return [MODEL]

    settings, model = remote_settings("https://gpu.example.org/v1", None, "key", served)

    assert model == MODEL and asked == [("https://gpu.example.org/v1", "key")]
    assert settings == {
        "PAC_EXTRAS": "agent",
        "PACO_LLM_BASE_URL": "https://gpu.example.org/v1",
        "PACO_LLM_MODEL": MODEL,
        "PACO_LLM_API_KEY": "key",
    }
    with pytest.raises(OSError, match="does not serve another/model"):
        remote_settings("https://gpu.example.org/v1", "another/model", None, served)
    # This machine's own address is out of the container's reach.
    with pytest.raises(OSError, match="--tunnel"):
        remote_settings("http://127.0.0.1:8001/v1", None, None, served)


def test_a_remote_server_that_does_not_answer_installs_nothing(tmp_path: Path) -> None:
    env = tmp_path / ".env"

    status = main(["--remote", "http://127.0.0.1:9/v1", "--env", str(env)])  # nothing listens

    assert status == 1 and not env.exists()
    assert main(["--without", "--env", str(env)]) == 0 and env.read_text() == ""
