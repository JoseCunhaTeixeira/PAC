"""The install step's GPU check (masw.gpu) on fake machines: what nvidia-smi and docker print,
and the sysfs tree of an AMD card."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from masw.gpu import (
    MODELS,
    RECOMMENDED,
    SMALLER,
    Runner,
    check,
    chosen_model,
    main,
    remote_settings,
    write_env,
)

FOURTEEN, EIGHT = MODELS[RECOMMENDED], MODELS[SMALLER]

DOCKER_WITH_NVIDIA = '{"nvidia":{"path":"nvidia-container-runtime"},"runc":{"path":"runc"}}'
DOCKER_PLAIN = '{"runc":{"path":"runc"}}'


def _machine(nvidia: str | None = None, docker: str | None = DOCKER_WITH_NVIDIA) -> Runner:
    def run(command: Sequence[str]) -> str | None:
        return {"nvidia-smi": nvidia, "docker": docker}.get(command[0])

    return run


def _amd(root: Path, vram_bytes: int, kfd: bool = True) -> tuple[Path, Path]:
    """The sysfs tree of an AMD card (a Radeon RX 9070: 17.1e9 bytes), and /dev."""
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


def test_a_24_gb_card_runs_the_model_chosen(tmp_path: Path) -> None:
    runner = _machine("NVIDIA GeForce RTX 4090, 24564, 8.9\n")

    verdict = check(FOURTEEN, "Linux", runner, drm=tmp_path, dev=tmp_path)

    assert verdict.compatible
    assert verdict.settings == {
        "PAC_EXTRAS": "agent",
        "COMPOSE_PROFILES": "agent",
        "PACO_LLM_MODEL": "Qwen/Qwen3-14B-FP8",
        "VLLM_MAX_MODEL_LEN": "16384",
    }
    smaller = check(EIGHT, "Linux", runner, drm=tmp_path, dev=tmp_path)
    assert smaller.settings["PACO_LLM_MODEL"] == "Qwen/Qwen3-8B-FP8"


def test_a_16_gb_card_runs_the_smaller_model_where_the_gpu_supports_fp8(tmp_path: Path) -> None:
    ampere = _machine("NVIDIA RTX A4000, 16376, 8.6\n")
    turing = check(EIGHT, "Linux", _machine("Tesla T4, 15360, 7.5\n"), drm=tmp_path, dev=tmp_path)

    assert check(EIGHT, "Linux", ampere, drm=tmp_path, dev=tmp_path).compatible
    assert not turing.compatible and "Ampere" in turing.reason
    # The recommended model does not fit: said, with the one that does, never chosen for them.
    too_small = check(FOURTEEN, "Linux", ampere, drm=tmp_path, dev=tmp_path)
    assert not too_small.compatible and too_small.settings == {}
    assert too_small.reason == (
        "NVIDIA RTX A4000 (16.0 GiB): Qwen/Qwen3-14B-FP8 needs a GPU of 24 GB or more. It runs "
        "Qwen/Qwen3-8B-FP8 (less performant)."
    )


def test_a_small_card_is_not_compatible(tmp_path: Path) -> None:
    verdict = check(EIGHT, "Linux", _machine("NVIDIA GeForce RTX 3060, 12288, 8.6\n"), drm=tmp_path)

    assert not verdict.compatible and "16 GB or more" in verdict.reason
    assert verdict.settings == {}


def test_docker_must_reach_the_gpu(tmp_path: Path) -> None:
    runner = _machine("NVIDIA GeForce RTX 4090, 24564, 8.9\n", docker=DOCKER_PLAIN)
    without_toolkit = check(FOURTEEN, "Linux", runner, drm=tmp_path, dev=tmp_path)
    nvidia = _machine("NVIDIA GeForce RTX 4090, 24564, 8.9\n", docker=None)
    without_docker = check(FOURTEEN, "Linux", nvidia)

    assert not without_toolkit.compatible and "Container Toolkit" in without_toolkit.reason
    assert not without_docker.compatible and "Docker does not answer" in without_docker.reason
    # Docker Desktop reaches NVIDIA GPUs on Windows by itself.
    assert check(FOURTEEN, "Windows", runner, drm=tmp_path, dev=tmp_path).compatible


def test_an_amd_16_gb_card_runs_the_smaller_model_with_rocms_override(tmp_path: Path) -> None:
    drm, dev = _amd(tmp_path, 17_095_983_104)

    verdict = check(EIGHT, "Linux", _machine(docker=DOCKER_PLAIN), drm=drm, dev=dev)

    assert verdict.compatible, verdict.reason
    assert verdict.gpu is not None and verdict.gpu.vram_gib == pytest.approx(15.92, abs=0.01)
    assert verdict.settings["PACO_LLM_MODEL"] == "Qwen/Qwen3-8B-FP8"
    assert verdict.settings["VLLM_MAX_MODEL_LEN"] == "12288"
    assert verdict.settings["COMPOSE_FILE"] == "docker-compose.yml:docker-compose.rocm.yml"
    no_rocm = check(
        EIGHT,
        "Linux",
        _machine(docker=DOCKER_PLAIN),
        drm=_amd(tmp_path / "b", 17_095_983_104, kfd=False)[0],
        dev=tmp_path / "b" / "dev",
    )
    assert not no_rocm.compatible and "/dev/kfd" in no_rocm.reason


def test_no_gpu_or_macos_is_not_compatible(tmp_path: Path) -> None:
    nvidia = _machine("NVIDIA GeForce RTX 4090, 24564, 8.9\n")
    assert "No GPU found" in check(FOURTEEN, "Linux", _machine(), drm=tmp_path, dev=tmp_path).reason
    assert "macOS" in check(FOURTEEN, "Darwin", nvidia).reason


def test_the_model_is_the_users_choice() -> None:
    def answers(reply: str) -> Callable[[str], str]:
        return lambda _question: reply

    assert chosen_model(None, answers("1")) == FOURTEEN
    assert chosen_model(None, answers("2")) == EIGHT
    # No answer, or another, chooses none: nothing is chosen for the user.
    assert chosen_model(None, answers("")) is None
    assert chosen_model(None, answers("3")) is None
    assert chosen_model("Qwen/Qwen3-14B-FP8") == FOURTEEN
    assert chosen_model("Qwen/Qwen3-4B") is None


def test_no_model_chosen_installs_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = tmp_path / ".env"

    assert main(["--with", "--model", "Qwen/Qwen3-4B", "--env", str(env)]) == 1

    assert not env.exists()
    assert "Not installed: choose a model, 1 or 2" in capsys.readouterr().out


def test_the_env_file_keeps_the_users_lines(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=secret\nPACO_LLM_MODEL=old\nCOMPOSE_PROFILES=agent\n")

    write_env(env, {"PAC_EXTRAS": "agent", "PACO_LLM_MODEL": SMALLER})
    assert env.read_text() == f"HF_TOKEN=secret\nPAC_EXTRAS=agent\nPACO_LLM_MODEL={SMALLER}\n"
    # Without the assistant, its lines go.
    write_env(env, {})
    assert env.read_text() == "HF_TOKEN=secret\n"


def test_a_remote_model_is_checked_first() -> None:
    asked: list[tuple[str, str]] = []

    def served(url: str, api_key: str) -> list[str]:
        asked.append((url, api_key))
        return [SMALLER]

    settings, model = remote_settings("https://gpu.example.org/v1", None, "key", served)

    assert model == SMALLER and asked == [("https://gpu.example.org/v1", "key")]
    assert settings == {
        "PAC_EXTRAS": "agent",
        "PACO_LLM_BASE_URL": "https://gpu.example.org/v1",
        "PACO_LLM_MODEL": SMALLER,
        "PACO_LLM_API_KEY": "key",
    }
    with pytest.raises(OSError, match="does not serve Qwen/Qwen3-4B"):
        remote_settings("https://gpu.example.org/v1", "Qwen/Qwen3-4B", None, served)
    # This machine's own address is out of the container's reach.
    with pytest.raises(OSError, match="--tunnel"):
        remote_settings("http://127.0.0.1:8001/v1", None, None, served)


def test_a_remote_server_that_does_not_answer_installs_nothing(tmp_path: Path) -> None:
    env = tmp_path / ".env"

    status = main(["--remote", "http://127.0.0.1:9/v1", "--env", str(env)])  # nothing listens

    assert status == 1 and not env.exists()
    assert main(["--without", "--env", str(env)]) == 0 and env.read_text() == ""
