"""Whether this machine's GPU can serve the assistant's model, and the settings that serve it: what
PAC's install step (install_assistant.py) checks before it installs the assistant. The standard
library only: it runs on the host, before PAC is installed.

Compatible: an NVIDIA GPU (CUDA, compute capability 8.0 or newer) or an AMD one (ROCm, Linux) with
16 GB or more, which Docker can reach. It serves Qwen3-8B in FP8 with a 12k context; from 24 GB,
Qwen3-8B in full precision. Below 16 GB, not compatible: the smaller models that fit, such as
Qwen3-4B, are too weak for PACo.

Python 3.9 or newer: the host's python3, not PAC's."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

GIB = 2**30
# A 16 GB card reports a little less: the RX 9070's 17.1e9 bytes are 15.9 GiB.
MIN_VRAM_GIB = 15.0
FULL_VRAM_GIB = 23.0
FP8_MODEL = "Qwen/Qwen3-8B-FP8"
FULL_MODEL = "Qwen/Qwen3-8B"
FP8_CONTEXT = 12_288  # tokens that fit beside the FP8 weights on 16 GB
FULL_CONTEXT = 16_384
# vLLM's FP8 kernels on NVIDIA start with Ampere.
MIN_CUDA_CAPABILITY = 8.0
# The .env lines the assistant owns: set when it is installed, removed when it is not.
AGENT_KEYS = (
    "PAC_EXTRAS",
    "COMPOSE_PROFILES",
    "COMPOSE_FILE",
    "PACO_LLM_MODEL",
    "PACO_LLM_BASE_URL",
    "PACO_LLM_API_KEY",
    "VLLM_MAX_MODEL_LEN",
    "MODEL_SSH",
    "MODEL_SSH_PORT",
    "MODEL_SSH_KEY",
    "MODEL_SSH_KNOWN_HOSTS",
)
# The keys the tunnel tries, first a key made for it (the docs show how).
SSH_KEYS = ("pac_tunnel", "id_ed25519", "id_ecdsa", "id_rsa")

# A command's output, or None when it is missing or fails.
Runner = Callable[[Sequence[str]], Optional[str]]


@dataclass(frozen=True)
class Gpu:
    vendor: str  # "nvidia" or "amd"
    name: str
    vram_gib: float
    compute_capability: float | None = None  # NVIDIA's


@dataclass(frozen=True)
class Verdict:
    compatible: bool
    reason: str
    gpu: Gpu | None = None
    settings: dict[str, str] = field(default_factory=dict[str, str])  # the .env lines


def run(command: Sequence[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        done = subprocess.run(
            list(command), capture_output=True, text=True, timeout=30, check=False
        )
    except OSError:
        return None
    except subprocess.TimeoutExpired:
        return None
    return done.stdout if done.returncode == 0 else None


def nvidia_gpus(runner: Runner = run) -> list[Gpu]:
    """The NVIDIA GPUs nvidia-smi (installed with NVIDIA's driver) lists."""
    output = runner(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[Gpu] = []
    for line in (output or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, mib, capability = parts
        try:
            gpus.append(Gpu("nvidia", name, float(mib) / 1024, float(capability)))
        except ValueError:
            continue
    return gpus


def amd_gpus(drm: Path = Path("/sys/class/drm"), runner: Runner = run) -> list[Gpu]:
    """The AMD GPUs the amdgpu driver exposes on Linux, with their memory (no ROCm tool needed on
    the host: vLLM's container brings its own)."""
    gpus: list[Gpu] = []
    cards = sorted(path for path in drm.glob("card*") if re.fullmatch(r"card\d+", path.name))
    for card in cards:
        device = card / "device"
        vram = _read(device / "mem_info_vram_total")
        if _read(device / "vendor") != "0x1002" or vram is None or not vram.isdigit():
            continue
        described = runner(["lspci", "-s", device.resolve().name])
        if described and ": " in described:
            name = described.split(": ", 1)[1].strip()
        else:
            name = f"AMD GPU {_read(device / 'device') or ''}".strip()
        gpus.append(Gpu("amd", name, int(vram) / GIB))
    return gpus


def check(
    system: str | None = None,
    runner: Runner = run,
    drm: Path = Path("/sys/class/drm"),
    dev: Path = Path("/dev"),
) -> Verdict:
    """Whether the assistant's model can run on this machine's GPU, through Docker."""
    system = system or platform.system()
    if system == "Darwin":
        return Verdict(
            False,
            "Docker on macOS cannot reach the GPU: the assistant's model needs Linux or Windows "
            "with an NVIDIA or AMD GPU of 16 GB or more.",
        )
    gpus = nvidia_gpus(runner) + (amd_gpus(drm, runner) if system == "Linux" else [])
    if not gpus:
        return Verdict(
            False,
            "No GPU found that could serve the assistant's model: an NVIDIA GPU (with its "
            "driver's nvidia-smi) or, on Linux, an AMD one (amdgpu driver), of 16 GB or more.",
        )
    gpu = max(gpus, key=lambda one: one.vram_gib)
    about = f"{gpu.name} ({gpu.vram_gib:.1f} GiB)"
    if gpu.vram_gib < MIN_VRAM_GIB:
        return Verdict(False, f"{about}: the assistant's model needs a GPU of 16 GB or more.", gpu)
    full = gpu.vram_gib >= FULL_VRAM_GIB
    capability = gpu.compute_capability
    if gpu.vendor == "nvidia" and not full and (capability or 0) < MIN_CUDA_CAPABILITY:
        return Verdict(
            False,
            f"{about}, compute capability {capability}: on 16 GB the model runs in FP8, which "
            "needs an Ampere GPU or newer (compute capability 8.0).",
            gpu,
        )
    if missing := _docker_misses(gpu, runner, dev, system):
        return Verdict(False, f"{about} fits the model, but {missing}.", gpu)
    model = FULL_MODEL if full else FP8_MODEL
    settings = {
        "PAC_EXTRAS": "agent",
        "COMPOSE_PROFILES": "agent",
        "PACO_LLM_MODEL": model,
        "VLLM_MAX_MODEL_LEN": str(FULL_CONTEXT if full else FP8_CONTEXT),
    }
    if gpu.vendor == "amd":
        settings["COMPOSE_FILE"] = "docker-compose.yml:docker-compose.rocm.yml"
    return Verdict(True, f"{about}: compatible, the assistant will run {model}.", gpu, settings)


def _docker_misses(gpu: Gpu, runner: Runner, dev: Path, system: str) -> str | None:
    """What keeps Docker from handing `gpu` to a container; None when nothing does."""
    runtimes = runner(["docker", "info", "--format", "{{json .Runtimes}}"])
    if runtimes is None:
        return "Docker does not answer: install it and start it"
    if gpu.vendor == "amd":
        if not (dev / "kfd").exists():
            return "ROCm's /dev/kfd is missing: the amdgpu driver with ROCm support is needed"
        return None
    # Docker Desktop on Windows reaches NVIDIA GPUs through WSL2 by itself.
    if system == "Linux" and "nvidia" not in runtimes:
        return "Docker cannot reach NVIDIA GPUs: install NVIDIA's Container Toolkit"
    return None


def served_models(url: str, api_key: str = "EMPTY", timeout: float = 10.0) -> list[str]:
    """The models the OpenAI-compatible server at `url` (…/v1) serves; raises OSError (with
    the reason) when it does not answer."""
    request = urllib.request.Request(
        f"{url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            listed = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        raise OSError(f"{url} does not answer ({error})") from error
    return [str(model.get("id")) for model in listed.get("data", [])]


def checked_model(
    url: str,
    model: str | None,
    api_key: str | None,
    models: Callable[..., list[str]] = served_models,
) -> str:
    """`model`, once the server at `url` answers and serves it; the one it serves when none is
    named and it serves one. Raises OSError when not."""
    served = models(url, api_key or "EMPTY")
    if model is None:
        if len(served) != 1:
            raise OSError(f"{url} serves {', '.join(served) or 'no model'}: name one with --model")
        return served[0]
    if model not in served:
        raise OSError(f"{url} does not serve {model}; it serves {', '.join(served) or 'none'}")
    return model


def remote_settings(
    url: str,
    model: str | None,
    api_key: str | None,
    models: Callable[..., list[str]] = served_models,
) -> tuple[dict[str, str], str]:
    """The .env lines of the assistant with its model served at `url` (another machine, behind
    HTTPS), checked first. Raises OSError when it does not answer or serve the model."""
    if urllib.parse.urlsplit(url).hostname in ("127.0.0.1", "localhost", "::1"):
        raise OSError(
            f"{url} is this machine itself, out of PAC's container's reach: serve the model with "
            "PAC (--with), or reach another machine through SSH (--tunnel USER@HOST)"
        )
    model = checked_model(url, model, api_key, models)
    settings = {"PAC_EXTRAS": "agent", "PACO_LLM_BASE_URL": url, "PACO_LLM_MODEL": model}
    if api_key:
        settings["PACO_LLM_API_KEY"] = api_key
    return settings, model


def ssh_key(ssh_dir: Path) -> Path | None:
    """The key the tunnel logs in with: the first of SSH_KEYS in `ssh_dir`."""
    return next((ssh_dir / name for name in SSH_KEYS if (ssh_dir / name).is_file()), None)


def tunnel_settings(
    target: str,
    port: int,
    key: Path,
    known_hosts: Path,
    model: str | None,
    api_key: str | None,
    check_port: int = 18_001,
) -> tuple[dict[str, str], str]:
    """The .env lines of the assistant with its model on the machine `target` (user@host) logs
    into, checked first through a tunnel opened from this machine for the time of the check.
    Raises OSError with what to do when SSH, or the model, does not answer."""
    tunnel = subprocess.Popen(
        [
            "ssh",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-i",
            str(key),
            "-p",
            str(port),
            "-L",
            f"127.0.0.1:{check_port}:127.0.0.1:8001",
            target,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            if tunnel.poll() is not None:
                raise OSError(
                    _ssh_advice(tunnel.stderr.read() if tunnel.stderr else "", target, key, port)
                )
            try:
                model = checked_model(f"http://127.0.0.1:{check_port}/v1", model, api_key)
                break
            except OSError as error:
                if time.monotonic() > deadline or "serves" in str(error):
                    raise OSError(
                        f"SSH logs into {target}, but no model answers on its 127.0.0.1:8001: "
                        "start it there (docker compose up -d model), and wait for it to be "
                        f"healthy ({error})"
                    ) from error
                time.sleep(1)
    finally:
        tunnel.terminate()
        tunnel.wait()
    settings = {
        "PAC_EXTRAS": "agent",
        "COMPOSE_PROFILES": "tunnel",
        "MODEL_SSH": target,
        "MODEL_SSH_PORT": str(port),
        "MODEL_SSH_KEY": str(key),
        "MODEL_SSH_KNOWN_HOSTS": str(known_hosts),
        "PACO_LLM_BASE_URL": "http://tunnel:8001/v1",
        "PACO_LLM_MODEL": model,
    }
    if api_key:
        settings["PACO_LLM_API_KEY"] = api_key
    return settings, model


def _ssh_advice(message: str, target: str, key: Path, port: int = 22) -> str:
    """What SSH's failure means, and what to do."""
    option = "" if port == 22 else f"-p {port} "
    if "Host key verification failed" in message or "No ED25519 host key" in message:
        return (
            f"this computer does not know {target} yet: log into it once with `ssh {option}{target}` "
            "and answer yes, then run this again"
        )
    if "Permission denied" in message or "passphrase" in message:
        return (
            f"SSH cannot log into {target} with {key} without a passphrase: make a key for the "
            f'tunnel (ssh-keygen -t ed25519 -f ~/.ssh/pac_tunnel -N ""), put it on the GPU '
            f"machine (ssh-copy-id {option}-i ~/.ssh/pac_tunnel.pub {target}), then run this again"
        )
    return f"SSH cannot reach {target}: {message.strip() or 'no answer'}"


def write_env(path: Path, settings: Mapping[str, str]) -> None:
    """`path`, compose's .env, with the assistant's lines replaced by `settings` (none: the
    assistant left out) and every other line kept."""
    lines = path.read_text().splitlines() if path.exists() else []
    kept = [line for line in lines if line.split("=", 1)[0].strip() not in AGENT_KEYS]
    kept += [f"{key}={value}" for key, value in settings.items()]
    path.write_text("".join(f"{line}\n" for line in kept))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install PAC's assistant, PACo, when this machine's GPU can serve its model: "
        "writes compose's .env, then `docker compose up -d --build` starts PAC with it."
    )
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--with", dest="answer", action="store_const", const="yes")
    choice.add_argument("--without", dest="answer", action="store_const", const="no")
    choice.add_argument(
        "--tunnel",
        metavar="USER@HOST",
        help="the assistant with its model on a GPU machine reached through SSH (the machine "
        "you log into with `ssh USER@HOST`), whatever this machine's GPU",
    )
    parser.add_argument("--ssh-port", type=int, default=22, help="the GPU machine's SSH port")
    parser.add_argument("--ssh-key", type=Path, help="the key the tunnel logs in with")
    parser.add_argument(
        "--known-hosts", type=Path, help="the file SSH knows the GPU machine from (~/.ssh's)"
    )
    choice.add_argument(
        "--remote",
        metavar="URL",
        help="the assistant with a model served at URL (behind HTTPS, with --api-key)",
    )
    parser.add_argument(
        "--model", help="the model a remote server serves (its only one by default)"
    )
    parser.add_argument("--api-key", help="the key a remote server asks for (its VLLM_API_KEY)")
    parser.add_argument("--env", type=Path, default=Path(".env"), help="compose's .env file")
    args = parser.parse_args(argv)

    if args.tunnel:
        ssh_dir = Path.home() / ".ssh"
        option = "" if args.ssh_port == 22 else f"-p {args.ssh_port} "
        key = args.ssh_key or ssh_key(ssh_dir)
        if key is None or not key.is_file():
            print(
                "Not installed: no SSH key to log into the GPU machine with. Make one for the "
                'tunnel (ssh-keygen -t ed25519 -f ~/.ssh/pac_tunnel -N ""), put it on the GPU '
                f"machine (ssh-copy-id {option}-i ~/.ssh/pac_tunnel.pub {args.tunnel}), then run this again."
            )
            return 1
        print(f"Checking the model on {args.tunnel} through SSH, with {key} ...")
        try:
            settings, model = tunnel_settings(
                args.tunnel,
                args.ssh_port,
                key.resolve(),
                (args.known_hosts or ssh_dir / "known_hosts").resolve(),
                args.model,
                args.api_key,
            )
        except OSError as error:
            print(f"Not installed: {error}.")
            return 1
        write_env(args.env, settings)
        print(f"The model answers: the assistant will ask {model} on {args.tunnel}, through SSH.")
        print("Next: docker compose up -d --build")
        return 0
    if args.remote:
        try:
            settings, model = remote_settings(args.remote, args.model, args.api_key)
        except OSError as error:
            print(f"Not installed: {error}.")
            return 1
        write_env(args.env, settings)
        print(
            f"The assistant will ask {model} at {args.remote} (from PAC's container: "
            f"{settings['PACO_LLM_BASE_URL']})."
        )
        print("Next: docker compose up -d --build")
        return 0
    answer = args.answer
    if answer is None:
        reply = input(
            "Install PAC's assistant, an AI agent that processes profiles for you? It needs an "
            "NVIDIA or AMD GPU of 16 GB or more. [y/N] "
        )
        answer = "yes" if reply.strip().lower() in ("y", "yes") else "no"
    if answer == "no":
        write_env(args.env, {})
        print("PAC will run without the assistant.")
        print("Next: docker compose up -d --build")
        return 0

    verdict = check()
    print(verdict.reason)
    write_env(args.env, verdict.settings)
    if not verdict.compatible:
        print(
            "PAC will run without the assistant. With a model served elsewhere: "
            "python3 install_assistant.py --remote URL --model NAME"
        )
        return 1
    print(f"Written to {args.env}: " + ", ".join(f"{k}={v}" for k, v in verdict.settings.items()))
    print("Next: docker compose up -d --build (the first start downloads the model)")
    return 0


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


if __name__ == "__main__":
    sys.exit(main())
