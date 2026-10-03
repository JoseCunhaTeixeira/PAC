# PAC — Passive and Active Computation of MASW

[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14808813.svg)](https://doi.org/10.5281/zenodo.14808813)

PAC is an app for processing **Multichannel Analysis of Surface Waves (MASW)** on linear arrays. It can handle both **passive** and **active** seismic data to automatically optimize and extract dispersion images, and it can apply cross-correlation to active data to sometimes improve dispersion retrieval quality. Dispersion curves can be semi-automatically picked on an interactive interface and then inverted into shear-wave velocity profiles.

PAC runs entirely **on your own computer**. You start it once, then use it through your web browser at a local address (`http://localhost:5173`). Nothing is uploaded anywhere — your data never leaves your machine.

## Features
- **Hybrid processing:** supports both passive and active MASW methods without needing source position information. Uses the package [sigpipe](https://github.com/JoseCunhaTeixeira/sigpipe), a signal processing Python pipeline.
- **Signal processing tools:** uses an automatic source detection algorithm combined with seismic interferometry.
- **Automated dispersion analysis:** extracts and visualizes dispersion images and curves.
- **Seismic inversion:** turns dispersion curves into shear-wave velocity profiles by Bayesian inversion (sigpipe). The number of layers is chosen by the data, or set by you; each model's dispersion curve is computed with [Disba](https://github.com/keurfonluu/disba).
- **Petrophysical inversion:** computes soil profiles from dispersion data using the AI inversion model from [silex](https://github.com/JoseCunhaTeixeira/silex).
- **User-friendly interface:** streamlined workflow with visualization capabilities.
- **Assistant (optional):** an AI agent, PACo, that processes profiles for you when you ask it in plain words, checking the quality of every step. It runs a language model, Qwen3-14B, on a graphics card with 24 GB of memory (on your computer or on another machine) or from an online service: see [The assistant](#the-assistant-optional).

## Overview
<img width="1643" height="1738" alt="pac-home" src="https://github.com/user-attachments/assets/c060f45f-52aa-4c34-a6b5-7689c2f9b046" />
<img width="1641" height="1740" alt="pac-visu" src="https://github.com/user-attachments/assets/0f791ee5-b1af-4260-982a-3f644b9b7ae1" />
<img width="1672" height="735" alt="pac-assistant" src="https://github.com/user-attachments/assets/d80f84a2-fec0-4d74-b6a8-181ef914b10b" />

## Running the app

### What you need
- **Docker** — a free tool that runs PAC in ready-made "containers", so you never have to install Python, Node, or any scientific library yourself. Get it from [docs.docker.com/get-docker](https://docs.docker.com/get-docker/): install **Docker Desktop** on Windows or macOS, or **Docker Engine** on Linux. Any recent install includes the Compose plugin PAC needs (the `docker compose` command — not the older standalone `docker-compose`).
- **A terminal** to type a few commands: the *Terminal* app on macOS or Linux, *PowerShell* on Windows (press the Start key and type "PowerShell").

To check everything is ready, open a terminal and run:
```sh
docker compose version
```
If a version number is printed, you're good to go. On Windows and macOS, keep the Docker Desktop application running whenever you use PAC.

### Which option should I pick?
- **Option 1 (recommended):** download ready-to-run PAC images. Fastest, and you never touch the source code.
- **Option 2:** download the source code and build the images yourself. Choose this only if you want to read or modify the code.

Both options give you the exact same app at http://localhost:5173.

### Option 1: Run the published docker images (recommended, no clone needed)
A backend image and a frontend image are built and published to GitHub Container Registry on every push to `main` ([`ghcr.io/josecunhateixeira/pac-backend`](https://github.com/JoseCunhaTeixeira/PAC/pkgs/container/pac-backend), [`ghcr.io/josecunhateixeira/pac-frontend`](https://github.com/JoseCunhaTeixeira/PAC/pkgs/container/pac-frontend)). Docker downloads and runs them for you — no source code needed. (Every commit is also tagged `sha-<short-sha>`, to pin a version.)

**Step 1 — Create a folder for PAC.** Pick any location you like (home folder, Documents, …). Inside it, PAC needs two data subfolders: `data/input` (where you'll put your seismic records) and `data/output` (where PAC writes its results).

On macOS or Linux:
```sh
# Creates a "pac" folder containing data/input and data/output, then enters it.
mkdir -p pac/data/input pac/data/output
cd pac
```
On Windows (PowerShell):
```powershell
mkdir pac\data\input, pac\data\output
cd pac
```
You can also create the same folders with Finder / File Explorer if you prefer — just make sure to open your terminal inside the `pac` folder afterwards, because the next commands must run from there.

**Step 2 — Download the Compose file into that folder.** `docker-compose.prod.yml` is a small text file that tells Docker which PAC images to run and how to connect them.

On macOS or Linux:
```sh
# Saves docker-compose.prod.yml into the current folder.
curl -O https://raw.githubusercontent.com/JoseCunhaTeixeira/PAC/main/docker-compose.prod.yml
```
On Windows (PowerShell):
```powershell
curl.exe -O https://raw.githubusercontent.com/JoseCunhaTeixeira/PAC/main/docker-compose.prod.yml
```
Alternatively, right-click [this link](https://raw.githubusercontent.com/JoseCunhaTeixeira/PAC/main/docker-compose.prod.yml), choose "Save link as…", and save the file into your `pac` folder.

**Step 3 — Start PAC.** Same command on every system:
```sh
# Downloads the two PAC images (first time only) and starts them in the background.
docker compose -f docker-compose.prod.yml up -d
```
What this does: `-f docker-compose.prod.yml` points Docker at the file you just downloaded, and `-d` runs PAC in the background so you get your terminal back. The first start downloads the images and can take a few minutes; after that it's nearly instant. PAC then keeps running (and even comes back automatically after a reboot) until you stop it in Step "Stop PAC" below.

**Step 4 — Open the app.** Go to http://localhost:5173 in your browser. Two demo profiles, `active_p1` and `passive_p1`, are already there so you can try PAC immediately (see [Adding your own data](#adding-your-own-data)). If the page doesn't load right away, wait a moment and refresh — or see [Troubleshooting](#troubleshooting).

#### Everyday commands (optional)
None of these are needed just to use PAC — it keeps running by itself. Use them when you want to inspect, update, or stop the app. Always run them from inside your `pac` folder.

*Watch what PAC is doing (live logs):*
```sh
# Streams the logs of both services. Press Ctrl-C to stop watching
# (this does NOT stop PAC itself).
docker compose -f docker-compose.prod.yml logs -f
```

*Update PAC to the newest published version (run both, in this order):*
```sh
# 1. Download the newest images. Nothing restarts yet.
docker compose -f docker-compose.prod.yml pull

# 2. Restart only what changed, now using the new images.
docker compose -f docker-compose.prod.yml up -d
```

*Stop PAC:*
```sh
# Stops and removes the containers. Your data/ folder is completely untouched,
# and the downloaded images stay cached — restart anytime with the Step 3 command.
docker compose -f docker-compose.prod.yml down
```

> **Note:** this setup assumes the browser and Docker run on the same machine — the published frontend image expects the backend at `http://localhost:8000`. To serve PAC from a remote server reached by its own domain/IP, use Option 2 and see [C: PAC on a server](#c-pac-on-a-server).

> **The assistant** is not in the published images: to use it, install PAC with Option 2, then follow [The assistant](#the-assistant-optional).

### Option 2: Build and run from the source code
Everything runs fully containerized here too (backend + frontend), with no local Python/Node setup required — the only extra requirement is **[Git](https://git-scm.com/downloads)** to download the source code. (No Git? On the [repository page](https://github.com/JoseCunhaTeixeira/PAC), click **Code → Download ZIP**, unzip it, and skip the `git clone` line below — but updating later then means re-downloading the ZIP instead of `git pull`.)

**Step 1 — Get the source code:**
```sh
# Downloads the full source code into a "PAC" folder and enters it.
git clone https://github.com/JoseCunhaTeixeira/PAC.git
cd PAC/
```

**Step 2 — Build and start PAC:**
```sh
# Builds the backend and frontend images from the source code, then starts
# them in the background. The first build takes several minutes; later
# builds reuse Docker's cache and are much faster.
docker compose up --build -d
```

Then open http://localhost:5173 in a browser. `data/input/` and `data/output/` are shared directly with your computer (bind-mounted from the host), so dropping a new profile folder into `data/input/` works exactly as in a native install, and results in `data/output/` persist across container restarts.

#### Everyday commands (optional)
Run these from inside the `PAC` folder. Since it contains the default `docker-compose.yml`, no `-f` flag is needed here.

*Watch what PAC is doing (live logs):*
```sh
# Streams the logs of both services. Ctrl-C stops watching, not PAC.
docker compose logs -f
```

*Update after new commits are published (run both, in this order):*
```sh
# 1. Download the latest source code changes.
git pull

# 2. Rebuild the images from the updated code and restart only what changed.
docker compose up --build -d
```

*Stop PAC:*
```sh
# Stops and removes the containers. data/ is untouched;
# restart anytime with the Step 2 command.
docker compose down
```

### Adding your own data
PAC looks for your recordings in `data/input/` and writes its results to `data/output/`, with **one folder per profile**:

- `data/`
    - `input/`: contains one folder per profile with your raw seismic records in any format ObsPy can read (e.g. SEG-2 `.dat` files)
        - `active_profile_1/`: one shot per seismic file, requires receiver **and** source positions
            - `file1.dat`
            - `file2.dat`
            - `receiver_positions.yaml`
            - `source_positions.yaml`
        - `passive_profile_2/`: passive recordings, only requires receiver positions
            - `file1.dat`
            - `file2.dat`
            - `receiver_positions.yaml`
    - `output/`: contains one folder per profile with dispersion and inversion results, one folder per run (each time you compute a profile, named by its date and time, e.g. `20260926-101557-75d0`)
        - `active_profile_1/`
            - `20260926-101557-75d0/`: a run, with one `xmid_<position>` folder per window
        - `passive_profile_2/`

The Dispersion Picking, Seismic Inversion, Petrophysical Inversion and Visualization pages list the runs, the most recent first, and open on it. Results computed by earlier versions of PAC, straight in `output/<profile>/`, are listed too.

The **Visualization** page shows one run: how it was made (by you or the assistant, each setting with where it comes from), the line with its windows coloured by quality at each stage, and for the window you select its image, picks, velocity model and checks.

To process your own data, simply drop a profile folder like the above into `data/input/` — no restart needed.

**About the demo profiles:** an empty `data/input` (a freshly created folder, as in Option 1) gets seeded on first start with two demo profiles, `active_p1` and `passive_p1`, so there's something to try immediately. A `data/input` that already has content — your own profile folders, or the demo data committed with the source code in Option 2 — is left untouched; add your own profile folders alongside or instead of the demo ones.

**About your results:** they are written to `data/output` on your computer, so stopping or removing the containers keeps them.

### Troubleshooting
- **`'docker' is not recognized` / `docker: command not found`** — Docker isn't installed, or the terminal was opened before installing it. Install [Docker](https://docs.docker.com/get-docker/), then open a **new** terminal and try again.
- **`Cannot connect to the Docker daemon` / `error during connect`** — Docker isn't running. Start the Docker Desktop application and wait for it to finish loading, then retry.
- **`permission denied ... docker.sock` (Linux)** — your user can't talk to Docker yet. Either prefix commands with `sudo`, or (better) [add your user to the `docker` group](https://docs.docker.com/engine/install/linux-postinstall/) and log out/in once.
- **`port is already allocated` / `address already in use`** — another program is already using port 5173 or 8000. Close it, or edit the Compose file and change the number on the **left** side of the colon under `ports:` (e.g. `5174:...`), then open that port in the browser instead.
- **The page won't load right after starting** — the very first start can take a few minutes while images are downloaded or built. Watch the progress with the *live logs* command above, then refresh the browser.

## The assistant (optional)

The assistant is an AI agent, **PACo**, that works PAC for you. You ask it in plain words — *"Process active_p1 and give me its dispersion curves"* — and it runs PAC's processing, checks the quality of every step (and retries what it can), picks the dispersion curves, inverts them into velocity models if you asked for them (or into soils and a water table, if you asked for those), then tells you what it did and which settings it changed. Its results are ordinary PAC runs: you open them in the other pages, to review or correct them.

You can keep working in PAC's other pages while it answers: a run one side is writing is refused to the other until it ends. Nothing is deleted: results a step replaces are kept in the window's `replaced` folder. The Assistant entry of the menu shows when it is working, and when an answer waits for your choice.

It runs a language model, **Qwen3-14B** (in FP8), which needs a **graphics card (GPU) with 24 GB of memory**. Run on your own machines, everything stays there: the model never sees your seismic records, only short summaries of PAC's results. An online service can serve the model too: see [The model](#the-model).

The assistant is optional. Without it, the rest of PAC works the same, and its menu has no Assistant page.

### Which setup is yours?

| Your situation | Setup |
|---|---|
| Your computer has a compatible GPU (see below) | [A: everything on your computer](#a-everything-on-your-computer) |
| Your computer has no compatible GPU, but you can use a machine that has one (a lab workstation, a GPU server rented in the cloud) | [B: the model on a GPU machine](#b-the-model-on-a-gpu-machine) |
| PAC runs on a server without a GPU, and you open it from your own computer | [C: PAC on a server](#c-pac-on-a-server) |
| No compatible GPU anywhere | Nothing to do: PAC runs without the assistant, or use an online service ([The model](#the-model)) |

**Compatible GPUs** — you don't need to work this out yourself: in every setup, the command `install_assistant.py` reads the GPU and tells you.
- **NVIDIA**, with 24 GB of memory or more, from the RTX 30 series on (compute capability 8.0 or newer, for FP8): for example RTX 3090, 4090, 5090, RTX A5000, A6000, L4, A10, A100. On Linux, or on Windows with Docker Desktop.
- **AMD**, with 24 GB of memory or more, on Linux: for example Radeon RX 7900 XTX.
- **Not compatible**: GPUs with less than 24 GB, and Macs (Docker cannot use their GPU). On such a computer, use setup B, or an online service.

**What every setup needs:**
- PAC installed from its source code ([Option 2](#option-2-build-and-run-from-the-source-code)): the published images of Option 1 don't include the assistant.
- **Python 3.9 or newer**, only to run the checking command `install_assistant.py`. Linux and macOS usually have it: `python3 --version` prints its version. On Windows, install it from [python.org](https://www.python.org/downloads/), and type `python` wherever this guide says `python3`.
- On the machine that runs the model, the GPU's driver and Docker's access to it:
  - **NVIDIA on Linux**: NVIDIA's driver (the command `nvidia-smi` then prints your card) and NVIDIA's [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), which lets Docker use the card.
  - **NVIDIA on Windows**: NVIDIA's Windows driver and Docker Desktop with its WSL 2 backend (the default).
  - **AMD on Linux**: the amdgpu driver with ROCm support, which recent distributions include: the command `ls /dev/kfd` then prints `/dev/kfd`.
  - Disk space for the model: about 16 GB, downloaded on the first start.

### A: everything on your computer

**Step 1 — Get PAC's source code** (skip this step if you already have it):
```sh
git clone https://github.com/JoseCunhaTeixeira/PAC.git
cd PAC/
```

**Step 2 — Check the GPU.** In the `PAC` folder:
```sh
python3 install_assistant.py
```
It asks whether you want the assistant: type `y` and press Enter. It then reads your GPU and prints its verdict, for example:
```
NVIDIA GeForce RTX 4090 (24.0 GiB): compatible, the assistant will run Qwen/Qwen3-14B-FP8.
Written to .env: PAC_EXTRAS=agent, COMPOSE_PROFILES=agent, PACO_LLM_MODEL=Qwen/Qwen3-14B-FP8, VLLM_MAX_MODEL_LEN=16384
Next: docker compose up -d --build (the first start downloads the model)
```
It writes these settings into a file named `.env`, in the `PAC` folder, which Docker reads in the next step. If your GPU is not compatible, it says why, and PAC is set up without the assistant.

**Step 3 — Build and start PAC:**
```sh
docker compose up -d --build
```
It is the same command as in Option 2: with the `.env` file, it also builds the assistant into PAC, and starts the model. The first time, Docker downloads the model (about 16 GB): count 10 to 30 minutes, depending on your connection.

**Step 4 — Wait until the model is ready:**
```sh
docker compose ps
```
The line of the `model` service ends with `(healthy)` once the model is ready, with `(health: starting)` while it downloads or loads. To follow its progress: `docker compose logs -f model` (Ctrl-C stops watching, not the model).

**Step 5 — Talk to the assistant.** Open http://localhost:5173: the menu now has an **Assistant** page. Type your request and press Enter, for example:
- *Which profiles can I process?*
- *Process active_p1 and give me its dispersion curves.*
- *Process active_p1 and invert it: I want its velocity section.*
- *Process active_p1 and give me the soil types and the water table.* The assistant runs the petrophysical inversion (a Silex model) on the curves the model was trained for: the bundled one needs curves reaching 43 Hz, and the assistant says how many of yours do.

The page shows each step the assistant takes (the tools it calls, in small grey text; untick *Show tool calls* to hide them) and a progress line while it works. Processing a profile takes a few minutes; an inversion, longer. When it has finished, open the **Dispersion Picking**, **Seismic Inversion**, **Petrophysical Inversion** or **Visualization** pages: they open on the latest run, the assistant's.

**Stop, restart, remove:**
- Stop PAC and the model: `docker compose down`. Start them again: `docker compose up -d`.
- Remove the assistant: `python3 install_assistant.py --without`, then `docker compose up -d --build`. PAC then runs as before, without the Assistant page.

### B: the model on a GPU machine

PAC runs on your computer, which needs no GPU; only the model runs on a machine with a compatible GPU — a lab workstation, or a GPU server rented in the cloud. They talk through **SSH**, the secure connection you use to log into the GPU machine: the model stays invisible from the internet, and your data stay on your computer (the GPU machine only receives your requests and PAC's short summaries).

You need an account on the GPU machine that you can log into from your computer with `ssh user@gpu-machine`. In every command below, replace `user@gpu-machine` with your own login (for example `jose@192.168.1.20`, or `ubuntu@my-server.example.org`).

**On the GPU machine** (once):

**Step 1 — Log into it, and get PAC's source code.** It needs Docker, Git, Python 3, and its GPU's driver and Docker access (see *What every setup needs*).
```sh
ssh user@gpu-machine
git clone https://github.com/JoseCunhaTeixeira/PAC.git
cd PAC/
```

**Step 2 — Check its GPU:**
```sh
python3 install_assistant.py --with
```
It must answer *compatible*.

**Step 3 — Start the model, and only the model:**
```sh
docker compose up -d model
```
Wait until `docker compose ps` shows the `model` service `(healthy)`: the first time, it downloads the model (10 to 30 minutes). The model listens on the GPU machine itself only, never on its network. You can now log out (`exit`): the model keeps running, and starts again by itself when the GPU machine reboots.

**On your computer:**

**Step 4 — Make a key for the connection.** PAC connects to the GPU machine by itself, in the background, so it needs an SSH key without a passphrase, made for it:
```sh
ssh-keygen -t ed25519 -f ~/.ssh/pac_tunnel -N ""
ssh-copy-id -i ~/.ssh/pac_tunnel.pub user@gpu-machine
```
The first command creates the key (the files `pac_tunnel` and `pac_tunnel.pub`, in the `.ssh` folder of your home folder). The second puts its public half on the GPU machine: it asks for your password there, once. On Windows, which has no `ssh-copy-id`, use this PowerShell command instead of the second one:
```powershell
type $env:USERPROFILE\.ssh\pac_tunnel.pub | ssh user@gpu-machine "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```
If you have never logged into the GPU machine from this computer, do it once now: `ssh user@gpu-machine`, answer `yes` to the question about the machine's fingerprint, then `exit`.

**Step 5 — Connect PAC to the model.** In the `PAC` folder of your computer ([Option 2](#option-2-build-and-run-from-the-source-code), Step 1, if you don't have it yet):
```sh
python3 install_assistant.py --tunnel user@gpu-machine
```
It logs into the GPU machine with the key, checks that the model answers, and writes `.env`:
```
The model answers: the assistant will ask Qwen/Qwen3-14B-FP8 on user@gpu-machine, through SSH.
Next: docker compose up -d --build
```
If something is missing, it says what to do: log in once, copy the key (Step 4), or start the model (Step 3).

**Step 6 — Start PAC:**
```sh
docker compose up -d --build
```
Then open http://localhost:5173 and the **Assistant** page, as in Step 5 of setup A. PAC keeps the connection to the model by itself (a small `tunnel` service), and restores it after a network cut or a reboot of either machine.

**If the GPU machine's SSH port is not 22**, add `--ssh-port` and your port to Step 5 (`--ssh-port 2222`), and `-p` with it to the commands of Step 4 (`ssh-copy-id -p 2222 ...`, `ssh -p 2222 ...`).

**Without SSH (for administrators).** The model can also be published over HTTPS, with a key: on the GPU machine, put `VLLM_API_KEY=a-long-random-secret` and `MODEL_BIND=0.0.0.0` in `.env` before Step 3, and put the model's port 8001 behind an HTTPS reverse proxy; on your computer, run `python3 install_assistant.py --remote https://model.example.org/v1 --api-key a-long-random-secret` instead of Step 5. Never open the model to a network without HTTPS and a key.

### C: PAC on a server

PAC can run on a server — with or without a GPU — and be opened from other computers' browsers. On the server, before starting PAC (Step 3 of setup A, or Step 6 of setup B), write the server's address into `.env`; replace `server` with the name or IP address the other computers reach it by:
```sh
echo "PAC_API_URL=http://server:8000" >> .env
echo "PAC_WEB_URL=http://server:5173" >> .env
```
Then start PAC with `docker compose up -d --build`, and open http://server:5173 from any computer of your network. For the assistant, follow setup A if the server has a compatible GPU, otherwise setup B, running the steps of "your computer" on the server.

**Caution:** anyone who reaches the server's ports 5173 and 8000 can use PAC and its assistant. Keep the server on a private network (or behind a VPN), and never open these ports to the internet.

### The model

The assistant runs Qwen3-14B (`Qwen/Qwen3-14B-FP8`), served by vLLM behind an OpenAI-compatible chat API with tool calling.

- **Several GPUs:** add `VLLM_TENSOR_PARALLEL_SIZE=2` to `.env` (the number of GPUs), and the model is spread over them.
- **The model does not start** and `docker compose logs model` says the memory is too small: lower `VLLM_MAX_MODEL_LEN` in `.env` (the conversation's length, 16384 by default; below 12288, long requests may not fit).

**Another model.** The assistant works with any model served behind an OpenAI-compatible chat API that supports tool calling: set `PACO_LLM_MODEL` (and `VLLM_MAX_MODEL_LEN`, its context) in `.env`, then `docker compose up -d`. A model of another family also needs its own tool-call parser in `docker-compose.yml`'s `model` service, in place of `--tool-call-parser=hermes`, and no `--reasoning-parser=qwen3`. PACo's tests (15 scenarios of the assistant's rules, played 5 times each) gave:

| Model | GPU memory | PACo's tests (75 plays) |
|---|---|---|
| `Qwen/Qwen3-14B-FP8` (the one PACo runs) | 24 GB | 74 passed |
| `Qwen/Qwen3-8B-FP8` | 16 GB, with a 12,288-token context | 68 passed |
| `Qwen/Qwen3.8-27B` | 40 to 48 GB (FP8) | 69 passed |
| `Qwen/Qwen3-30B-A3B-FP8` | 48 GB, or two 24 GB GPUs | not tested |
| `Qwen/Qwen3-32B-FP8` | 48 GB, or two 24 GB GPUs | not tested |

**An online service** (no GPU needed). Several companies serve models behind an OpenAI-compatible API; choose one that serves Qwen3-14B. From the service, you need three things: the API's address (it ends in `/v1`), the model's name, and an API key. Then, in the `PAC` folder:
```sh
python3 install_assistant.py --remote https://api.example.com/v1 --model <the model's name> --api-key <your key>
docker compose up -d --build
```
The command checks that the service answers and serves that model, then writes the three into `.env`. The key stays in that file, on your computer: never share or commit it.

What the service then receives: your messages to the assistant, the descriptions of PACo's tools, and what the tools return, that is the quality gates' summaries (profile and run names, settings, verdicts, velocities and depths). Your records, dispersion images and models stay on your computer. Check that this suits your data's rules before using an online service. Services charge by use: one request, such as processing and picking a profile, takes a few tens of thousands of tokens.

### If something goes wrong

| What you see | What to do |
|---|---|
| No **Assistant** in the menu | PAC was built without it: run `python3 install_assistant.py`, then `docker compose up -d --build`. |
| *The assistant is not available: The model server ... does not answer* | Setup A: the model is still loading (`docker compose ps`: wait for `(healthy)`), or it failed (`docker compose logs model`). Setup B: check that the model runs on the GPU machine (Step 3), then the connection (`docker compose logs tunnel`). |
| `docker compose logs model` mentions *out of memory* | Another program uses the GPU's memory (a game, another model): close it, then `docker compose restart model`. Otherwise the card is too small: see *Compatible GPUs*. |
| `docker compose logs tunnel` mentions *Permission denied* | The key is not on the GPU machine: redo Step 4 of setup B. |
| *Host key verification failed* | Your computer does not know the GPU machine yet: log into it once (`ssh user@gpu-machine`, answer `yes`), then redo Step 5 of setup B. |
| The assistant is slow | That is normal: it runs PAC's processing, which takes minutes; the progress line under the conversation shows what it is doing. |
| The assistant made a choice you disagree with | Tell it, in the same conversation: *use windows of 24 receivers*, *invert with 3 layers*. **New conversation** starts afresh. |

### For developers (without Docker)
With the source code and [uv](https://docs.astral.sh/uv/), and a model server at hand (for example vLLM, serving Qwen3-14B):
```sh
uv sync --extra agent                                # PAC with PACo, the assistant
export PACO_LLM_BASE_URL=http://127.0.0.1:8001/v1    # the model server
export PACO_LLM_MODEL=Qwen/Qwen3-14B-FP8             # the model it serves
export PACO_LLM_CONTEXT=16384                        # its context (vLLM's --max-model-len)
uv run uvicorn masw.api.main:app --host 127.0.0.1 --port 8000
```
and the web app with `npm install && npm run dev` in `frontend/`. With an online service, also set `PACO_LLM_API_KEY`. These settings can instead go in a `.env` file in the `PAC` folder, which the backend reads when started from there.

The assistant's processing uses half of the computer's cores (`PACO_WORKERS` to change it), and keeps the windows' images it makes in `data/output/.cache`, at most 2 GB, to take them again when a run makes the same ones (`PACO_CACHE_GB=0` keeps none).

## License
This project is under Creative Commons Attribution 4.0 International license, allowing re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited.
Please cite as:
> Cunha Teixeira, J. (2025). PAC - Passive and Active Computation of MASW. Zenodo. doi:[10.5281/zenodo.14808813](https://doi.org/10.5281/zenodo.14808813)


## Acknowledgments
This work was developed with contributions from the geophysics research community. Special thanks to open-source developers for their invaluable tools.
The algorithms are based on the PhD thesis of José Cunha Teixeira, funded by a cooperation between Sorbonne University, Mines Paris - PSL, SNCF Réseau, and the European Union's Horizon Europe research and innovation program under Grant Agreement No 101101966.
Please refer to:
> Cunha Teixeira, J., Bodet, L., Dangeard, M., Gesret, A., Hallier, A., Rivière, A., Burzawa, A., Cárdenas Chapellín, J. J., Fonda, M., Sanchez Gonzalez, R., Dhemaied, A., & Boisson Gaboriau, J. (2025). Nondestructive testing of railway embankments by measuring multi-modal dispersion of surface waves induced by high-speed trains with linear geophone arrays. *Seismica*, *4*(1). doi:[10.26443/seismica.v4i1.1150](https://doi.org/10.26443/seismica.v4i1.1150)

> Cunha Teixeira, J., Bodet, L., Rivière, A., Solazzi, S. G., Hallier, A., Gesret, A., El Janyani, S., Dangeard, M., Dhemaied, A., & Boisson Gaboriau, J. (2025). Neural Machine Translation of Seismic Ambient Noise for Soil Nature and Water Saturation Characterization. *Geophysical Research Letters*, *52*(13), e2025GL114852. doi:[10.1029/2025GL114852](https://doi.org/10.1029/2025GL114852)
