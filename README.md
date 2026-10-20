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
- **Assistant:** an AI agent, PACo, that processes profiles for you when you ask it in plain words, checking the quality of every step. It uses the language model you choose: a provider's API (Claude, OpenAI, Gemini, Mistral) or your own server. See [The assistant](#the-assistant).

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

**Step 1 — Create a folder for PAC.** Pick any location you like (home folder, Documents, …). Inside it, PAC needs three data subfolders: `data/input` (where you'll put your seismic records), `data/output` (where PAC writes its results) and `data/settings` (where the assistant keeps its settings).

On macOS or Linux:
```sh
# Creates a "pac" folder containing data/input, data/output and data/settings, then enters it.
mkdir -p pac/data/input pac/data/output pac/data/settings
cd pac
```
On Windows (PowerShell):
```powershell
mkdir pac\data\input, pac\data\output, pac\data\settings
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

**Step 4 — Open the app.** Go to http://localhost:5173 in your browser. Two test profiles, `active_test_profile` and `passive_test_profile`, are already there so you can try PAC immediately (see [Adding your own data](#adding-your-own-data)). If the page doesn't load right away, wait a moment and refresh — or see [Troubleshooting](#troubleshooting).

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

> **Note:** this setup assumes the browser and Docker run on the same machine — the published frontend image expects the backend at `http://localhost:8000`. To serve PAC from a remote server reached by its own domain/IP, use Option 2 and see [PAC on a server](#pac-on-a-server).

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

### PAC on a server
PAC can run on a server and be opened from other computers' browsers. With Option 2, before starting PAC, write the server's address into `.env`, in the `PAC` folder; replace `server` with the name or IP address the other computers reach it by:
```sh
echo "PAC_API_URL=http://server:8000" >> .env
echo "PAC_WEB_URL=http://server:5173" >> .env
```
Then start PAC with `docker compose up -d --build`, and open http://server:5173 from any computer of your network.

**Caution:** anyone who reaches the server's ports 5173 and 8000 can use PAC and its assistant. Keep the server on a private network (or behind a VPN), and never open these ports to the internet.

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

**About the test profiles:** an empty `data/input` (a freshly created folder, as in Option 1) gets seeded on first start with two test profiles, so there's something to try immediately: `active_test_profile` (25 shots on 96 receivers) and `passive_test_profile` (10 records of ambient noise on the same receivers), light copies of real lines. A `data/input` that already has content — your own profile folders, or the test profiles committed with the source code in Option 2 — is left untouched; add your own profile folders alongside or instead of the test profiles.

**About your results:** they are written to `data/output` on your computer, so stopping or removing the containers keeps them.

### Troubleshooting
- **`'docker' is not recognized` / `docker: command not found`** — Docker isn't installed, or the terminal was opened before installing it. Install [Docker](https://docs.docker.com/get-docker/), then open a **new** terminal and try again.
- **`Cannot connect to the Docker daemon` / `error during connect`** — Docker isn't running. Start the Docker Desktop application and wait for it to finish loading, then retry.
- **`permission denied ... docker.sock` (Linux)** — your user can't talk to Docker yet. Either prefix commands with `sudo`, or (better) [add your user to the `docker` group](https://docs.docker.com/engine/install/linux-postinstall/) and log out/in once.
- **`port is already allocated` / `address already in use`** — another program is already using port 5173 or 8000. Close it, or edit the Compose file and change the number on the **left** side of the colon under `ports:` (e.g. `5174:...`), then open that port in the browser instead.
- **The page won't load right after starting** — the very first start can take a few minutes while images are downloaded or built. Watch the progress with the *live logs* command above, then refresh the browser.

## The assistant

The assistant is an AI agent, **PACo**, that works PAC for you. You ask it in plain words — *"Process active_test_profile and give me its dispersion curves"* — and it runs PAC's processing, checks the quality of every step (and retries what it can), picks the dispersion curves, inverts them into velocity models if you asked for them (or into soils and a water table, if you asked for those), then tells you what it did and which settings it changed. Its results are ordinary PAC runs: you open them in the other pages, to review or correct them.

You can keep working in PAC's other pages while it answers: a run one side is writing is refused to the other until it ends. Nothing is deleted: results a step replaces are kept in the window's `replaced` folder. The Assistant entry of the menu shows when it is working, and when an answer waits for your choice.

It needs a language model, which you choose in its settings (below): a provider's API (Claude, OpenAI, Gemini or Mistral, with your key), or your own server, any server speaking OpenAI's chat API (vLLM, Ollama, LM Studio, OpenRouter, a company's gateway). Running a model server is up to you: see its own documentation. Until a model is set, the rest of PAC works as usual.

### Talk to the assistant

Open the **AI assistant** page, type your request and press Enter, for example:
- *Which profiles can I process?*
- *Process active_test_profile and give me its dispersion curves.*
- *Process active_test_profile and invert it: I want its velocity section.*
- *Process active_test_profile and give me the soil types and the water table.* The assistant runs the petrophysical inversion (a Silex model) on the curves the model was trained for: the bundled one needs curves reaching 43 Hz, and the assistant says how many of yours do.

The page shows each step the assistant takes (the tools it calls, in small grey text; untick *Show tool calls* to hide them) and a progress line while it works. Under each answer, the tokens the model read and wrote for it; the list of conversations gives each one's total. Processing a profile takes a few minutes; an inversion, longer. When it has finished, open the **Dispersion Picking**, **Seismic Inversion**, **Petrophysical Inversion** or **Visualization** pages: they open on the latest run, the assistant's.

### The assistant's settings

On the **AI assistant** page, a click on the assistant's state (its mode and model, beside **Conversations**) opens its settings in a popup. The popup also opens by itself when PAC starts while the settings are incomplete, as when `.env` gives none. Nothing is chosen for you: the popup shows what you saved, or what `.env` sets.

- **The mode.** *Autonomous*: the model surveys the line, tries its own settings and judges them with the gates, decides each window, then inverts. Its journal is on the run's card in **Visualization**. It needs a frontier model with vision, since it judges every image by eye, and a long context (128k tokens or more). It uses many more tokens: hundreds of thousands for a line, which an online service bills. *Workflow*: PACo leads the model step by step, and its quality gates check each stage. A small model on your own GPU is enough, such as Qwen3-14B.
- **The model**, the same for both modes, in one of two places: *a provider's API* (Claude, OpenAI, Gemini or Mistral: the model's name and your key), or *your server*, any server speaking OpenAI's chat API (vLLM, Ollama, LM Studio, OpenRouter, a company's gateway: its address, the model's name, its context). **Test the connection** makes one short call to the model, and counts down the minute it may take.

The popup keeps its settings in `data/settings/assistant.env`, the API key with them, readable by you alone (never share or commit it), and applies them over `.env`'s. The settings cannot change while an answer runs: let it finish, or stop it. Saving new settings clears the conversations (their transcripts stay in `data/output/agent_logs`), and your next message starts a new one with them. Unchanged settings cannot be saved: the button then reads *Current settings*.

### What the model's provider receives

Your messages to the assistant, the descriptions of PACo's tools, and what the tools return, that is the quality gates' summaries (profile and run names, settings, verdicts, velocities and depths). In the autonomous mode, its figures (records, dispersion images, models) go to the model too. With a provider's API, check that this suits your data's rules. Providers charge by use: one request, such as processing and picking a profile, takes a few tens of thousands of tokens in the workflow mode, and hundreds of thousands in the autonomous mode.

### If something goes wrong

| What you see | What to do |
|---|---|
| *The assistant is not available: ...* | Open its settings (a click on its state, beside **Conversations**) and **Test the connection**: it says what is wrong, a key, an address or a model's name. |
| The settings popup opens at every start | The settings are incomplete: choose a mode and a model, and save. |
| The assistant is slow | That is normal: it runs PAC's processing, which takes minutes; the progress line under the conversation shows what it is doing. |
| The assistant made a choice you disagree with | Tell it, in the same conversation: *use windows of 24 receivers*, *invert with 3 layers*. **New conversation** starts afresh. |

### For developers (without Docker)
With the source code and [uv](https://docs.astral.sh/uv/):
```sh
uv sync                                              # PAC, its assistant included
uv run uvicorn masw.api.main:app --host 127.0.0.1 --port 8000
```
and the web app with `npm install && npm run dev` in `frontend/`. Set the assistant's model in its settings popup, or in a `.env` file in the `PAC` folder, which the backend reads when started from there (`PACO_LLM_PROVIDER`, `PACO_LLM_MODEL`, `PACO_LLM_API_KEY` or `PACO_LLM_BASE_URL`, `PACO_LLM_CONTEXT`); the popup's settings win.

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
