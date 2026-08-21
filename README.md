# PAC — Passive and Active Computation of MASW

New faster version of PAC, rebuilt with a React frontend and a cleaned-up backend!

PAC is an app for processing **Multichannel Analysis of Surface Waves (MASW)** on linear arrays. It can handle both **passive** and **active** seismic data to automatically optimize and extract dispersion images, and it can apply cross-correlation to active data to sometimes improve dispersion retrieval quality. Dispersion curves can be semi-automatically picked on an interactive interface and then inverted into shear-wave velocity profiles.

PAC runs entirely **on your own computer**. You start it once, then use it through your web browser at a local address (`http://localhost:5173`). Nothing is uploaded anywhere — your data never leaves your machine.

## Features
- **Hybrid processing:** supports both passive and active MASW methods without needing source position information. Uses the package [sigpipe](https://github.com/JoseCunhaTeixeira/sigpipe), a signal processing Python pipeline.
- **Signal processing tools:** uses an automatic source detection algorithm combined with seismic interferometry.
- **Automated dispersion analysis:** extracts and visualizes dispersion images and curves.
- **Velocity inversion:** computes shear-wave velocity profiles from dispersion data using the MCMC package [BayesBay](https://bayes-bay.readthedocs.io/en/latest/#) and the forward modeling package [Disba](https://github.com/keurfonluu/disba).
- **User-friendly interface:** streamlined workflow with visualization capabilities.
- **Python-based:** lightweight and extensible for custom modifications.

## Overview
<img width="2186" height="1992" alt="image" src="https://github.com/user-attachments/assets/fcd8642e-970d-411c-8d52-441b75cb72e5" />

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
A backend image and a frontend image are built and published to GitHub Container Registry on every push to `main` ([`ghcr.io/josecunhateixeira/pac-backend`](https://github.com/JoseCunhaTeixeira/PAC/pkgs/container/pac-backend), [`ghcr.io/josecunhateixeira/pac-frontend`](https://github.com/JoseCunhaTeixeira/PAC/pkgs/container/pac-frontend)). Docker downloads and runs them for you — no source code needed. (Advanced: besides `latest`, every commit is also tagged `sha-<short-sha>`, so you can pin to or roll back to a specific version instead of always tracking the newest one.)

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

> **Note:** this setup assumes the browser and Docker run on the same machine — the published frontend image expects the backend at `http://localhost:8000`. To serve PAC from a remote server reached by its own domain/IP, you'd need to rebuild the frontend image with a different `VITE_API_URL` build argument (see Option 2 below).

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
    - `input/`: contains one folder per profile with your raw seismic records
        - `active_profile_1/`: one shot per seismic file, requires receiver **and** source positions
            - `file1.segd`
            - `file2.segd`
            - `receiver_positions.yaml`
            - `source_positions.yaml`
        - `passive_profile_2/`: passive recordings, only requires receiver positions
            - `file1.segd`
            - `file2.segd`
            - `receiver_positions.yaml`
    - `output/`: contains one folder per profile with dispersion and inversion results
        - `active_profile_1/`
        - `passive_profile_2/`

To process your own data, simply drop a profile folder like the above into `data/input/` — no restart needed.

**About the demo profiles:** an empty `data/input` (a freshly created folder, as in Option 1) gets seeded on first start with two demo profiles, `active_p1` and `passive_p1`, so there's something to try immediately. A `data/input` that already has content — your own profile folders, or the demo data committed with the source code in Option 2 — is left untouched; add your own profile folders alongside or instead of the demo ones.

**About your results:** results survive stopping/removing the containers only because `data/output` is shared with your computer (bind-mounted). Removing that mount — or running the image directly without `-v` — would lose the results when the container is removed.

### Troubleshooting
- **`'docker' is not recognized` / `docker: command not found`** — Docker isn't installed, or the terminal was opened before installing it. Install [Docker](https://docs.docker.com/get-docker/), then open a **new** terminal and try again.
- **`Cannot connect to the Docker daemon` / `error during connect`** — Docker isn't running. Start the Docker Desktop application and wait for it to finish loading, then retry.
- **`permission denied ... docker.sock` (Linux)** — your user can't talk to Docker yet. Either prefix commands with `sudo`, or (better) [add your user to the `docker` group](https://docs.docker.com/engine/install/linux-postinstall/) and log out/in once.
- **`port is already allocated` / `address already in use`** — another program is already using port 5173 or 8000. Close it, or edit the Compose file and change the number on the **left** side of the colon under `ports:` (e.g. `5174:...`), then open that port in the browser instead.
- **The page won't load right after starting** — the very first start can take a few minutes while images are downloaded or built. Watch the progress with the *live logs* command above, then refresh the browser.

## License
This project is under Creative Commons Attribution 4.0 International license, allowing re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited.
Please cite as:
- Cunha Teixeira, J. (2025). PAC - Passive and Active Computation of MASW. Zenodo. doi:[10.5281/zenodo.14808813](https://doi.org/10.5281/zenodo.14808813)


## Acknowledgments
This work was developed with contributions from the geophysics research community. Special thanks to open-source developers for their invaluable tools.
The algorithms are based on the PhD thesis of José Cunha Teixeira, funded by a cooperation between Sorbonne University, Mines Paris - PSL, SNCF Réseau, and the European Union's Horizon Europe research and innovation program under Grant Agreement No 101101966.
Please refer to:
- Cunha Teixeira, J., Bodet, L., Dangeard, M., Gesret, A., Hallier, A., Rivière, A., Burzawa, A., Cárdenas Chapellín, J. J., Fonda, M., Sanchez Gonzalez, R., Dhemaied, A., & Boisson Gaboriau, J. (2025). Nondestructive testing of railway embankments by measuring multi-modal dispersion of surface waves induced by high-speed trains with linear geophone arrays. Seismica, 4(1). doi:[10.26443/seismica.v4i1.1150](https://doi.org/10.26443/seismica.v4i1.1150)
