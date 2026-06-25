# PAC - Passive and Active Computation of MASW

New faster version of PAC using a Spark frontend and a new cleaned backend !

PAC is an app designed for processing Multichannel Analysis of Surface Waves (MASW) on linear arrays.
It can handle both passive and active seismic data to automatically optimize and extract dispersion images.
It is also possible to apply cross-correlation to active data to sometimes improve discpersion retreival quality.
Dispersion curves can be semi-automatically picked on an interactive interface and then be inverted into shear wave velocity profiles.


## Features
- **Hybrid Processing:** Supports both passive and active MASW methods without needing source position information. Uses [sigpipe](https://github.com/JoseCunhaTeixeira/sigpipe), a signal processing Python pipeline.
- **Signal Processing Tools:** Uses an automatic source detection algorithm combined with seismic interferometry. 
- **Automated Dispersion Analysis:** Extracts and visualizes dispersion images and curves.
- **Velocity Inversion:** Computes shear wave velocity profiles from dispersion data using the MCMC algorithm [BayesBay](https://bayes-bay.readthedocs.io/en/latest/#) and forward modeling algorithm [Disba](https://github.com/keurfonluu/disba).
- **User-Friendly Interface:** Streamlined workflow with visualization capabilities.
- **Python-Based:** Lightweight and extensible for custom modifications.


## Overview (to be updated)
https://github.com/user-attachments/assets/983d6761-53d0-4f0a-9dff-7742f1432696
  

## Running app
### Requirements
- [Docker](https://www.docker.com/) installed on your machine, with the Compose plugin (`docker compose`, not the older standalone `docker-compose`) — included by default in current Docker Desktop / Docker Engine installs.

There are two ways to run PAC: pull the published images (fastest, no clone), or clone the repo and build them yourself (lets you inspect or modify the source).

### Option 1: Run the published image (no clone needed)
A backend image and a frontend image are built and published to GitHub Container Registry on every push to `main` ([`ghcr.io/josecunhateixeira/pac-backend`](https://github.com/JoseCunhaTeixeira/PAC/pkgs/container/pac-backend), [`ghcr.io/josecunhateixeira/pac-frontend`](https://github.com/JoseCunhaTeixeira/PAC/pkgs/container/pac-frontend)), tagged `latest` plus a `sha-<short-sha>` per commit if you want to pin to (or roll back to) a specific version instead of always tracking the newest one. You can run the app from these without cloning the repo at all:

```sh
mkdir -p data/input data/output
curl -O https://raw.githubusercontent.com/JoseCunhaTeixeira/PAC/main/docker-compose.prod.yml
docker compose -f docker-compose.prod.yml up -d
```

Then open http://localhost:5173 in a browser — `active_p1`/`passive_p1` demo profiles are there to try immediately (see [Data volumes](#data-volumes)).

```sh
docker compose -f docker-compose.prod.yml logs -f        # tail both services
docker compose -f docker-compose.prod.yml pull           # fetch the newest published images
docker compose -f docker-compose.prod.yml up -d          # restart with whatever was just pulled
docker compose -f docker-compose.prod.yml down           # stop (data/ untouched)
```

This assumes you're opening the browser on the same machine running Docker — the published frontend image is built with the API URL baked in as `http://localhost:8000`. Deploying backend and frontend on a separate remote server reachable by its own domain/IP would need the frontend image rebuilt locally with a different `VITE_API_URL` build arg (see Option 2 below).

### Option 2: Build and run docker image

Clone the repo.

```sh
git clone https://github.com/JoseCunhaTeixeira/PAC.git
cd PAC/
```

The app can be run fully containerized (backend + frontend), with no local Python/Node setup required.

```sh
docker compose up --build -d
```

Then open http://localhost:5173 in a browser. `data/input/` and `data/output/` are bind-mounted from the host, so dropping a new profile folder into `data/input/` works exactly as in a native install, and results in `data/output/` persist across container restarts.

```sh
docker compose logs -f             # tail both services
docker compose down                # stop (data/ untouched)
```

To update after pulling new commits, rebuild and restart:
```sh
git pull
docker compose up --build -d
```

### Data volumes
- `data/`
    - `input/`: Contains one folder per profile with your raw seismic records
        - `active_profile_1/`: one shot per seismic file, requires receiver and source positions
            - `file1.segy`
            - `file2.segy`
            - `receiver_positions.yaml`
            - `source_positions.yaml`
        - `passive_profile_2/`: passive recordings, only requires receiver positions
            - `file1.segy`
            - `file2.segy`
            - `receiver_positions.yaml`
    - `output/`: Contains one folder per profile with dispersion and inversion results
        - `active_profile_1/`
        - `passive_profile_2/`

An empty `data/input` (a freshly `mkdir`'d folder, as in Option 1 above) gets seeded on first start with two demo profiles, `active_p1` and `passive_p1`, so there's something to try immediately. A `data/input` that already has content — your own profile folders, or a clone's committed demo data in Option 2 — is left untouched; just drop your own profile folders in alongside or instead of the demo ones. Results only persist across container restarts/removal because `data/output` is bind-mounted — removing that mount (or running the image directly without `-v`) would lose results when the container is removed.


## License
This project is under Creative Commons Attribution 4.0 International license, allowing re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited.
Please cite as:
- Cunha Teixeira, J. (2025). PAC - Passive and Active Computation of MASW. Zenodo. doi:[10.5281/zenodo.14808813](https://doi.org/10.5281/zenodo.14808813)
- Cunha Teixeira, J., Bodet, L., Dangeard, M., Gesret, A., Hallier, A., Rivière, A., Burzawa, A., Cárdenas Chapellín, J. J., Fonda, M., Sanchez Gonzalez, R., Dhemaied, A., & Boisson Gaboriau, J. (2025). Nondestructive testing of railway embankments by measuring multi-modal dispersion of surface waves induced by high-speed trains with linear geophone arrays. Seismica, 4(1). doi:[10.26443/seismica.v4i1.1150](https://doi.org/10.26443/seismica.v4i1.1150)


## Acknowledgments
This work is part of the PhD thesis of José Cunha Teixeira and was funded by a cooperation between Sorbonne University, Mines Paris - PSL, SNCF Réseau, and the European Union’s Horizon Europe research and innovation program under Grant Agreement No 101101966.
It was developed with contributions from the geophysics research community. Special thanks to open-source developers for their invaluable tools.


