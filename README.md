# PAC - Passive and Active Computation of MASW

New faster version of PAC using a Spark frontend and a new cleaned backend !

PAC is an app designed for processing 2D Multichannel Analysis of Surface Waves (MASW).
It can handle both passive and active seismic data to automatically optimize and extract dispersion images.
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
  

## Runing app
### Requirements
- Requires [Docker](https://www.docker.com/) installed in your machine.

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

### Data volumes
- `data/`
    - `input/`: Contains one folder per profile with your raw seismic records
        - `active_profile_1/`: one shot per seimic file, requires receiver and source positions
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


## License
This project is under Creative Commons Attribution 4.0 International license, allowing re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited.
Please cite as:
- Cunha Teixeira, J. (2025). PAC - Passive and Active Computation of MASW. Zenodo. doi:[10.5281/zenodo.14808813](https://doi.org/10.5281/zenodo.14808813)
- Cunha Teixeira, J., Bodet, L., Dangeard, M., Gesret, A., Hallier, A., Rivière, A., Burzawa, A., Cárdenas Chapellín, J. J., Fonda, M., Sanchez Gonzalez, R., Dhemaied, A., & Boisson Gaboriau, J. (2025). Nondestructive testing of railway embankments by measuring multi-modal dispersion of surface waves induced by high-speed trains with linear geophone arrays. Seismica, 4(1). doi:[10.26443/seismica.v4i1.1150](https://doi.org/10.26443/seismica.v4i1.1150)


## Acknowledgments
This work is part of the PhD thesis of José Cunha Teixeira and was funded by a cooperation between Sorbonne University, Mines Paris - PSL, SNCF Réseau, and the European Union’s Horizon Europe research and innovation program under Grant Agreement No 101101966.
It was developed with contributions from the geophysics research community. Special thanks to open-source developers for their invaluable tools.


