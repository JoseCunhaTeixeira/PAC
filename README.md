# PAC - Passive and Active Computation of MASW

PAC is an app designed for processing 2D Multichannel Analysis of Surface Waves (MASW).
It can handle both passive and active seismic data to automatically optimize and extract dispersion images.
Dispersion curves can be semi-automatically picked on an interactive interface and then be inverted into shear wave velocity profiles.

## Features
- **Hybrid Processing:** Supports both passive and active MASW methods without needing source position information.
- **Signal Processing Tools:** Uses an automatic source detection algorithm combined with seismic interferometry.
- **Automated Dispersion Analysis:** Extracts and visualizes dispersion images and curves.
- **Velocity Inversion:** Computes shear wave velocity profiles from dispersion data using the MCMC algorithm BayesBay and forward modeling algorithm Disba.
- **User-Friendly Interface:** Streamlined workflow with visualization capabilities.
- **Python-Based:** Lightweight and extensible for custom modifications.

## Installation
### Requirements
- Python 3.10+
- Required libraries: `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`, `bayesbay`, `streamlit`, `obspy`, `arviz`, `disba`
- On Mac OS, subprocess mapping may be incompatible with Streamlit, and BayesBay needs to be installed from source with no architecture target specified on the Makefile.
- Developed on Linux OS under Streamlit 1.41.1 and Python 3.10.12.

### Clone Repository
```sh
git clone https://github.com/JoseCunhaTeixeira/PAC.git
cd PAC/
```

## Usage
### Running the app
```sh
streamlit run Home.py
```
if it doesn't work, try
```sh
streamlit run Home.py --server.enableXsrfProtection false
```

Content:
- `Home.py`: App home
- `home_images/`: Images for app home
- `pages/`: App tabs (Computing, Dispersion Picking, Inversion, and Visualization)
- `scripts/`: MASW and inversion scripts
- `modules/`: Surface wave signal processing functions
- `Paths.py`: Paths to the PAC main folder, and output and input directories
- `input/`: Contains one folder per profile with raw seismic records
    - `profile_1/`: seismic files for profile 1
    - `profile_2/`: seismic files for profile 2
    - ...
- `output/`: Contains one folder per profile with dispersion images, picked dispersion curves and inversion results
    - `profile_1/`: results from computing, picking and inversion for profile 1
    - `profile_2/`: results from computing, picking and inversion for profile 2
    - ...
- (`input/` and `output/` are created automatically at app launching if do not exist)

## Contributors
- **José Cunha Teixeira**
- **Benjamin Decker** for phase-shift algorithm optimization
  
## License
This project is under Creative Commons Attribution 4.0 International license, allowing re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited.
It is part of the PhD thesis of José Cunha Teixeira and was funded by a cooperation between Sorbonne University, Mines Paris - PSL, SNCF Réseau, and the European Union.
Please cite as Cunha Teixeira, J. (2025). PAC - Passive and Active Computation of MASW. Zenodo. doi:[10.5281/zenodo.14808813](https://doi.org/10.5281/zenodo.14808813)

## Acknowledgments
This work was developed with contributions from the geophysics research community. Special thanks to open-source developers for their invaluable tools.

![Screenshot from 2025-02-03 13-44-27](https://github.com/user-attachments/assets/59ada0fa-fbf0-4913-8d1a-4799da60a539)
