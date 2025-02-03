# PAC - Passive and Active Computation

PAC (Passive and Active Computation) is a tool designed for processing automatic 2D Multichannel Analysis of Surface Waves (MASW). It can process both passive and active seismic data to extract dispersion curves and estimate subsurface shear wave velocity profiles.

## Features
- **Hybrid Processing:** Supports both passive and active MASW methods without needing source position information.
- **Signal Processing Tools:** Uses an automatic source detection algorithm combined with seismic interferometry.
- **Automated Dispersion Analysis:** Extracts and visualizes dispersion images and curves.
- **Velocity Inversion:** Computes shear wave velocity profiles from dispersion data using the MCMC algorithm BayesBay and direct modeling algorithm Disba.
- **User-Friendly Interface:** Streamlined workflow with visualization capabilities.
- **Python-Based:** Lightweight and extensible for custom modifications.

## Installation
### Requirements
- Python 3.10+
- Required libraries: `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`, `bayesbay`, `streamlit`, `obspy`
- On Mac OS, the GUI for folder selection and subprocess mapping may be incompatible with Streamlit.
- On Mac OS, BaysBay needs to be installed from source and architecture target on Makefile needs to be erased.

### Clone Repository
```sh
git clone https://github.com/JoseCunhaTeixeira/PAC.git
cd PAC/src/
```

## Usage
### Running the app
```sh
streamlit run Home.py
```
Content:
- `input/`: Seismic records
- `src/`: Source code
    - `Paths.py`: Path to you folder (to modify)
    - `Home.py`: App home
    - `images/`: Images for app home
    - `pages/`: App tabs
    - `scripts/`: Signal processing scripts
    - `modules/`: Functions
- `output/`: Dispersion images, picked dispersion curves and inversion results.

## Contributors
- **José Cunha Teixeira**
- **Benjamin Becker**
- *Keurfon Luu for Disba Package*
- *Fabrizio Magrini for BayesBay Package*

## License
This open source project is part of the PhD thesis of José Cunha Teixeira and was funded by UMR 7619 METIS (Sorbonne Université), Mines Paris - PSL, SNCF Réseau, and the European Union.

## Acknowledgments
This work was developed with contributions from the geophysics research community. Special thanks to open-source developers for their invaluable tools.

![Screenshot from 2025-02-03 13-44-27](https://github.com/user-attachments/assets/59ada0fa-fbf0-4913-8d1a-4799da60a539)
