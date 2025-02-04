# PAC - Passive and Active Computation of MASW

PAC is an app designed for processing 2D Multichannel Analysis of Surface Waves (MASW).
It can handle both passive and active seismic data to automatically optimize and extract dispersion images.
Dispersion curves can be semi-automatically picked on a interactive interface and then be inverted into shear wave velocity profiles.

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
- Required libraries: `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`, `bayesbay`, `streamlit`, `obspy`
- On Mac OS, the GUI for folder selection and subprocess mapping may be incompatible with Streamlit.
- On Mac OS, BayesBay needs to be installed from source and architecture target on Makefile needs to be erased.
- Developed on Linux OS under Streamlit 1.41.1 and Python 3.10.12.

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
- `Home.py`: App home
- `home_images/`: Images for app home
- `pages/`: App tabs (Computing, Dispersion Picking, Inversion, and Visualization)
- `scripts/`: MASW and inversion scripts
- `modules/`: Surface wave signal processing functions
- `Paths.py`: Paths to the PAC main folder, and output and input directories
- `input/`: Contains folders (one per profile) with seismic records
- `output/`: Contains folders (one per profile) with dispersion images, picked dispersion curves and inversion results

## Contributors
- **José Cunha Teixeira**
- **Benjamin Decker** for phase-shift algorithm optimization
  
## License
This project is under Creative Commons Attribution 4.0 International license, allowing re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited.
It is part of the PhD thesis of José Cunha Teixeira and was funded by UMR 7619 METIS (Sorbonne University), Mines Paris - PSL, SNCF Réseau, and the European Union.

## Acknowledgments
This work was developed with contributions from the geophysics research community. Special thanks to open-source developers for their invaluable tools.

![Screenshot from 2025-02-03 13-44-27](https://github.com/user-attachments/assets/59ada0fa-fbf0-4913-8d1a-4799da60a539)
