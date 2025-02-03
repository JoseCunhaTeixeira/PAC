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

### Clone Repository
```sh
git clone https://github.com/JoseCunhaTeixeira/PAC.git
cd pac
```

## Usage
### Running the app
```sh
streamlit run Home.py
```
Folders:
- `input`: Seismic records
- `src`: Codes
- `output`: Dispersion images, picked dispersion curves and inversion results.

## Contributors
- **José Cunha Teixeira**
- **Benjamin Becker**
- *Keurfon Luu for Disba Package*
- *Fabrizio Magrini for BayesBay*

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This work was developed with contributions from the geophysics research community. Special thanks to open-source developers for their invaluable tools.

![Screenshot from 2025-02-03 13-44-27](https://github.com/user-attachments/assets/59ada0fa-fbf0-4913-8d1a-4799da60a539)
