"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025

---

Launch the app with the following command:
streamlit run Home.py
or
streamlit run Home.py --server.enableXsrfProtection false
"""

import streamlit as st
import os
from Paths import input_dir, output_dir

if not os.path.exists(f"{input_dir}"):
    os.makedirs(f"{input_dir}")
    
if not os.path.exists(f"{output_dir}"):
    os.makedirs(f"{output_dir}")


# Clear cache and session state
st.cache_data.clear()
st.session_state.clear()


### HEADER ----------------------------------------------------------------------------------------
st.set_page_config(
    layout="centered",
    page_title="Home",
    page_icon="🏠",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:jose.cunha-teixeira@protonmail.com',
        'Report a bug': 'mailto:jose.cunha-teixeira@protonmail.com',
        'About': """This project is under Creative Commons Attribution 4.0 International license, allowing re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited.
                    It was funded by a cooperation between Sorbonne University, Mines Paris - PSL, SNCF Réseau, and the European Union’s Horizon Europe research and innovation program under Grant Agreement No 101101966.
                    Please cite as Cunha Teixeira, J. (2025). PAC - Passive and Active Computation of MASW. Zenodo. doi:[10.5281/zenodo.14808813](https://doi.org/10.5281/zenodo.14808813), and Cunha Teixeira, J., Bodet, L., Dangeard, M., Gesret, A., Hallier, A., Rivière, A., Burzawa, A., Cárdenas Chapellín, J. J., Fonda, M., Sanchez Gonzalez, R., Dhemaied, A., & Boisson Gaboriau, J. (2025). Nondestructive testing of railway embankments by measuring multi-modal dispersion of surface waves induced by high-speed trains with linear sensor arrays. Seismica, 4(1). doi:[10.26443/seismica.v4i1.1150](https://doi.org/10.26443/seismica.v4i1.1150)"""
    }
)

# Home Section
st.title("🏠 Home")
st.header("**PAC** - Passive and Active Computation of Multichannel Analysis of Surface Waves")

# Main Description
st.write(
    """
    👋 Welcome to **PAC**!
    
    📚 This app allows you to perform 2D surface wave dispersion analysis using both traffic-induced seismic noise and conventional active sources.
    
    The surface wave processing workflow is based on the methodology outlined in [Cunha Teixeira et al. (2024)](https://doi.org/10.26443/seismica.v4i1.1150).
    The inversion of the dispersion curves is performed using the MCMC package [BayesBay](https://bayes-bay.readthedocs.io/en/latest/#), and forward modeling package [Disba](https://github.com/keurfonluu/disba).

    """
)
st.info("ⓘ For more information, please visit our [GitHub repository](https://github.com/JoseCunhaTeixeira/PAC).")

st.text("")
st.text("")
st.text("")
st.text("")

st.image("./home_images/logo.png")

st.text("")
st.text("")

st.image("./home_images/logo2.png")
### -----------------------------------------------------------------------------------------------
