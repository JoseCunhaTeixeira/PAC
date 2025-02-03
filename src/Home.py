"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : Decdember 18, 2024

Compile C++ functions:
python3 setup.py build_ext --inplace clean

Launch the app with the following command:
streamlit run Home.py
or
streamlit run Home.py --server.enableXsrfProtection false
"""




import os
import streamlit as st

from Paths import input_dir, output_dir



### HEADER ----------------------------------------------------------------------------------------
st.set_page_config(
    layout="centered",
    page_title="Home",
    page_icon="🏠",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:jose.cunha-teixeira@protonmail.com',
        'Report a bug': 'mailto:jose.cunha-teixeira@protonmail.com',
        'About': "This app was developped by José Cunha Teixeira in the context of a PhD thesis at UMR 7619 METIS (Sorbonne Université) and Mines Paris - PSL, funded by SNCF Réseau."
    }
)

# Home Section
st.title("🏠 Home")
st.header("**PAC** - Passive and Active Computation")
st.subheader("Multichannel Analysis of Surface Waves")

# Main Description
st.write(
    """
    👋 Welcome to **PAC**!
    
    📚 This app allows you to perform 2D surface wave dispersion analysis using both traffic-induced seismic noise and conventional active sources.
    
    The surface wave processing workflow is based on the methodology outlined in [Cunha Teixeira et al. (2024)](https://doi.org/10.26443/seismica.v4i1.1150). The inversion of the dispersion curves is performed using the MCMC package [BayesBay](https://bayes-bay.readthedocs.io/en/latest/#).
    """
)
st.info("ⓘ For more information, please visit our [GitHub repository](https://github.com/JoseCunhaTeixeira/PAC).")

st.text("")
st.text("")
st.text("")
st.text("")


st.image("./images/logo.png")

st.text("")
st.text("")

st.image("./images/logo2.png")

st.text("")
st.text("")
st.text("")
st.text("")
st.write(f"*streamlit version: {st.__version__}*")


if not os.path.exists(f"{input_dir}"):
    os.makedirs(f"{input_dir}")
    
if not os.path.exists(f"{output_dir}"):
    os.makedirs(f"{output_dir}")
### -----------------------------------------------------------------------------------------------
