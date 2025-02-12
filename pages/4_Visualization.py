"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 20252025
"""

import os
import sys
import glob
import numpy as np
import streamlit as st
from obspy import read
import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import generic_filter

from Paths import input_dir, output_dir

sys.path.append("./modules/")
from dispersion import resamp_wavelength, resamp_frequency
from display import plot_pseudo_section, plot_wiggle, plot_disp, plot_inverted_section, plot_std_section, display_inverted_section, display_pseudo_sections
from obspy2numpy import stream_to_array
from misc import arange

import warnings
warnings.filterwarnings("ignore")



### FUNCTIONS ---------------------------------------------------------------------------------------------------------------------------------------
def clear_session():
    st.cache_data.clear()
    st.session_state.clear()

def initialize_session():
    for key in st.session_state:
        if 'VIZ' not in key:
            st.session_state.pop(key)
    if "VIZ_mode" not in st.session_state:
        st.session_state.VIZ_mode = None
    if 'VIZ_selected_folder' not in st.session_state:
        st.session_state.VIZ_selected_folder = None
    if 'VIZ_folder_path' not in st.session_state:
        st.session_state.VIZ_folder_path = None

    
    if 'VIZ_v_xd_median_layered' not in st.session_state:
        st.session_state.VIZ_v_xd_median_layered = None
    if 'VIZ_v_xd_median_layered_lateralsmooth' not in st.session_state:
        st.session_state.VIZ_v_xd_median_layered_lateralsmooth = None

    if 'VIZ_v_xd_verticalsmooth_median_layered' not in st.session_state:
        st.session_state.VIZ_v_xd_verticalsmooth_median_layered = None
    if 'VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth' not in st.session_state:
        st.session_state.VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth = None


    if 'VIZ_v_xd_best_layered' not in st.session_state:
        st.session_state.VIZ_v_xd_best_layered = None
    if 'VIZ_v_xd_best_layered_lateralsmooth' not in st.session_state:
        st.session_state.VIZ_v_xd_best_layered_lateralsmooth = None

    if 'VIZ_v_xd_verticalsmooth_best_layered' not in st.session_state:
        st.session_state.VIZ_v_xd_verticalsmooth_best_layered = None
    if 'VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth' not in st.session_state:
        st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth = None

    
    if 'VIZ_v_xd_median_ensemble' not in st.session_state:
        st.session_state.VIZ_v_xd_median_ensemble = None
    if 'VIZ_v_xd_median_ensemble_lateralsmooth' not in st.session_state:
        st.session_state.VIZ_v_xd_median_ensemble_lateralsmooth = None


    if 'VIZ_cmap' not in st.session_state:
        st.session_state.VIZ_cmap = 'Terrain'
    if 'VIZ_model' not in st.session_state:
        st.session_state.VIZ_model = 'Median layered'
    if 'VIZ_vertical_smoothing' not in st.session_state:
        st.session_state.VIZ_smoothing = 'Yes'
    if 'VIZ_lateral_smoothing' not in st.session_state:
        st.session_state.VIZ_smoothing = 'Yes'

        
def get_subplot_layout(n_subplots):
    rows = int(np.sqrt(n_subplots))
    cols = int(np.ceil(n_subplots / rows))
    return rows, cols

def mode_filter_median(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mode = np.nanmedian(values)
    return mode
### -------------------------------------------------------------------------------------------------------------------------------------------------



### HANDLERS ----------------------------------------------------------------------------------------------------------------------------------------
def load_sections():
    
    all_gm_median_layered = []
    all_std_median_layered = []
    for folder in st.session_state.VIZ_folders:
        try:
            gm = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_layered_model.gm", skiprows=1)
            std = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_layered_std.gm", skiprows=1)
        except:
            gm = None
            std = None
        all_gm_median_layered.append(gm)
        all_std_median_layered.append(std)

    all_gm_verticalsmooth_median_layered = []
    all_std_verticalsmooth_median_layered = []
    for folder in st.session_state.VIZ_folders:
        try:
            gm = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_smooth_median_layered_model.gm", skiprows=1)
            std = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_smooth_median_layered_std.gm", skiprows=1)
        except:
            gm = None
            std = None
        all_gm_verticalsmooth_median_layered.append(gm)
        all_std_verticalsmooth_median_layered.append(std)


    all_gm_best_layered = []
    for folder in st.session_state.VIZ_folders:
        try:
            gm = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_best_layered_model.gm", skiprows=1)
        except:
            gm = None
            std = None
        all_gm_best_layered.append(gm)

    all_gm_verticalsmooth_best_layered = []
    for folder in st.session_state.VIZ_folders:
        try:
            gm = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_smooth_best_layered_model.gm", skiprows=1)
        except:
            gm = None
            std = None
        all_gm_verticalsmooth_best_layered.append(gm)

    
    all_gm_median_ensemble = []
    all_std_median_ensemble = []
    for folder in st.session_state.VIZ_folders:
        try:
            gm = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_ensemble_model.gm", skiprows=1)
            std = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_ensemble_std.gm", skiprows=1)
        except:
            gm = None
            std = None
        all_gm_median_ensemble.append(gm)
        all_std_median_ensemble.append(std)       
    

    if not all(gm is None for gm in all_gm_median_layered):
        max_depth_median_layered = np.nanmax([np.sum(gm[:,0]) for gm in all_gm_median_layered if gm is not None])
    else:
        max_depth_median_layered = np.nan
    if not all(gm is None for gm in all_gm_verticalsmooth_median_layered):
        max_depth_vertical_smooth_median_layered = np.nanmax([np.sum(gm[:,0]) for gm in all_gm_verticalsmooth_median_layered if gm is not None])
    else:
        max_depth_vertical_smooth_median_layered = np.nan
    if not all(gm is None for gm in all_gm_best_layered):
        max_depth_best_layered = np.nanmax([np.sum(gm[:,0]) for gm in all_gm_best_layered if gm is not None])
    else:
        max_depth_best_layered = np.nan
    if not all(gm is None for gm in all_gm_verticalsmooth_best_layered):
        max_depth_vertical_smooth_best_layered = np.nanmax([np.sum(gm[:,0]) for gm in all_gm_verticalsmooth_best_layered if gm is not None])
    else:
        max_depth_vertical_smooth_best_layered = np.nan
    if not all(gm is None for gm in all_gm_median_ensemble):
        max_depth_median_ensemble = np.nanmax([np.sum(gm[:,0]) for gm in all_gm_median_ensemble if gm is not None])
    else:
        max_depth_median_ensemble = np.nan
    depth_max = np.nanmax([max_depth_median_layered, max_depth_vertical_smooth_median_layered, max_depth_best_layered, max_depth_vertical_smooth_best_layered, max_depth_median_ensemble])
    dz = 0.01
    

    if not all(gm is None for gm in all_gm_median_layered) and not all(gm is None for gm in all_std_median_layered):
        st.session_state.VIZ_depths_median_layered = arange(0, depth_max, dz)
        
        st.session_state.VIZ_v_xd_median_layered = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_median_layered)), np.nan)
        st.session_state.VIZ_std_xd_median_layered = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_median_layered)), np.nan)
                        
        for j, (gm, gm_std) in enumerate(zip(all_gm_median_layered, all_std_median_layered)):
            if gm is not None and gm_std is not None:
                col = []
                col_std = []
                for (thick, vp, vs, rho), (thick_std, vp_std, vs_std, rho_std) in zip(gm, gm_std):
                    col += [vs] * int(thick/dz)
                    col_std += [vs_std] * int(thick/dz)
                if len(col) < len(st.session_state.VIZ_depths_median_layered):
                    col += [gm[-1,2]] * (len(st.session_state.VIZ_depths_median_layered)-len(col))
                    col_std += [gm_std[-1,2]] * (len(st.session_state.VIZ_depths_median_layered)-len(col_std))
                st.session_state.VIZ_v_xd_median_layered[j, :] = col
                st.session_state.VIZ_std_xd_median_layered[j, :] = col_std
                
        st.session_state.VIZ_vs_min_median_layered = np.floor(np.nanmin(st.session_state.VIZ_v_xd_median_layered))
        st.session_state.VIZ_vs_max_median_layered = np.ceil(np.nanmax(st.session_state.VIZ_v_xd_median_layered))
        
        st.session_state.VIZ_v_xd_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_median_layered, mode_filter_median, size=(4,1))
        st.session_state.VIZ_v_xd_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_median_layered_lateralsmooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_v_xd_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_median_layered_lateralsmooth, mode_filter_median, size=(2,1))
        
        st.session_state.VIZ_std_xd_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_std_xd_median_layered, mode_filter_median, size=(4,1))
        st.session_state.VIZ_std_xd_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_std_xd_median_layered_lateralsmooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_std_xd_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_std_xd_median_layered_lateralsmooth, mode_filter_median, size=(2,1))
    else:
        st.session_state.VIZ_v_xd_median_layered = None
        st.session_state.VIZ_v_xd_median_layered_lateralsmooth = None
        st.session_state.VIZ_std_xd_median_layered = None
        st.session_state.VIZ_std_xd_median_layered_lateralsmooth = None
        st.session_state.VIZ_depths_median_layered = None
        st.session_state.VIZ_vs_min_median_layered = None
        st.session_state.VIZ_vs_max_median_layered = None

    if not all(gm is None for gm in all_gm_verticalsmooth_median_layered) and not all(gm is None for gm in all_std_verticalsmooth_median_layered):
        st.session_state.VIZ_depths_verticalsmooth_median_layered = arange(0, depth_max, dz)
        
        st.session_state.VIZ_v_xd_verticalsmooth_median_layered = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_verticalsmooth_median_layered)), np.nan)
        st.session_state.VIZ_std_xd_verticalsmooth_median_layered = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_verticalsmooth_median_layered)), np.nan)
                        
        for j, (gm, gm_std) in enumerate(zip(all_gm_verticalsmooth_median_layered, all_std_verticalsmooth_median_layered)):
            if gm is not None and gm_std is not None:
                col = []
                col_std = []
                for (thick, vp, vs, rho), (thick_std, vp_std, vs_std, rho_std) in zip(gm, gm_std):
                    col += [vs] * int(thick/dz)
                    col_std += [vs_std] * int(thick/dz)
                if len(col) < len(st.session_state.VIZ_depths_verticalsmooth_median_layered):
                    col += [gm[-1,2]] * (len(st.session_state.VIZ_depths_verticalsmooth_median_layered)-len(col))
                    col_std += [gm_std[-1,2]] * (len(st.session_state.VIZ_depths_verticalsmooth_median_layered)-len(col_std))
                st.session_state.VIZ_v_xd_verticalsmooth_median_layered[j, :] = col
                st.session_state.VIZ_std_xd_verticalsmooth_median_layered[j, :] = col_std
                
        st.session_state.VIZ_vs_min_verticalsmooth_median_layered = np.floor(np.nanmin(st.session_state.VIZ_v_xd_verticalsmooth_median_layered))
        st.session_state.VIZ_vs_max_verticalsmooth_median_layered = np.ceil(np.nanmax(st.session_state.VIZ_v_xd_verticalsmooth_median_layered))
        
        st.session_state.VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_verticalsmooth_median_layered, mode_filter_median, size=(4,1))
        st.session_state.VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth, mode_filter_median, size=(2,1))
        
        st.session_state.VIZ_std_xd_verticalsmooth_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_std_xd_verticalsmooth_median_layered, mode_filter_median, size=(4,1))
        st.session_state.VIZ_std_xd_verticalsmooth_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_std_xd_verticalsmooth_median_layered_lateralsmooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_std_xd_verticalsmooth_median_layered_lateralsmooth = generic_filter(st.session_state.VIZ_std_xd_verticalsmooth_median_layered_lateralsmooth, mode_filter_median, size=(2,1))
    else:
        st.session_state.VIZ_v_xd_verticalsmooth_median_layered = None
        st.session_state.VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth = None
        st.session_state.VIZ_std_xd_verticalsmooth_median_layered = None
        st.session_state.VIZ_std_xd_verticalsmooth_median_layered_lateralsmooth = None
        st.session_state.VIZ_depths_verticalsmooth_median_layered = None
        st.session_state.VIZ_vs_min_verticalsmooth_median_layered = None
        st.session_state.VIZ_vs_max_verticalsmooth_median_layered = None

    
    if not all(gm is None for gm in all_gm_best_layered):
        st.session_state.VIZ_depths_best_layered = arange(0, depth_max, dz)
        
        st.session_state.VIZ_v_xd_best_layered = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_best_layered)), np.nan)
                        
        for j, gm in enumerate(all_gm_best_layered):
            if gm is not None:
                col = []
                for (thick, vp, vs, rho) in gm:
                    col += [vs] * int(thick/dz)
                if len(col) < len(st.session_state.VIZ_depths_best_layered):
                    col += [gm[-1,2]] * (len(st.session_state.VIZ_depths_best_layered)-len(col))
                st.session_state.VIZ_v_xd_best_layered[j, :] = col
                
        st.session_state.VIZ_vs_min_best_layered = np.floor(np.nanmin(st.session_state.VIZ_v_xd_best_layered))
        st.session_state.VIZ_vs_max_best_layered = np.ceil(np.nanmax(st.session_state.VIZ_v_xd_best_layered))
        
        st.session_state.VIZ_v_xd_best_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_best_layered, mode_filter_median, size=(4,1))
        st.session_state.VIZ_v_xd_best_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_best_layered_lateralsmooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_v_xd_best_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_best_layered_lateralsmooth, mode_filter_median, size=(2,1))
        
    else:
        st.session_state.VIZ_v_xd_best_layered = None
        st.session_state.VIZ_v_xd_best_layered_lateralsmooth = None
        st.session_state.VIZ_depths_best_layered = None
        st.session_state.VIZ_vs_min_best_layered = None
        st.session_state.VIZ_vs_max_best_layered = None

    if not all(gm is None for gm in all_gm_verticalsmooth_best_layered):
        st.session_state.VIZ_depths_verticalsmooth_best_layered = arange(0, depth_max, dz)
        
        st.session_state.VIZ_v_xd_verticalsmooth_best_layered = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_verticalsmooth_best_layered)), np.nan)
                        
        for j, gm in enumerate(all_gm_verticalsmooth_best_layered):
            if gm is not None:
                col = []
                for (thick, vp, vs, rho) in gm:
                    col += [vs] * int(thick/dz)
                if len(col) < len(st.session_state.VIZ_depths_verticalsmooth_best_layered):
                    col += [gm[-1,2]] * (len(st.session_state.VIZ_depths_verticalsmooth_best_layered)-len(col))
                st.session_state.VIZ_v_xd_verticalsmooth_best_layered[j, :] = col
                
        st.session_state.VIZ_vs_min_verticalsmooth_best_layered = np.floor(np.nanmin(st.session_state.VIZ_v_xd_verticalsmooth_best_layered))
        st.session_state.VIZ_vs_max_verticalsmooth_best_layered = np.ceil(np.nanmax(st.session_state.VIZ_v_xd_verticalsmooth_best_layered))
        
        st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_verticalsmooth_best_layered, mode_filter_median, size=(4,1))
        st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth, mode_filter_median, size=(2,1))
        
    else:
        st.session_state.VIZ_v_xd_verticalsmooth_best_layered = None
        st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth = None
        st.session_state.VIZ_depths_verticalsmooth_best_layered = None
        st.session_state.VIZ_vs_min_verticalsmooth_best_layered = None
        st.session_state.VIZ_vs_max_verticalsmooth_best_layered = None

    
    if not all(gm is None for gm in all_gm_median_ensemble) and not all(gm is None for gm in all_std_median_ensemble):
        st.session_state.VIZ_depths_median_ensemble = arange(0, depth_max, dz)
        
        st.session_state.VIZ_v_xd_median_ensemble = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_median_ensemble)), np.nan)
        st.session_state.VIZ_std_xd_median_ensemble = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_median_ensemble)), np.nan)
                        
        for j, (gm, gm_std) in enumerate(zip(all_gm_median_ensemble, all_std_median_ensemble)):
            if gm is not None and gm_std is not None:
                col = []
                col_std = []
                for (thick, vp, vs, rho), (thick_std, vp_std, vs_std, rho_std) in zip(gm, gm_std):
                    col += [vs] * int(thick/dz)
                    col_std += [vs_std] * int(thick/dz)
                if len(col) < len(st.session_state.VIZ_depths_median_ensemble):
                    col += [gm[-1,2]] * (len(st.session_state.VIZ_depths_median_ensemble)-len(col))
                    col_std += [gm_std[-1,2]] * (len(st.session_state.VIZ_depths_median_ensemble)-len(col_std))
                st.session_state.VIZ_v_xd_median_ensemble[j, :] = col
                st.session_state.VIZ_std_xd_median_ensemble[j, :] = col_std
                
        st.session_state.VIZ_vs_min_median_ensemble = np.floor(np.nanmin(st.session_state.VIZ_v_xd_median_ensemble))
        st.session_state.VIZ_vs_max_median_ensemble = np.ceil(np.nanmax(st.session_state.VIZ_v_xd_median_ensemble))
        
        st.session_state.VIZ_v_xd_median_ensemble_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_median_ensemble, mode_filter_median, size=(4,1))
        st.session_state.VIZ_v_xd_median_ensemble_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_median_ensemble_lateralsmooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_v_xd_median_ensemble_lateralsmooth = generic_filter(st.session_state.VIZ_v_xd_median_ensemble_lateralsmooth, mode_filter_median, size=(2,1))
        
        st.session_state.VIZ_std_xd_median_ensemble_lateralsmooth = generic_filter(st.session_state.VIZ_std_xd_median_ensemble, mode_filter_median, size=(4,1))
        st.session_state.VIZ_std_xd_median_ensemble_lateralsmooth = generic_filter(st.session_state.VIZ_std_xd_median_ensemble_lateralsmooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_std_xd_median_ensemble_lateralsmooth = generic_filter(st.session_state.VIZ_std_xd_median_ensemble_lateralsmooth, mode_filter_median, size=(2,1))
    else:
        st.session_state.VIZ_v_xd_median_ensemble = None
        st.session_state.VIZ_v_xd_median_ensemble_lateralsmooth = None
        st.session_state.VIZ_std_xd_median_ensemble = None
        st.session_state.VIZ_std_xd_median_ensemble_lateralsmooth = None
        st.session_state.VIZ_depths_median_ensemble = None
        st.session_state.VIZ_vs_min_median_ensemble = None
        st.session_state.VIZ_vs_max_median_ensemble = None

    
    if 'VIZ_cmap' in st.session_state:
        st.session_state.VIZ_cmap = 'Terrain'
    if 'VIZ_model' in st.session_state:
        st.session_state.VIZ_model = 'Median layered'
    if 'VIZ_vertical_smoothing' in st.session_state:
        st.session_state.VIZ_vertical_smoothing = 'Yes'
    if 'VIZ_lateral_smoothing' in st.session_state:
        st.session_state.VIZ_lateral_smoothing = 'Yes'

        
def save_images():
    # Read picked modes xmid folder
    nb_picked_modes_by_position = []
    picked_modes_by_position = []
    for folder, pos in zip(st.session_state.VIZ_folders, st.session_state.VIZ_xmids):
        if os.path.exists(f"{st.session_state.VIZ_folder_path}/{folder}/pick/"):
            picked_modes = [int(fname.split("_M")[1].split(".")[0]) for fname in os.listdir(f"{st.session_state.VIZ_folder_path}/{folder}/pick") if fname.endswith(".pvc") and fname.startswith(f"xmid{pos}_obs_M")]
            picked_modes = sorted(picked_modes)
        else:
            picked_modes = []
        picked_modes_by_position.append(picked_modes)
        nb_picked_modes_by_position.append(len(picked_modes))

    # Count distinct modes
    distinct_modes = {}
    for picked_modes in picked_modes_by_position:
        for picked_mode in picked_modes:
            if picked_mode not in distinct_modes.keys():
                distinct_modes[picked_mode] = 1
            else:
                distinct_modes[picked_mode] += 1
    distinct_modes = dict(sorted(distinct_modes.items()))
    
    path_name = f"{st.session_state.VIZ_folder_path}/vs_section.svg"
    
            
    if st.session_state.VIZ_model == 'Median layered':
        if st.session_state.VIZ_vertical_smoothing == 'No':
            if st.session_state.VIZ_lateral_smoothing == 'No':
                if st.session_state.VIZ_v_xd_median_layered is not None:
                    display_inverted_section(vs_section=st.session_state.VIZ_v_xd_median_layered,
                                            std_section=st.session_state.VIZ_std_xd_median_layered,
                                            positions=st.session_state.VIZ_xmids,
                                            depths=st.session_state.VIZ_depths_median_layered,
                                            path=path_name,
                                            zmin=st.session_state.VIZ_vmin_vmax[0],
                                            zmax=st.session_state.VIZ_vmin_vmax[1],
                                            cmap=st.session_state.VIZ_cmap)
            elif st.session_state.VIZ_lateral_smoothing == 'Yes':
                if st.session_state.VIZ_v_xd_median_layered_lateralsmooth is not None:
                    display_inverted_section(vs_section=st.session_state.VIZ_v_xd_median_layered_lateralsmooth,
                                            std_section=st.session_state.VIZ_std_xd_median_layered_lateralsmooth,
                                            positions=st.session_state.VIZ_xmids,
                                            depths=st.session_state.VIZ_depths_median_layered,
                                            path=path_name,
                                            zmin=st.session_state.VIZ_vmin_vmax[0],
                                            zmax=st.session_state.VIZ_vmin_vmax[1],
                                            cmap=st.session_state.VIZ_cmap)
        elif st.session_state.VIZ_vertical_smoothing == 'Yes':
            if st.session_state.VIZ_lateral_smoothing == 'No':
                if st.session_state.VIZ_v_xd_verticalsmooth_median_layered is not None:
                    display_inverted_section(vs_section=st.session_state.VIZ_v_xd_verticalsmooth_median_layered,
                                            std_section=st.session_state.VIZ_std_xd_verticalsmooth_median_layered,
                                            positions=st.session_state.VIZ_xmids,
                                            depths=st.session_state.VIZ_depths_verticalsmooth_median_layered,
                                            path=path_name,
                                            zmin=st.session_state.VIZ_vmin_vmax[0],
                                            zmax=st.session_state.VIZ_vmin_vmax[1],
                                            cmap=st.session_state.VIZ_cmap)
            elif st.session_state.VIZ_lateral_smoothing == 'Yes':
                if st.session_state.VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth is not None:
                    display_inverted_section(vs_section=st.session_state.VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth,
                                            std_section=st.session_state.VIZ_std_xd_verticalsmooth_median_layered_lateralsmooth,
                                            positions=st.session_state.VIZ_xmids,
                                            depths=st.session_state.VIZ_depths_verticalsmooth_median_layered,
                                            path=path_name,
                                            zmin=st.session_state.VIZ_vmin_vmax[0],
                                            zmax=st.session_state.VIZ_vmin_vmax[1],
                                            cmap=st.session_state.VIZ_cmap)
        for mode in distinct_modes.keys():
            # Observed
            obs_fs_per_position = []
            obs_vs_per_position = []
            for j, folder in enumerate(st.session_state.VIZ_folders):
                try :
                    pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/pick/{folder}_obs_M{mode}.pvc")
                    if len(pvc.shape) == 1:
                            pvc = pvc.reshape(1,-1)
                    pvc = np.round(pvc, 2)
                    fs = pvc[:,0]
                    vs = pvc[:,1]
                    fs, vs = resamp_frequency(fs, vs)
                    obs_fs_per_position.append(fs)
                    obs_vs_per_position.append(vs)
                except:
                    obs_fs_per_position.append([])
                    obs_vs_per_position.append([])
            # Predicted
            pred_fs_per_position = []
            pred_vs_per_position = []
            for j, folder in enumerate(st.session_state.VIZ_folders):
                try :
                    if st.session_state.VIZ_vertical_smoothing == 'No':
                        pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_layered_M{mode}.pvc")
                    elif st.session_state.VIZ_vertical_smoothing == 'Yes':
                        pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_smooth_median_layered_M{mode}.pvc")
                    if len(pvc.shape) == 1:
                            pvc = pvc.reshape(1,-1)
                    pvc = np.round(pvc, 2)
                    fs = pvc[:,0]
                    vs = pvc[:,1]
                    fs, vs = resamp_frequency(fs, vs)
                    pred_fs_per_position.append(fs)
                    pred_vs_per_position.append(vs)
                except:
                    pred_fs_per_position.append([])
                    pred_vs_per_position.append([])
            # General
            obs_f_min = np.min([np.min(fs) for fs in obs_fs_per_position if len(fs) > 0])
            obs_f_max = np.max([np.max(fs) for fs in obs_fs_per_position if len(fs) > 0])
            if not all(len(fs) == 0 for fs in pred_fs_per_position):
                pred_f_min = np.min([np.min(fs) for fs in pred_fs_per_position if len(fs) > 0])
                pred_f_max = np.max([np.max(fs) for fs in pred_fs_per_position if len(fs) > 0])
            else:
                pred_f_min = obs_f_max
                pred_f_max = obs_f_max
            f_min = np.min([obs_f_min, pred_f_min])
            f_max = np.max([obs_f_max, pred_f_max])
            fs = arange(f_min, f_max, 1)
            pred_v_fx = np.full((len(st.session_state.VIZ_xmids), len(fs)), np.nan)
            obs_v_fx = np.full((len(st.session_state.VIZ_xmids), len(fs)), np.nan)
            for j, (obs_fs, obs_vs, pred_fs, pred_vs) in enumerate(zip(obs_fs_per_position, obs_vs_per_position, pred_fs_per_position, pred_vs_per_position)):
                if len(obs_fs) > 0:
                    fi_start = np.where(fs >= obs_fs[0])[0][0]
                    fi_end = np.where(fs >= obs_fs[-1])[0][0]
                    obs_v_fx[j, fi_start:fi_end+1] = obs_vs
                if len(pred_fs) > 0:
                    fi_start = np.where(fs >= pred_fs[0])[0][0]
                    fi_end = np.where(fs >= pred_fs[-1])[0][0]
                    pred_v_fx[j, fi_start:fi_end+1] = pred_vs
            path_name = f"{st.session_state.VIZ_folder_path}/M{mode}_vr_pseudo-section.svg"
            display_pseudo_sections(obs_v_fx, pred_v_fx, fs, st.session_state.VIZ_xmids, path_name)

    if st.session_state.VIZ_model == 'Best layered':
        if st.session_state.VIZ_vertical_smoothing == 'No':
            if st.session_state.VIZ_lateral_smoothing == 'No':
                if st.session_state.VIZ_v_xd_best_layered is not None:
                    display_inverted_section(vs_section=st.session_state.VIZ_v_xd_best_layered,
                                            std_section=np.full(st.session_state.VIZ_v_xd_best_layered.shape, np.nan),
                                            positions=st.session_state.VIZ_xmids,
                                            depths=st.session_state.VIZ_depths_best_layered,
                                            path=path_name,
                                            zmin=st.session_state.VIZ_vmin_vmax[0],
                                            zmax=st.session_state.VIZ_vmin_vmax[1],
                                            cmap=st.session_state.VIZ_cmap)
            elif st.session_state.VIZ_lateral_smoothing == 'Yes':
                if st.session_state.VIZ_v_xd_best_layered_lateralsmooth is not None:
                    display_inverted_section(vs_section=st.session_state.VIZ_v_xd_best_layered_lateralsmooth,
                                            std_section=np.full(st.session_state.VIZ_v_xd_best_layered_lateralsmooth.shape, np.nan),
                                            positions=st.session_state.VIZ_xmids,
                                            depths=st.session_state.VIZ_depths_best_layered,
                                            path=path_name,
                                            zmin=st.session_state.VIZ_vmin_vmax[0],
                                            zmax=st.session_state.VIZ_vmin_vmax[1],
                                            cmap=st.session_state.VIZ_cmap)
        elif st.session_state.VIZ_vertical_smoothing == 'Yes':
            if st.session_state.VIZ_lateral_smoothing == 'No':
                if st.session_state.VIZ_v_xd_verticalsmooth_best_layered is not None:
                    display_inverted_section(vs_section=st.session_state.VIZ_v_xd_verticalsmooth_best_layered,
                                            std_section=np.full(st.session_state.VIZ_v_xd_verticalsmooth_best_layered.shape, np.nan),
                                            positions=st.session_state.VIZ_xmids,
                                            depths=st.session_state.VIZ_depths_verticalsmooth_best_layered,
                                            path=path_name,
                                            zmin=st.session_state.VIZ_vmin_vmax[0],
                                            zmax=st.session_state.VIZ_vmin_vmax[1],
                                            cmap=st.session_state.VIZ_cmap)
            elif st.session_state.VIZ_lateral_smoothing == 'Yes':
                if st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth is not None:
                    display_inverted_section(vs_section=st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth,
                                            std_section=np.full(st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth.shape, np.nan),
                                            positions=st.session_state.VIZ_xmids,
                                            depths=st.session_state.VIZ_depths_verticalsmooth_best_layered,
                                            path=path_name,
                                            zmin=st.session_state.VIZ_vmin_vmax[0],
                                            zmax=st.session_state.VIZ_vmin_vmax[1],
                                            cmap=st.session_state.VIZ_cmap)
        for mode in distinct_modes.keys():
            # Observed
            obs_fs_per_position = []
            obs_vs_per_position = []
            for j, folder in enumerate(st.session_state.VIZ_folders):
                try :
                    pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/pick/{folder}_obs_M{mode}.pvc")
                    if len(pvc.shape) == 1:
                            pvc = pvc.reshape(1,-1)
                    pvc = np.round(pvc, 2)
                    fs = pvc[:,0]
                    vs = pvc[:,1]
                    fs, vs = resamp_frequency(fs, vs)
                    obs_fs_per_position.append(fs)
                    obs_vs_per_position.append(vs)
                except:
                    obs_fs_per_position.append([])
                    obs_vs_per_position.append([])
            # Predicted
            pred_fs_per_position = []
            pred_vs_per_position = []
            for j, folder in enumerate(st.session_state.VIZ_folders):
                try :
                    if st.session_state.VIZ_vertical_smoothing == 'No':
                        pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_best_layered_M{mode}.pvc")
                    elif st.session_state.VIZ_vertical_smoothing == 'Yes':
                        pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_smooth_best_layered_M{mode}.pvc")
                    if len(pvc.shape) == 1:
                            pvc = pvc.reshape(1,-1)
                    pvc = np.round(pvc, 2)
                    fs = pvc[:,0]
                    vs = pvc[:,1]
                    fs, vs = resamp_frequency(fs, vs)
                    pred_fs_per_position.append(fs)
                    pred_vs_per_position.append(vs)
                except:
                    pred_fs_per_position.append([])
                    pred_vs_per_position.append([])
            # General
            obs_f_min = np.min([np.min(fs) for fs in obs_fs_per_position if len(fs) > 0])
            obs_f_max = np.max([np.max(fs) for fs in obs_fs_per_position if len(fs) > 0])
            if not all(len(fs) == 0 for fs in pred_fs_per_position):
                pred_f_min = np.min([np.min(fs) for fs in pred_fs_per_position if len(fs) > 0])
                pred_f_max = np.max([np.max(fs) for fs in pred_fs_per_position if len(fs) > 0])
            else:
                pred_f_min = obs_f_max
                pred_f_max = obs_f_max
            f_min = np.min([obs_f_min, pred_f_min])
            f_max = np.max([obs_f_max, pred_f_max])
            fs = arange(f_min, f_max, 1)
            pred_v_fx = np.full((len(st.session_state.VIZ_xmids), len(fs)), np.nan)
            obs_v_fx = np.full((len(st.session_state.VIZ_xmids), len(fs)), np.nan)
            for j, (obs_fs, obs_vs, pred_fs, pred_vs) in enumerate(zip(obs_fs_per_position, obs_vs_per_position, pred_fs_per_position, pred_vs_per_position)):
                if len(obs_fs) > 0:
                    fi_start = np.where(fs >= obs_fs[0])[0][0]
                    fi_end = np.where(fs >= obs_fs[-1])[0][0]
                    obs_v_fx[j, fi_start:fi_end+1] = obs_vs
                if len(pred_fs) > 0:
                    fi_start = np.where(fs >= pred_fs[0])[0][0]
                    fi_end = np.where(fs >= pred_fs[-1])[0][0]
                    pred_v_fx[j, fi_start:fi_end+1] = pred_vs
            path_name = f"{st.session_state.VIZ_folder_path}/M{mode}_vr_pseudo-section.svg"
            display_pseudo_sections(obs_v_fx, pred_v_fx, fs, st.session_state.VIZ_xmids, path_name)

    
    if st.session_state.VIZ_model == 'Median ensemble':
        if st.session_state.VIZ_lateral_smoothing == 'No':
            if st.session_state.VIZ_v_xd_median_ensemble is not None:
                display_inverted_section(vs_section=st.session_state.VIZ_v_xd_median_ensemble,
                                        std_section=st.session_state.VIZ_std_xd_median_ensemble,
                                        positions=st.session_state.VIZ_xmids,
                                        depths=st.session_state.VIZ_depths_median_ensemble,
                                        path=path_name,
                                        zmin=st.session_state.VIZ_vmin_vmax[0],
                                        zmax=st.session_state.VIZ_vmin_vmax[1],
                                        cmap=st.session_state.VIZ_cmap)
        elif st.session_state.VIZ_lateral_smoothing == 'Yes':
            if st.session_state.VIZ_v_xd_median_ensemble_lateralsmooth is not None:
                display_inverted_section(vs_section=st.session_state.VIZ_v_xd_median_ensemble_lateralsmooth,
                                        std_section=st.session_state.VIZ_std_xd_median_ensemble_lateralsmooth,
                                        positions=st.session_state.VIZ_xmids,
                                        depths=st.session_state.VIZ_depths_median_ensemble,
                                        path=path_name,
                                        zmin=st.session_state.VIZ_vmin_vmax[0],
                                        zmax=st.session_state.VIZ_vmin_vmax[1],
                                        cmap=st.session_state.VIZ_cmap)
        for mode in distinct_modes.keys():
            # Observed
            obs_fs_per_position = []
            obs_vs_per_position = []
            for j, folder in enumerate(st.session_state.VIZ_folders):
                try :
                    pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/pick/{folder}_obs_M{mode}.pvc")
                    if len(pvc.shape) == 1:
                            pvc = pvc.reshape(1,-1)
                    pvc = np.round(pvc, 2)
                    fs = pvc[:,0]
                    vs = pvc[:,1]
                    fs, vs = resamp_frequency(fs, vs)
                    obs_fs_per_position.append(fs)
                    obs_vs_per_position.append(vs)
                except:
                    obs_fs_per_position.append([])
                    obs_vs_per_position.append([])
            # Predicted
            pred_fs_per_position = []
            pred_vs_per_position = []
            for j, folder in enumerate(st.session_state.VIZ_folders):
                try :
                    pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_ensemble_M{mode}.pvc")
                    if len(pvc.shape) == 1:
                            pvc = pvc.reshape(1,-1)
                    pvc = np.round(pvc, 2)
                    fs = pvc[:,0]
                    vs = pvc[:,1]
                    fs, vs = resamp_frequency(fs, vs)
                    pred_fs_per_position.append(fs)
                    pred_vs_per_position.append(vs)
                except:
                    pred_fs_per_position.append([])
                    pred_vs_per_position.append([])
            # General
            obs_f_min = np.min([np.min(fs) for fs in obs_fs_per_position if len(fs) > 0])
            obs_f_max = np.max([np.max(fs) for fs in obs_fs_per_position if len(fs) > 0])
            if not all(len(fs) == 0 for fs in pred_fs_per_position):
                pred_f_min = np.min([np.min(fs) for fs in pred_fs_per_position if len(fs) > 0])
                pred_f_max = np.max([np.max(fs) for fs in pred_fs_per_position if len(fs) > 0])
            else:
                pred_f_min = obs_f_max
                pred_f_max = obs_f_max
            f_min = np.min([obs_f_min, pred_f_min])
            f_max = np.max([obs_f_max, pred_f_max])
            fs = arange(f_min, f_max, 1)
            pred_v_fx = np.full((len(st.session_state.VIZ_xmids), len(fs)), np.nan)
            obs_v_fx = np.full((len(st.session_state.VIZ_xmids), len(fs)), np.nan)
            for j, (obs_fs, obs_vs, pred_fs, pred_vs) in enumerate(zip(obs_fs_per_position, obs_vs_per_position, pred_fs_per_position, pred_vs_per_position)):
                if len(obs_fs) > 0:
                    fi_start = np.where(fs >= obs_fs[0])[0][0]
                    fi_end = np.where(fs >= obs_fs[-1])[0][0]
                    obs_v_fx[j, fi_start:fi_end+1] = obs_vs
                if len(pred_fs) > 0:
                    fi_start = np.where(fs >= pred_fs[0])[0][0]
                    fi_end = np.where(fs >= pred_fs[-1])[0][0]
                    pred_v_fx[j, fi_start:fi_end+1] = pred_vs
            path_name = f"{st.session_state.VIZ_folder_path}/M{mode}_vr_pseudo-section.svg"
            display_pseudo_sections(obs_v_fx, pred_v_fx, fs, st.session_state.VIZ_xmids, path_name)
            
    
                        
def handle_select_folder():
    mode_tmp = st.session_state.VIZ_mode
    selected_folder_tmp = st.session_state.VIZ_selected_folder
    clear_session()
    initialize_session()
    st.session_state.VIZ_mode = mode_tmp
    st.session_state.VIZ_selected_folder = selected_folder_tmp
    
    if st.session_state.VIZ_selected_folder is not None:
        if st.session_state.VIZ_mode == 'Signal':
            st.session_state.VIZ_folder_path = f"{input_dir}/{st.session_state.VIZ_selected_folder}/"
        if st.session_state.VIZ_mode in ['Dispersion', 'Inversion']:
            st.session_state.VIZ_folder_path = f"{output_dir}/{st.session_state.VIZ_selected_folder}/"
        
def handle_select_mode():
    mode_tmp = st.session_state.VIZ_mode
    clear_session()
    initialize_session()
    st.session_state.VIZ_mode = mode_tmp
    if 'VIZ_selected_folder' in st.session_state:
        st.session_state.VIZ_selected_folder = None
    if 'VIZ_folder_path' in st.session_state:
        st.session_state.VIZ_folder_path = None
### -------------------------------------------------------------------------------------------------------------------------------------------------



### START INTERFACE ---------------------------------------------------------------------------------------------------------------------------------
initialize_session()

st.set_page_config(
    layout="centered",
    page_title="Visualization",
    page_icon="🧐",
    initial_sidebar_state="expanded"
)

st.title("🧐 Visualization")
st.write("🛈 Visualization of raw seismic records, dispersion images and inversion results.")

st.divider() # --------------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Data selection")

st.text('')
st.text('')

# Option
st.selectbox(options=['Signal', 'Dispersion', 'Inversion'], key='VIZ_mode', label='**Visualization mode**', index=None, placeholder='Select', on_change=handle_select_mode)
if st.session_state.VIZ_mode is None:
    if st.session_state.VIZ_selected_folder is not None:
        st.session_state.VIZ_selected_folder = None
    if st.session_state.VIZ_folder_path is not None:
        st.session_state.VIZ_folder_path = None
    st.info("👆 Select the type of data to visualize.")
    st.stop()

if st.session_state.VIZ_mode == 'Signal':
    # Folder selection
    files_depth_1 = glob.glob(f"{input_dir}/*")
    input_folders = filter(lambda f: os.path.isdir(f), files_depth_1)
    input_folders = [os.path.relpath(folder, input_dir) for folder in input_folders]
    input_folders = sorted(input_folders)
    if input_folders:
        st.selectbox("**Data folder**", input_folders, key='VIZ_selected_folder', on_change=handle_select_folder, index=None, placeholder='Select')
    else:
        st.error("❌ No input data folders found.")
        st.stop()

    if st.session_state.VIZ_folder_path is None:
        st.info("👆 Select a folder containing the raw seismic files.")
        st.stop()

    files = [file for file in os.listdir(st.session_state.VIZ_folder_path)]

    if len(files) < 1:
        st.error("❌ Selected input data folder empty.")
        st.stop()

    st.success(f"📁 Selected folder: **{st.session_state.VIZ_folder_path}**")
            
    # List all files in the folder
    files = [file for file in os.listdir(st.session_state.VIZ_folder_path)]
    files = sorted(files)
    folders = files

    # Read all seismic records
    streams = []
    for file in folders:
        stream = read(st.session_state.VIZ_folder_path + file)
        streams.append(stream)
    st.session_state.VIZ_streams = streams

    # Compute the duration of each seismic record
    st.session_state.VIZ_durations = [stream[0].stats.endtime - stream[0].stats.starttime for stream in st.session_state.VIZ_streams]

    st.divider()
    st.subheader("**Seismic records**")
    normalize = st.toggle("Normalize by trace", value=False)
    
    if normalize:
        norm = 'trace'
    else:
        norm = 'global'
    
    # Plot the records on a subplot
    for i, stream in enumerate(st.session_state.VIZ_streams):
        TX = stream_to_array(stream, len(stream), len(stream[0].data))
        delta = stream[0].stats.delta
        total_points = TX.shape[0]
        if total_points > 1000:
            factor = total_points // 1000
            TX = TX[::factor, :]
        delta = delta * factor
        x_sensors = arange(1, len(stream), 1)
        fig = plot_wiggle(TX.T, x_sensors, delta, norm=norm)
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.markdown(f"**Record {i+1}**")
        st.plotly_chart(fig, key=f"{i}")
                
elif st.session_state.VIZ_mode == 'Dispersion':
    # Folder selection
    files_depth_3 = glob.glob(f"{output_dir}/*/*/*")
    input_folders = filter(lambda f: os.path.isdir(f), files_depth_3)
    input_folders = [os.path.relpath(folder, output_dir) for folder in input_folders]
    input_folders = sorted(input_folders)
    if input_folders:
        st.selectbox("**Data folder**", input_folders, key='VIZ_selected_folder', on_change=handle_select_folder, index=None, placeholder='Select')
    else:
        st.error("❌ No input data folders found.")
        st.stop()

    if st.session_state.VIZ_folder_path is None:
        st.info("👆 Select a folder containing the dispersion data.")
        st.stop()

    folders = [xmid for xmid in os.listdir(st.session_state.VIZ_folder_path) if xmid[0:4] == 'xmid']

    if len(folders) < 1:
        st.error("❌ Selected output data folder empty.")
        st.stop()

    st.success(f"📁 Selected folder: **{st.session_state.VIZ_folder_path}**")
    
    xmids = [float(folder[4:]) for folder in folders]
    st.session_state.VIZ_xmids, st.session_state.VIZ_folders = zip(*sorted(zip(xmids, folders)))

    # MASW parameters from json file
    with open(f"{st.session_state.VIZ_folder_path}/computing_params.json", "r") as f:
        computing_param = json.load(f)
        st.session_state.VIZ_Nx = computing_param["MASW_length"]
        st.session_state.VIZ_dx = computing_param["positions"][1] - computing_param["positions"][0]    
    
    all_dispersion = []
    all_fs = []
    all_vs = []
    availables = []
    for folder in st.session_state.VIZ_folders:
        list_dir = os.listdir(f'{st.session_state.VIZ_folder_path}/{folder}/comp/')
        if f'{folder}_dispersion.csv' in list_dir and f'{folder}_fs.csv' in list_dir and f'{folder}_vs.csv' in list_dir:
            all_dispersion.append(np.loadtxt(f'{st.session_state.VIZ_folder_path}/{folder}/comp/{folder}_dispersion.csv', delimiter=','))
            all_fs.append(np.loadtxt(f'{st.session_state.VIZ_folder_path}/{folder}/comp/{folder}_fs.csv', delimiter=','))
            all_vs.append(np.loadtxt(f'{st.session_state.VIZ_folder_path}/{folder}/comp/{folder}_vs.csv', delimiter=','))
            availables.append(True)
        else:
            availables.append(False)
            all_dispersion.append(None)
            all_fs.append(None)
            all_vs.append(None)
    
    st.divider()
    st.subheader("**Dispersion images**")
    for available, xmid, dispersion, fs, vs in zip(availables, st.session_state.VIZ_xmids, all_dispersion, all_fs, all_vs):
        if available:
            fig = plot_disp(dispersion, fs, vs, norm='Frequencies', dx=st.session_state.VIZ_dx, Nx=st.session_state.VIZ_Nx)
            
            if os.path.exists(f"{st.session_state.VIZ_folder_path}/xmid{xmid}/pick/"):
                picked_modes = [fname for fname in os.listdir(f'{st.session_state.VIZ_folder_path}/xmid{xmid}/pick/') if fname.endswith(".pvc") and fname.startswith(f"xmid{xmid}_obs_M")]
            else:
                picked_modes = []     
            for picked_mode in picked_modes:
                pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/xmid{xmid}/pick/{picked_mode}")
                if len(pvc.shape) == 1:
                        pvc = pvc.reshape(1,-1)
                fig.add_trace(go.Scatter(x=pvc[:,0],
                                    y=pvc[:,1],
                                    mode='lines',
                                    name=f'Mode {int(picked_mode.split("_M")[1].split(".")[0])}',
                                    line=dict(color='white', width=2),
                                    error_y=dict(type='data', array=pvc[:,2], visible=True, color='white', width=2, thickness=0.75),
                                    showlegend=False
                                    ))
            fig.update_layout(xaxis_range=[min(fs), max(fs)])
            fig.update_layout(yaxis={'range': [min(vs), max(vs)], 'autorange': False})
            fig.update_layout(height=400)
            st.text('')
            st.text('')
            st.text('')
            st.text('')
            st.markdown(f"**Position: {xmid} m**")
            st.plotly_chart(fig, key=f"{xmid}")
        else:
            st.text('')
            st.text('')
            st.text('')
            st.text('')
            st.markdown(f"**Position: {xmid} m**")
            st.error(f"📁 Dispersion data missing.")
    
    # Read picked modes xmid folder
    nb_picked_modes_by_position = []
    picked_modes_by_position = []
    for folder, pos in zip(st.session_state.VIZ_folders, st.session_state.VIZ_xmids):
        if os.path.exists(f"{st.session_state.VIZ_folder_path}/{folder}/pick/"):
            picked_modes = [int(fname.split("_M")[1].split(".")[0]) for fname in os.listdir(f"{st.session_state.VIZ_folder_path}/{folder}/pick") if fname.endswith(".pvc") and fname.startswith(f"xmid{pos}_obs_M")]
            picked_modes = sorted(picked_modes)
        else:
            picked_modes = []
        picked_modes_by_position.append(picked_modes)
        nb_picked_modes_by_position.append(len(picked_modes))

    # Count distinct modes
    distinct_modes = {}
    for picked_modes in picked_modes_by_position:
        for picked_mode in picked_modes:
            if picked_mode not in distinct_modes.keys():
                distinct_modes[picked_mode] = 1
            else:
                distinct_modes[picked_mode] += 1
    distinct_modes = dict(sorted(distinct_modes.items()))
    
    if len(distinct_modes) > 0:
        st.divider()

        st.subheader("**Pseudo-section**")
        on = st.toggle("OFF: Frequencies | ON: Wavelengths", value=False)
    
    for mode, count in distinct_modes.items():
        if on:
            ws_per_position = []
            vs_per_position = []
            for j, folder in enumerate(st.session_state.VIZ_folders):
                try :
                    pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/pick/{folder}_obs_M{mode}.pvc")
                    if len(pvc.shape) == 1:
                        pvc = pvc.reshape(1,-1)
                    pvc = np.round(pvc, 2)
                    fs = pvc[:,0]
                    vs = pvc[:,1]
                    if len(fs) > 1 and len(vs) > 1:
                        ws, vs = resamp_wavelength(fs, vs)
                    else:
                        ws = np.round(vs/fs)
                    ws_per_position.append(ws)
                    vs_per_position.append(vs)
                except:
                    ws_per_position.append([])
                    vs_per_position.append([])
            min_ws = np.min([np.min(ws) for ws in ws_per_position if len(ws) > 0])
            max_ws = np.max([np.max(ws) for ws in ws_per_position if len(ws) > 0])
            obs_ws = arange(min_ws, max_ws, 1)
            obs_v_wx = np.full((len(st.session_state.VIZ_xmids), len(obs_ws)), np.nan)
            for j, (ws, vs) in enumerate(zip(ws_per_position, vs_per_position)):
                if len(ws) > 0:
                    fi_start = np.where(obs_ws >= ws[0])[0][0]
                    fi_end = np.where(obs_ws >= ws[-1])[0][0]
                    obs_v_wx[j, fi_start:fi_end+1] = vs
            fig = plot_pseudo_section(obs_v_wx, obs_ws, st.session_state.VIZ_xmids, wavelength=True)
            st.plotly_chart(fig)
        else:
            fs_per_position = []
            vs_per_position = []
            for j, folder in enumerate(st.session_state.VIZ_folders):
                try :
                    pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/pick/{folder}_obs_M{mode}.pvc")
                    if len(pvc.shape) == 1:
                        pvc = pvc.reshape(1,-1)
                    pvc = np.round(pvc, 2)
                    fs = pvc[:,0]
                    vs = pvc[:,1]
                    if len(fs) > 1 and len(vs) > 1:
                        fs, vs = resamp_frequency(fs, vs)
                    else:
                        fs = np.round(fs)
                    fs_per_position.append(fs)
                    vs_per_position.append(vs)
                except:
                    fs_per_position.append([])
                    vs_per_position.append([])
            min_fs = np.min([np.min(fs) for fs in fs_per_position if len(fs) > 0])
            max_fs = np.max([np.max(fs) for fs in fs_per_position if len(fs) > 0])
            obs_fs = arange(min_fs, max_fs, 1)
            obs_v_fx = np.full((len(st.session_state.VIZ_xmids), len(obs_fs)), np.nan)
            for j, (fs, vs) in enumerate(zip(fs_per_position, vs_per_position)):
                if len(fs) > 0:
                    fi_start = np.where(obs_fs >= fs[0])[0][0]
                    fi_end = np.where(obs_fs >= fs[-1])[0][0]
                    obs_v_fx[j, fi_start:fi_end+1] = vs
            fig = plot_pseudo_section(obs_v_fx, obs_fs, st.session_state.VIZ_xmids)
            st.plotly_chart(fig)                     
                        
elif st.session_state.VIZ_mode == 'Inversion':
    # Folder selection
    files_depth_3 = glob.glob(f"{output_dir}/*/*/*")
    input_folders = filter(lambda f: os.path.isdir(f), files_depth_3)
    input_folders = [os.path.relpath(folder, output_dir) for folder in input_folders]
    input_folders = sorted(input_folders)
    if input_folders:
        st.selectbox("**Data folder**", input_folders, key='VIZ_selected_folder', on_change=handle_select_folder, index=None, placeholder='Select')
    else:
        st.error("❌ No input data folders found.")
        st.stop()

    if st.session_state.VIZ_folder_path is None:
        st.info("👆 Select a folder containing the inversion results.")
        st.stop()

    folders = [xmid for xmid in os.listdir(st.session_state.VIZ_folder_path) if xmid[0:4] == 'xmid']

    if len(folders) < 1:
        st.error("❌ Selected output data folder empty.")
        st.stop()

    st.success(f"📁 Selected folder: **{st.session_state.VIZ_folder_path}**")

    xmids = [float(folder[4:]) for folder in folders]
    st.session_state.VIZ_xmids, st.session_state.VIZ_folders = zip(*sorted(zip(xmids, folders)))

    # MASW parameters from json file
    with open(f"{st.session_state.VIZ_folder_path}/computing_params.json", "r") as f:
        computing_param = json.load(f)
        st.session_state.VIZ_Nx = computing_param["MASW_length"]
        st.session_state.VIZ_dx = computing_param["positions"][1] - computing_param["positions"][0]                    
                        
    st.divider()
    
    st.header("🚨 Inversion results")
    
    st.text('')
    st.text('')
    
    if st.session_state.VIZ_v_xd_median_ensemble is None and st.session_state.VIZ_v_xd_median_layered is None and st.session_state.VIZ_v_xd_best_layered is None:
        load_sections()
    
    if st.session_state.VIZ_v_xd_median_ensemble is None and st.session_state.VIZ_v_xd_median_layered is None and st.session_state.VIZ_v_xd_best_layered is None:
        st.text('')
        st.text('')
        st.error(f"📁 Inversion data missing.")
        st.divider()
        st.stop()
        
    st.subheader("**Shear wave velocity profile**")
    
    st.write('')
    st.write('')
    st.markdown(f"🔧 **Display settings**")
    with st.container(border=True):
        columns = st.columns(4)
        with columns[0]:
            st.radio("**Colormap**", ['Terrain', 'Viridis', 'Jet', 'Gray'],
                key='VIZ_cmap')
        with columns[1]:
            st.radio("**Model**", ['Median layered', 'Best layered', 'Median ensemble'],
                key='VIZ_model')
        with columns[2]:
            st.radio("**Vertical smoothing**", ['Yes', 'No'],
                key='VIZ_vertical_smoothing')
        with columns[3]:
            st.radio("**Lateral smoothing**", ['Yes', 'No'],
                key='VIZ_lateral_smoothing')
            
    st.text('')
    st.text('')
        










    if st.session_state.VIZ_model == 'Median ensemble':
        
        if st.session_state.VIZ_v_xd_median_ensemble is not None:
            
            vs_min = st.session_state.VIZ_vs_min_median_ensemble
            vs_max = st.session_state.VIZ_vs_max_median_ensemble
            
            with st.container(border=True):
                st.select_slider(
                    "**$v_{S}$ range**",
                    options=arange(np.floor(vs_min), np.ceil(vs_max), 1),
                    value=(np.floor(vs_min), np.ceil(vs_max)),
                    key='VIZ_vmin_vmax'
                )

            if st.session_state.VIZ_vmin_vmax is None:
                vmin = vs_min
                vmax = vs_max
            else:
                vmin = st.session_state.VIZ_vmin_vmax[0]
                vmax = st.session_state.VIZ_vmin_vmax[1]
            if st.session_state.VIZ_lateral_smoothing == 'Yes':
                fig = plot_inverted_section(st.session_state.VIZ_v_xd_median_ensemble_lateralsmooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_median_ensemble, zmin=vmin, zmax=vmax)
            elif st.session_state.VIZ_lateral_smoothing == 'No':
                fig = plot_inverted_section(st.session_state.VIZ_v_xd_median_ensemble, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_median_ensemble, zmin=vmin, zmax=vmax)
                
            fig.update_layout(height=400)
            if st.session_state.VIZ_cmap is not None:
                if st.session_state.VIZ_cmap == 'Terrain':
                        cmap = plt.get_cmap('terrain')
                        colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]
                        fig.update_coloraxes(colorscale=colorscale)
                else:
                    fig.update_coloraxes(colorscale=st.session_state.VIZ_cmap)
            st.plotly_chart(fig)
            
            if st.session_state.VIZ_lateral_smoothing == 'Yes':
                fig = plot_std_section(st.session_state.VIZ_std_xd_median_ensemble_lateralsmooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_median_ensemble)
            elif st.session_state.VIZ_lateral_smoothing == 'No':
                fig = plot_std_section(st.session_state.VIZ_std_xd_median_ensemble, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_median_ensemble)
            fig.update_layout(height=400)
            st.plotly_chart(fig)
            
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.button("Save images", type="primary", use_container_width=True, on_click=save_images)
            st.markdown(r"🛈 *Will save the above $v_{S}$ and standard deviation sections, and the corresponding $v_{R}$ pseudo-sections.*")
                
        else:
            vs_min = 0
            vs_max = 1
            st.text('')
            st.text('')
            st.error(f"📁 Inversion data missing for median ensemble model.")











            
    if st.session_state.VIZ_model == 'Median layered':
        if st.session_state.VIZ_vertical_smoothing == 'No':
        
            if st.session_state.VIZ_v_xd_median_layered is not None:
                
                vs_min = st.session_state.VIZ_vs_min_median_layered
                vs_max = st.session_state.VIZ_vs_max_median_layered
                
                with st.container(border=True):
                    st.select_slider(
                        "**$v_{S}$ range**",
                        options=arange(np.floor(vs_min), np.ceil(vs_max), 1),
                        value=(np.floor(vs_min), np.ceil(vs_max)),
                        key='VIZ_vmin_vmax'
                    )

                if st.session_state.VIZ_vmin_vmax is None:
                    vmin = vs_min
                    vmax = vs_max
                else:
                    vmin = st.session_state.VIZ_vmin_vmax[0]
                    vmax = st.session_state.VIZ_vmin_vmax[1]
                if st.session_state.VIZ_lateral_smoothing == 'Yes':
                    fig = plot_inverted_section(st.session_state.VIZ_v_xd_median_layered_lateralsmooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_median_layered, zmin=vmin, zmax=vmax)
                elif st.session_state.VIZ_lateral_smoothing == 'No':
                    fig = plot_inverted_section(st.session_state.VIZ_v_xd_median_layered, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_median_layered, zmin=vmin, zmax=vmax)
                    
                fig.update_layout(height=400)
                if st.session_state.VIZ_cmap is not None:
                    if st.session_state.VIZ_cmap == 'Terrain':
                            cmap = plt.get_cmap('terrain')
                            colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]
                            fig.update_coloraxes(colorscale=colorscale)
                    else:
                        fig.update_coloraxes(colorscale=st.session_state.VIZ_cmap)
                st.plotly_chart(fig)
                
                if st.session_state.VIZ_lateral_smoothing == 'Yes':
                    fig = plot_std_section(st.session_state.VIZ_std_xd_median_layered_lateralsmooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_median_layered)
                elif st.session_state.VIZ_lateral_smoothing == 'No':
                    fig = plot_std_section(st.session_state.VIZ_std_xd_median_layered, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_median_layered)
                fig.update_layout(height=400)
                st.plotly_chart(fig)
                
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.button("Save images", type="primary", use_container_width=True, on_click=save_images)
                st.markdown(r"🛈 *Will save the above $v_{S}$ and standard deviation sections, and the corresponding $v_{R}$ pseudo-sections.*")
                    
            else:
                vs_min = 0
                vs_max = 1
                st.text('')
                st.text('')
                st.error(f"📁 Inversion data missing for median layered model.")
        
        elif st.session_state.VIZ_vertical_smoothing == 'Yes':
        
            if st.session_state.VIZ_v_xd_verticalsmooth_median_layered is not None:
                
                vs_min = st.session_state.VIZ_vs_min_verticalsmooth_median_layered
                vs_max = st.session_state.VIZ_vs_max_verticalsmooth_median_layered
                
                with st.container(border=True):
                    st.select_slider(
                        "**$v_{S}$ range**",
                        options=arange(np.floor(vs_min), np.ceil(vs_max), 1),
                        value=(np.floor(vs_min), np.ceil(vs_max)),
                        key='VIZ_vmin_vmax'
                    )

                if st.session_state.VIZ_vmin_vmax is None:
                    vmin = vs_min
                    vmax = vs_max
                else:
                    vmin = st.session_state.VIZ_vmin_vmax[0]
                    vmax = st.session_state.VIZ_vmin_vmax[1]
                if st.session_state.VIZ_lateral_smoothing == 'Yes':
                    fig = plot_inverted_section(st.session_state.VIZ_v_xd_verticalsmooth_median_layered_lateralsmooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_verticalsmooth_median_layered, zmin=vmin, zmax=vmax)
                elif st.session_state.VIZ_lateral_smoothing == 'No':
                    fig = plot_inverted_section(st.session_state.VIZ_v_xd_verticalsmooth_median_layered, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_verticalsmooth_median_layered, zmin=vmin, zmax=vmax)
                    
                fig.update_layout(height=400)
                if st.session_state.VIZ_cmap is not None:
                    if st.session_state.VIZ_cmap == 'Terrain':
                            cmap = plt.get_cmap('terrain')
                            colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]
                            fig.update_coloraxes(colorscale=colorscale)
                    else:
                        fig.update_coloraxes(colorscale=st.session_state.VIZ_cmap)
                st.plotly_chart(fig)
                
                if st.session_state.VIZ_lateral_smoothing == 'Yes':
                    fig = plot_std_section(st.session_state.VIZ_std_xd_verticalsmooth_median_layered_lateralsmooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_verticalsmooth_median_layered)
                elif st.session_state.VIZ_lateral_smoothing == 'No':
                    fig = plot_std_section(st.session_state.VIZ_std_xd_verticalsmooth_median_layered, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_verticalsmooth_median_layered)
                fig.update_layout(height=400)
                st.plotly_chart(fig)
                
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.button("Save images", type="primary", use_container_width=True, on_click=save_images)
                st.markdown(r"🛈 *Will save the above $v_{S}$ and standard deviation sections, and the corresponding $v_{R}$ pseudo-sections.*")
                    
            else:
                vs_min = 0
                vs_max = 1
                st.text('')
                st.text('')
                st.error(f"📁 Inversion data missing for median layered model.")








    if st.session_state.VIZ_model == 'Best layered':
        if st.session_state.VIZ_vertical_smoothing == 'No':
        
            if st.session_state.VIZ_v_xd_best_layered is not None:
                
                vs_min = st.session_state.VIZ_vs_min_best_layered
                vs_max = st.session_state.VIZ_vs_max_best_layered
                
                with st.container(border=True):
                    st.select_slider(
                        "**$v_{S}$ range**",
                        options=arange(np.floor(vs_min), np.ceil(vs_max), 1),
                        value=(np.floor(vs_min), np.ceil(vs_max)),
                        key='VIZ_vmin_vmax'
                    )

                if st.session_state.VIZ_vmin_vmax is None:
                    vmin = vs_min
                    vmax = vs_max
                else:
                    vmin = st.session_state.VIZ_vmin_vmax[0]
                    vmax = st.session_state.VIZ_vmin_vmax[1]
                if st.session_state.VIZ_lateral_smoothing == 'Yes':
                    fig = plot_inverted_section(st.session_state.VIZ_v_xd_best_layered_lateralsmooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_best_layered, zmin=vmin, zmax=vmax)
                elif st.session_state.VIZ_lateral_smoothing == 'No':
                    fig = plot_inverted_section(st.session_state.VIZ_v_xd_best_layered, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_best_layered, zmin=vmin, zmax=vmax)
                    
                fig.update_layout(height=400)
                if st.session_state.VIZ_cmap is not None:
                    if st.session_state.VIZ_cmap == 'Terrain':
                            cmap = plt.get_cmap('terrain')
                            colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]
                            fig.update_coloraxes(colorscale=colorscale)
                    else:
                        fig.update_coloraxes(colorscale=st.session_state.VIZ_cmap)
                st.plotly_chart(fig)
                
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.button("Save images", type="primary", use_container_width=True, on_click=save_images)
                st.markdown(r"🛈 *Will save the above $v_{S}$ and standard deviation sections, and the corresponding $v_{R}$ pseudo-sections.*")
                    
            else:
                vs_min = 0
                vs_max = 1
                st.text('')
                st.text('')
                st.error(f"📁 Inversion data missing for best layered model.")
        
        elif st.session_state.VIZ_vertical_smoothing == 'Yes':
        
            if st.session_state.VIZ_v_xd_verticalsmooth_best_layered is not None:
                
                vs_min = st.session_state.VIZ_vs_min_verticalsmooth_best_layered
                vs_max = st.session_state.VIZ_vs_max_verticalsmooth_best_layered
                
                with st.container(border=True):
                    st.select_slider(
                        "**$v_{S}$ range**",
                        options=arange(np.floor(vs_min), np.ceil(vs_max), 1),
                        value=(np.floor(vs_min), np.ceil(vs_max)),
                        key='VIZ_vmin_vmax'
                    )

                if st.session_state.VIZ_vmin_vmax is None:
                    vmin = vs_min
                    vmax = vs_max
                else:
                    vmin = st.session_state.VIZ_vmin_vmax[0]
                    vmax = st.session_state.VIZ_vmin_vmax[1]
                if st.session_state.VIZ_lateral_smoothing == 'Yes':
                    fig = plot_inverted_section(st.session_state.VIZ_v_xd_verticalsmooth_best_layered_lateralsmooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_verticalsmooth_best_layered, zmin=vmin, zmax=vmax)
                elif st.session_state.VIZ_lateral_smoothing == 'No':
                    fig = plot_inverted_section(st.session_state.VIZ_v_xd_verticalsmooth_best_layered, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_verticalsmooth_best_layered, zmin=vmin, zmax=vmax)
                    
                fig.update_layout(height=400)
                if st.session_state.VIZ_cmap is not None:
                    if st.session_state.VIZ_cmap == 'Terrain':
                            cmap = plt.get_cmap('terrain')
                            colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]
                            fig.update_coloraxes(colorscale=colorscale)
                    else:
                        fig.update_coloraxes(colorscale=st.session_state.VIZ_cmap)
                st.plotly_chart(fig)
                
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.button("Save images", type="primary", use_container_width=True, on_click=save_images)
                st.markdown(r"🛈 *Will save the above $v_{S}$ and standard deviation sections, and the corresponding $v_{R}$ pseudo-sections.*")
                    
            else:
                vs_min = 0
                vs_max = 1
                st.text('')
                st.text('')
                st.error(f"📁 Inversion data missing for best layered model.")

                    



    
st.divider() # --------------------------------------------------------------------------------------------------------------------------------------
### END INTERFACE------------------------------------------------------------------------------------------------------------------------------------