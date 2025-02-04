"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 20252025
"""

import os
import glob
import numpy as np
import streamlit as st
from obspy import read
import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import generic_filter

from modules.dispersion import resamp_wavelength, resamp_frequency
from modules.display import plot_pseudo_section, plot_wiggle, plot_disp, plot_inverted_section, plot_std_section, display_inverted_section, display_pseudo_sections
from modules.obspy2numpy import stream_to_array
from modules.misc import arange

from Paths import input_dir, output_dir

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
    if 'VIZ_v_xd_layered_raw' not in st.session_state:
        st.session_state.VIZ_v_xd_layered_raw = None
    if 'VIZ_v_xd_layered_smooth' not in st.session_state:
        st.session_state.VIZ_v_xd_layered_smooth = None
    if 'VIZ_v_xd_ridge_raw' not in st.session_state:
        st.session_state.VIZ_v_xd_ridge_raw = None
    if 'VIZ_v_xd_ridge_smooth' not in st.session_state:
        st.session_state.VIZ_v_xd_ridge_smooth = None
    if 'VIZ_v_xd_smooth_raw' not in st.session_state:
        st.session_state.VIZ_v_xd_smooth_raw = None
    if 'VIZ_v_xd_smooth_smooth' not in st.session_state:
        st.session_state.VIZ_v_xd_smooth_smooth = None
    if 'VIZ_cmap' not in st.session_state:
        st.session_state.VIZ_cmap = 'Terrain'
    if 'VIZ_model' not in st.session_state:
        st.session_state.VIZ_model = 'Smooth'
    if 'VIZ_smoothing' not in st.session_state:
        st.session_state.VIZ_smoothing = 'Smooth'
        
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
    
    all_gm_layered = []
    all_std_layered = []
    for folder in st.session_state.VIZ_folders:
        try:
            gm = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_layered_model.gm", skiprows=1)
            std = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_layered_std.gm", skiprows=1)
        except:
            gm = None
            std = None
        all_gm_layered.append(gm)
        all_std_layered.append(std)
    
    all_gm_ridge = []
    all_std_ridge = []
    for folder in st.session_state.VIZ_folders:
        try:
            gm = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_ridge_model.gm", skiprows=1)
            std = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_ridge_std.gm", skiprows=1)
        except:
            gm = None
            std = None
        all_gm_ridge.append(gm)
        all_std_ridge.append(std)

    all_gm_smooth = []
    all_std_smooth = []
    for folder in st.session_state.VIZ_folders:
        try:
            gm = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_smooth_model.gm", skiprows=1)
            std = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_smooth_std.gm", skiprows=1)
        except:
            gm = None
            std = None
        all_gm_smooth.append(gm)
        all_std_smooth.append(std)        
    
    if not all(gm is None for gm in all_gm_layered):
        max_depth_layered = np.nanmax([np.sum(gm[:,0]) for gm in all_gm_layered if gm is not None])
    else:
        max_depth_layered = np.nan
    if not all(gm is None for gm in all_gm_ridge):
        max_depth_ridge = np.nanmax([np.sum(gm[:,0]) for gm in all_gm_ridge if gm is not None])
    else:
        max_depth_ridge = np.nan
    if not all(gm is None for gm in all_gm_smooth):
        max_depth_smooth = np.nanmax([np.sum(gm[:,0]) for gm in all_gm_smooth if gm is not None])
    else:
        max_depth_smooth = np.nan
    depth_max = np.nanmax([max_depth_layered, max_depth_ridge, max_depth_smooth])
    
    dz = 0.1
    
    if not all(gm is None for gm in all_gm_layered) and not all(gm is None for gm in all_std_layered):
        st.session_state.VIZ_depths_layered = arange(0, depth_max, dz)
        
        st.session_state.VIZ_v_xd_layered_raw = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_layered)), np.nan)
        st.session_state.VIZ_std_xd_layered_raw = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_layered)), np.nan)
                        
        for j, (gm, gm_std) in enumerate(zip(all_gm_layered, all_std_layered)):
            if gm is not None and gm_std is not None:
                col = []
                col_std = []
                for (thick, vp, vs, rho), (thick_std, vp_std, vs_std, rho_std) in zip(gm, gm_std):
                    col += [vs] * int(thick/dz)
                    col_std += [vs_std] * int(thick/dz)
                if len(col) < len(st.session_state.VIZ_depths_layered):
                    col += [gm[-1,2]] * (len(st.session_state.VIZ_depths_layered)-len(col))
                    col_std += [gm_std[-1,2]] * (len(st.session_state.VIZ_depths_layered)-len(col_std))
                st.session_state.VIZ_v_xd_layered_raw[j, :] = col
                st.session_state.VIZ_std_xd_layered_raw[j, :] = col_std
                
        st.session_state.VIZ_vs_min_layered = np.floor(np.nanmin(st.session_state.VIZ_v_xd_layered_raw))
        st.session_state.VIZ_vs_max_layered = np.ceil(np.nanmax(st.session_state.VIZ_v_xd_layered_raw))
        
        st.session_state.VIZ_v_xd_layered_smooth = generic_filter(st.session_state.VIZ_v_xd_layered_raw, mode_filter_median, size=(4,1))
        st.session_state.VIZ_v_xd_layered_smooth = generic_filter(st.session_state.VIZ_v_xd_layered_smooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_v_xd_layered_smooth = generic_filter(st.session_state.VIZ_v_xd_layered_smooth, mode_filter_median, size=(2,1))
        
        st.session_state.VIZ_std_xd_layered_smooth = generic_filter(st.session_state.VIZ_std_xd_layered_raw, mode_filter_median, size=(4,1))
        st.session_state.VIZ_std_xd_layered_smooth = generic_filter(st.session_state.VIZ_std_xd_layered_smooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_std_xd_layered_smooth = generic_filter(st.session_state.VIZ_std_xd_layered_smooth, mode_filter_median, size=(2,1))
    else:
        st.session_state.VIZ_v_xd_layered_raw = None
        st.session_state.VIZ_v_xd_layered_smooth = None
        st.session_state.VIZ_std_xd_layered_raw = None
        st.session_state.VIZ_std_xd_layered_smooth = None
        st.session_state.VIZ_depths_layered = None
        st.session_state.VIZ_vs_min_layered = None
        st.session_state.VIZ_vs_max_layered = None
        

    if not all(gm is None for gm in all_gm_ridge) and not all(gm is None for gm in all_std_ridge):    
        st.session_state.VIZ_depths_ridge = arange(0, depth_max, dz)
        
        st.session_state.VIZ_v_xd_ridge_raw = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_ridge)), np.nan)
        st.session_state.VIZ_std_xd_ridge_raw = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_ridge)), np.nan)
                        
        for j, (gm, gm_std) in enumerate(zip(all_gm_ridge, all_std_ridge)):
            if gm is not None and gm_std is not None:
                col = []
                col_std = []
                for (thick, vp, vs, rho), (thick_std, vp_std, vs_std, rho_std) in zip(gm, gm_std):
                    col += [vs] * int(thick/dz)
                    col_std += [vs_std] * int(thick/dz)
                if len(col) < len(st.session_state.VIZ_depths_ridge):
                    col += [gm[-1,2]] * (len(st.session_state.VIZ_depths_ridge)-len(col))
                    col_std += [gm_std[-1,2]] * (len(st.session_state.VIZ_depths_ridge)-len(col_std))
                st.session_state.VIZ_v_xd_ridge_raw[j, :] = col
                st.session_state.VIZ_std_xd_ridge_raw[j, :] = col_std
        
        st.session_state.VIZ_vs_min_ridge = np.floor(np.nanmin(st.session_state.VIZ_v_xd_ridge_raw))
        st.session_state.VIZ_vs_max_ridge = np.ceil(np.nanmax(st.session_state.VIZ_v_xd_ridge_raw))
        
        st.session_state.VIZ_v_xd_ridge_smooth = generic_filter(st.session_state.VIZ_v_xd_ridge_raw, mode_filter_median, size=(4,1))
        st.session_state.VIZ_v_xd_ridge_smooth = generic_filter(st.session_state.VIZ_v_xd_ridge_smooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_v_xd_ridge_smooth = generic_filter(st.session_state.VIZ_v_xd_ridge_smooth, mode_filter_median, size=(2,1))
        
        st.session_state.VIZ_std_xd_ridge_smooth = generic_filter(st.session_state.VIZ_std_xd_ridge_raw, mode_filter_median, size=(4,1))
        st.session_state.VIZ_std_xd_ridge_smooth = generic_filter(st.session_state.VIZ_std_xd_ridge_smooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_std_xd_ridge_smooth = generic_filter(st.session_state.VIZ_std_xd_ridge_smooth, mode_filter_median, size=(2,1))
    else:
        st.session_state.VIZ_v_xd_ridge_raw = None
        st.session_state.VIZ_v_xd_ridge_smooth = None
        st.session_state.VIZ_std_xd_ridge_raw = None
        st.session_state.VIZ_std_xd_ridge_smooth = None
        st.session_state.VIZ_depths_ridge = None
        st.session_state.VIZ_vs_min_ridge = None
        st.session_state.VIZ_vs_max_ridge = None


    if not all(gm is None for gm in all_gm_smooth) and not all(gm is None for gm in all_std_smooth): 
        st.session_state.VIZ_depths_smooth = arange(0, depth_max, dz)
        
        st.session_state.VIZ_v_xd_smooth_raw = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_smooth)), np.nan)
        st.session_state.VIZ_std_xd_smooth_raw = np.full((len(st.session_state.VIZ_xmids), len(st.session_state.VIZ_depths_smooth)), np.nan)
                        
        for j, (gm, gm_std) in enumerate(zip(all_gm_smooth, all_std_smooth)):
            if gm is not None and gm_std is not None:
                col = []
                col_std = []
                for (thick, vp, vs, rho), (thick_std, vp_std, vs_std, rho_std) in zip(gm, gm_std):
                    col += [vs] * int(thick/dz)
                    col_std += [vs_std] * int(thick/dz)
                if len(col) < len(st.session_state.VIZ_depths_smooth):
                    col += [gm[-1,2]] * (len(st.session_state.VIZ_depths_smooth)-len(col))
                    col_std += [gm_std[-1,2]] * (len(st.session_state.VIZ_depths_smooth)-len(col_std))
                st.session_state.VIZ_v_xd_smooth_raw[j, :] = col
                st.session_state.VIZ_std_xd_smooth_raw[j, :] = col_std
        
        st.session_state.VIZ_vs_min_smooth = np.floor(np.nanmin(st.session_state.VIZ_v_xd_smooth_raw))
        st.session_state.VIZ_vs_max_smooth = np.ceil(np.nanmax(st.session_state.VIZ_v_xd_smooth_raw))
        
        st.session_state.VIZ_v_xd_smooth_smooth = generic_filter(st.session_state.VIZ_v_xd_smooth_raw, mode_filter_median, size=(4,1))
        st.session_state.VIZ_v_xd_smooth_smooth = generic_filter(st.session_state.VIZ_v_xd_smooth_smooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_v_xd_smooth_smooth = generic_filter(st.session_state.VIZ_v_xd_smooth_smooth, mode_filter_median, size=(2,1))
        
        st.session_state.VIZ_std_xd_smooth_smooth = generic_filter(st.session_state.VIZ_std_xd_smooth_raw, mode_filter_median, size=(4,1))
        st.session_state.VIZ_std_xd_smooth_smooth = generic_filter(st.session_state.VIZ_std_xd_smooth_smooth, mode_filter_median, size=(3,1))
        st.session_state.VIZ_std_xd_smooth_smooth = generic_filter(st.session_state.VIZ_std_xd_smooth_smooth, mode_filter_median, size=(2,1))
    else:
        st.session_state.VIZ_v_xd_smooth_raw = None
        st.session_state.VIZ_v_xd_smooth_smooth = None
        st.session_state.VIZ_std_xd_smooth_raw = None
        st.session_state.VIZ_std_xd_smooth_smooth = None
        st.session_state.VIZ_depths_smooth = None
        st.session_state.VIZ_vs_min_smooth = None
        st.session_state.VIZ_vs_max_smooth = None
        
    if 'VIZ_cmap' in st.session_state:
        st.session_state.VIZ_cmap = 'Terrain'
    if 'VIZ_model' in st.session_state:
        st.session_state.VIZ_model = 'Smooth'
    if 'VIZ_smoothing' in st.session_state:
        st.session_state.VIZ_smoothing = 'Smooth'
        
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
    if st.session_state.VIZ_model == 'Layered':
        if st.session_state.VIZ_smoothing == 'Raw':
            if st.session_state.VIZ_v_xd_layered_raw is not None:
                display_inverted_section(vs_section=st.session_state.VIZ_v_xd_layered_raw,
                                        std_section=st.session_state.VIZ_std_xd_layered_raw,
                                        positions=st.session_state.VIZ_xmids,
                                        depths=st.session_state.VIZ_depths_layered,
                                        path=path_name,
                                        zmin=st.session_state.VIZ_vmin_vmax[0],
                                        zmax=st.session_state.VIZ_vmin_vmax[1],
                                        cmap=st.session_state.VIZ_cmap)
        if st.session_state.VIZ_smoothing == 'Smooth':
            if st.session_state.VIZ_v_xd_layered_smooth is not None:
                display_inverted_section(vs_section=st.session_state.VIZ_v_xd_layered_smooth,
                                        std_section=st.session_state.VIZ_std_xd_layered_smooth,
                                        positions=st.session_state.VIZ_xmids,
                                        depths=st.session_state.VIZ_depths_layered,
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
                    pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_layered_M{mode}.pvc")
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
            
    if st.session_state.VIZ_model == 'Ridge':
        if st.session_state.VIZ_smoothing == 'Raw':
            if st.session_state.VIZ_v_xd_ridge_raw is not None:
                display_inverted_section(vs_section=st.session_state.VIZ_v_xd_ridge_raw,
                                        std_section=st.session_state.VIZ_std_xd_ridge_raw,
                                        positions=st.session_state.VIZ_xmids,
                                        depths=st.session_state.VIZ_depths_ridge,
                                        path=path_name,
                                        zmin=st.session_state.VIZ_vmin_vmax[0],
                                        zmax=st.session_state.VIZ_vmin_vmax[1],
                                        cmap=st.session_state.VIZ_cmap)
        if st.session_state.VIZ_smoothing == 'Smooth':
            if st.session_state.VIZ_v_xd_ridge_smooth is not None:
                display_inverted_section(vs_section=st.session_state.VIZ_v_xd_ridge_smooth,
                                        std_section=st.session_state.VIZ_std_xd_ridge_smooth,
                                        positions=st.session_state.VIZ_xmids,
                                        depths=st.session_state.VIZ_depths_ridge,
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
                    pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_ridge_M{mode}.pvc")
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
            
    if st.session_state.VIZ_model == 'Smooth':
        if st.session_state.VIZ_smoothing == 'Raw':
            if st.session_state.VIZ_v_xd_smooth_raw is not None:
                display_inverted_section(vs_section=st.session_state.VIZ_v_xd_smooth_raw,
                                        std_section=st.session_state.VIZ_std_xd_smooth_raw,
                                        positions=st.session_state.VIZ_xmids,
                                        depths=st.session_state.VIZ_depths_smooth,
                                        path=path_name,
                                        zmin=st.session_state.VIZ_vmin_vmax[0],
                                        zmax=st.session_state.VIZ_vmin_vmax[1],
                                        cmap=st.session_state.VIZ_cmap)
        if st.session_state.VIZ_smoothing == 'Smooth':
            if st.session_state.VIZ_v_xd_smooth_smooth is not None:
                display_inverted_section(vs_section=st.session_state.VIZ_v_xd_smooth_smooth,
                                        std_section=st.session_state.VIZ_std_xd_smooth_smooth,
                                        positions=st.session_state.VIZ_xmids,
                                        depths=st.session_state.VIZ_depths_smooth,
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
                    pvc = np.loadtxt(f"{st.session_state.VIZ_folder_path}/{folder}/inv/{folder}_median_smooth_M{mode}.pvc")
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

    st.success(f"📁 Selected folder: **{st.session_state.VIZ_folder_path}**")
    
    folders = [xmid for xmid in os.listdir(st.session_state.VIZ_folder_path) if xmid[0:4] == 'xmid']
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
                    ws, vs = resamp_wavelength(fs, vs)
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
                    fs, vs = resamp_frequency(fs, vs)
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

    st.success(f"📁 Selected folder: **{st.session_state.VIZ_folder_path}**")
    
    folders = [xmid for xmid in os.listdir(st.session_state.VIZ_folder_path) if xmid[0:4] == 'xmid']
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
    
    if st.session_state.VIZ_v_xd_ridge_raw is None and st.session_state.VIZ_v_xd_layered_raw is None and st.session_state.VIZ_v_xd_smooth_raw is None:
        load_sections()
    
    if st.session_state.VIZ_v_xd_ridge_raw is None and st.session_state.VIZ_v_xd_layered_raw is None and st.session_state.VIZ_v_xd_smooth_raw is None:
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
        columns = st.columns(3)
        with columns[0]:
            st.radio("**Colormap**", ['Terrain', 'Viridis', 'Jet', 'Gray'],
                key='VIZ_cmap')
        with columns[1]:
            st.radio("**Model**", ['Smooth', 'Layered', 'Ridge'],
                key='VIZ_model')
        with columns[2]:
            st.radio("**Lateral smoothing**", ['Smooth', 'Raw'],
                key='VIZ_smoothing')
            
    st.text('')
    st.text('')
        
    if st.session_state.VIZ_model == 'Ridge':
        
        if st.session_state.VIZ_v_xd_ridge_raw is not None:
            
            vs_min = st.session_state.VIZ_vs_min_ridge
            vs_max = st.session_state.VIZ_vs_max_ridge
            
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
            if st.session_state.VIZ_smoothing == 'Smooth':
                fig = plot_inverted_section(st.session_state.VIZ_v_xd_ridge_smooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_ridge, zmin=vmin, zmax=vmax)
            elif st.session_state.VIZ_smoothing == 'Raw':
                fig = plot_inverted_section(st.session_state.VIZ_v_xd_ridge_raw, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_ridge, zmin=vmin, zmax=vmax)
                
            fig.update_layout(height=400)
            if st.session_state.VIZ_cmap is not None:
                if st.session_state.VIZ_cmap == 'Terrain':
                        cmap = plt.get_cmap('terrain')
                        colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]
                        fig.update_coloraxes(colorscale=colorscale)
                else:
                    fig.update_coloraxes(colorscale=st.session_state.VIZ_cmap)
            st.plotly_chart(fig)
            
            if st.session_state.VIZ_smoothing == 'Smooth':
                fig = plot_std_section(st.session_state.VIZ_std_xd_ridge_smooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_ridge)
            elif st.session_state.VIZ_smoothing == 'Raw':
                fig = plot_std_section(st.session_state.VIZ_std_xd_ridge_raw, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_ridge)
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
            st.error(f"📁 Inversion data missing for ridge model.")
            
    if st.session_state.VIZ_model == 'Layered':
        
        if st.session_state.VIZ_v_xd_layered_raw is not None:
            
            vs_min = st.session_state.VIZ_vs_min_layered
            vs_max = st.session_state.VIZ_vs_max_layered
            
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
            if st.session_state.VIZ_smoothing == 'Smooth':
                fig = plot_inverted_section(st.session_state.VIZ_v_xd_layered_smooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_layered, zmin=vmin, zmax=vmax)
            elif st.session_state.VIZ_smoothing == 'Raw':
                fig = plot_inverted_section(st.session_state.VIZ_v_xd_layered_raw, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_layered, zmin=vmin, zmax=vmax)
                
            fig.update_layout(height=400)
            if st.session_state.VIZ_cmap is not None:
                if st.session_state.VIZ_cmap == 'Terrain':
                        cmap = plt.get_cmap('terrain')
                        colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]
                        fig.update_coloraxes(colorscale=colorscale)
                else:
                    fig.update_coloraxes(colorscale=st.session_state.VIZ_cmap)
            st.plotly_chart(fig)
            
            if st.session_state.VIZ_smoothing == 'Smooth':
                fig = plot_std_section(st.session_state.VIZ_std_xd_layered_smooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_layered)
            elif st.session_state.VIZ_smoothing == 'Raw':
                fig = plot_std_section(st.session_state.VIZ_std_xd_layered_raw, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_layered)
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
            st.error(f"📁 Inversion data missing for layered model.")
    
    if st.session_state.VIZ_model == 'Smooth':
        
        if st.session_state.VIZ_v_xd_smooth_raw is not None:
            
            vs_min = st.session_state.VIZ_vs_min_smooth
            vs_max = st.session_state.VIZ_vs_max_smooth
            
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
            if st.session_state.VIZ_smoothing == 'Smooth':
                fig = plot_inverted_section(st.session_state.VIZ_v_xd_smooth_smooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_smooth, zmin=vmin, zmax=vmax)
            elif st.session_state.VIZ_smoothing == 'Raw':
                fig = plot_inverted_section(st.session_state.VIZ_v_xd_smooth_raw, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_smooth, zmin=vmin, zmax=vmax)
                
            fig.update_layout(height=400)
            if st.session_state.VIZ_cmap is not None:
                if st.session_state.VIZ_cmap == 'Terrain':
                        cmap = plt.get_cmap('terrain')
                        colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]
                        fig.update_coloraxes(colorscale=colorscale)
                else:
                    fig.update_coloraxes(colorscale=st.session_state.VIZ_cmap)
            st.plotly_chart(fig)
            
            if st.session_state.VIZ_smoothing == 'Smooth':
                fig = plot_std_section(st.session_state.VIZ_std_xd_smooth_smooth, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_smooth)
            elif st.session_state.VIZ_smoothing == 'Raw':
                fig = plot_std_section(st.session_state.VIZ_std_xd_smooth_raw, st.session_state.VIZ_xmids, st.session_state.VIZ_depths_smooth)
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
            st.error(f"📁 Inversion data missing for smooth model.")
                    
    
st.divider() # --------------------------------------------------------------------------------------------------------------------------------------
### END INTERFACE------------------------------------------------------------------------------------------------------------------------------------