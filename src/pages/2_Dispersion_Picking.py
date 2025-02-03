"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : Decdember 18, 2024
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import json
import pandas as pd
import plotly.graph_objects as go

from modules.dispersion import extract_curve, lorentzian_error, resamp, resamp_wavelength, resamp_frequency
from modules.display import plot_disp, plot_pseudo_section, display_dispersion_img
from modules.misc import arange

import warnings
warnings.filterwarnings("ignore")


### FUNCTIONS --------------------------------------------------------------------------------------------------------------------------------------
def select_folder():
   root = tk.Tk()
   root.withdraw()
   folder_path = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path

def clear_session():
    st.cache_data.clear()
    st.session_state.clear()

def initialize_session():
    if 'DISP_folders' not in st.session_state:
        st.session_state.DISP_folders = None
        
    if 'DISP_index' not in st.session_state:
        st.session_state.DISP_index = 0
        
    if "DISP_clicked_pick" not in st.session_state:
        st.session_state.DISP_clicked_pick = False
### -------------------------------------------------------------------------------------------------------------------------------------------------



### HANDLERS ----------------------------------------------------------------------------------------------------------------------------------------
def handle_picked():
    if st.session_state.DISP_event:
        output_dir = f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        selected_data = st.session_state.DISP_event.selection['lasso']
        if selected_data:
            x = selected_data[0]['x']
            y = selected_data[0]['y']
            poly_coords = np.array([x, y]).T
            f_picked, v_picked = extract_curve(st.session_state.DISP_FV, st.session_state.DISP_fs, st.session_state.DISP_vs, poly_coords, smooth=False)
            dc = lorentzian_error(v_picked, f_picked, st.session_state.DISP_dx, st.session_state.DISP_Nx)
            f_picked, v_picked, dc = resamp(f_picked, v_picked, dc)#, wmax=50)
            st.session_state.DISP_fs_picked = f_picked
            st.session_state.DISP_vs_picked = v_picked
            st.session_state.DISP_dc_picked = dc
    pvc = np.array([st.session_state.DISP_fs_picked, st.session_state.DISP_vs_picked, st.session_state.DISP_dc_picked], dtype=float).T
    np.savetxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick/xmid{st.session_state.DISP_positions[i]}_obs_M{st.session_state.DISP_mode}.pvc", pvc, delimiter=' ', fmt='%.5f')
    
    FV = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/comp/xmid{st.session_state.DISP_positions[i]}_dispersion.csv", delimiter=',')
    fs = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/comp/xmid{st.session_state.DISP_positions[i]}_fs.csv", delimiter=',')
    vs = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/comp/xmid{st.session_state.DISP_positions[i]}_vs.csv", delimiter=',')
    name_path = f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick/xmid{st.session_state.DISP_positions[i]}_dispersion.svg"
    modes = [int(fname.split("_M")[1].split(".")[0]) for fname in os.listdir(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick/") if fname.endswith(".pvc") and fname.startswith(f"xmid{st.session_state.DISP_positions[i]}_obs_M")]
    arr = []
    for mode in modes:
        pvc = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick/xmid{st.session_state.DISP_positions[i]}_obs_M{mode}.pvc")
        arr.append(pvc)
    display_dispersion_img(FV, fs, vs, obs_modes=arr , path=name_path, normalization='Frequency', dx=st.session_state.DISP_dx)
    
    st.session_state.DISP_clicked_pick = False
        
def update_selected_option(selected, tabs):
    st.session_state.DISP_index = tabs.index(selected)

def clicked_clear(path):
    os.remove(path)
    
    FV = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/comp/xmid{st.session_state.DISP_positions[i]}_dispersion.csv", delimiter=',')
    fs = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/comp/xmid{st.session_state.DISP_positions[i]}_fs.csv", delimiter=',')
    vs = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/comp/xmid{st.session_state.DISP_positions[i]}_vs.csv", delimiter=',')
    name_path = f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick/xmid{st.session_state.DISP_positions[i]}_dispersion.svg"
    modes = [int(fname.split("_M")[1].split(".")[0]) for fname in os.listdir(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick") if fname.endswith(".pvc") and fname.startswith(f"xmid{st.session_state.DISP_positions[i]}_obs_M")]
    arr = []
    for mode in modes:
        pvc = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick/xmid{st.session_state.DISP_positions[i]}_obs_M{mode}.pvc")
        arr.append(pvc)
    display_dispersion_img(FV, fs, vs, obs_modes=arr , path=name_path, normalization='Frequency', dx=st.session_state.DISP_dx)
    
def set_clicked_pick():
    st.session_state.DISP_clicked_pick = True

def cancel_picking():
    st.session_state.DISP_clicked_pick = False
### -------------------------------------------------------------------------------------------------------------------------------------------------



### START INTERFACE ---------------------------------------------------------------------------------------------------------------------------------
initialize_session()

st.set_page_config(
    layout="centered",
    page_title="Dispersion Picking",
    page_icon="📌",
    initial_sidebar_state="expanded"
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

st.title("📌 Dispersion Picking")
st.write("🛈 This tab allows you to pick the computed dispersion images.")

st.divider() # --------------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Select folder")

st.text('')
st.text('')

# Folder selection button
folder_select_button = st.button("Select Folder", type="primary", use_container_width=True)

if folder_select_button:
    
    # Clear and initialize session
    clear_session()
    initialize_session()
    
    # Main folder with all xmids folders
    folder_path = select_folder()
    folder_path = f"{folder_path}/"
    st.session_state.DISP_folder_path = folder_path
    
    # xmids folders
    DISP_folders = [folder for folder in os.listdir(st.session_state.DISP_folder_path) if os.path.isdir(os.path.join(st.session_state.DISP_folder_path, folder))]
    
    # Positions of xmids folders
    DISP_positions = [float(folder[4:]) for folder in DISP_folders]
    
    # Sort by position
    DISP_positions, DISP_folders = zip(*sorted(zip(DISP_positions, DISP_folders)))
    
    # Save in session state
    st.session_state.DISP_folders = DISP_folders
    st.session_state.DISP_positions = DISP_positions
    
    # MASW parameters from json file
    with open(f"{st.session_state.DISP_folder_path}/computing_params.json", "r") as f:
        computing_param = json.load(f)
        st.session_state.DISP_Nx = computing_param["MASW_length"]
        st.session_state.DISP_dx = computing_param["positions"][1] - computing_param["positions"][0]       
        
if st.session_state.DISP_folders is None:
    st.text('')
    st.text('')
    st.info("👆 Select a folder containing the seismic files.")
    st.stop()
    
st.text('')
st.text('')

st.success("👌 Dispersion files loaded.")
data = {
    'Folder' : [st.session_state.DISP_folder_path],
    'MASW positions [m]' : [st.session_state.DISP_positions],
    'Number of positions' : [len(st.session_state.DISP_positions)]
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True, use_container_width=True)
    
st.divider() # ----------------------------------------------------------------------------------------------------------------------------------

st.header("🚨 Picking")

st.text('')
st.text('')

# Read picked modes xmid folder
nb_picked_modes_by_position = []
picked_modes_by_position = []
disp_data_exists = []
for folder, pos in zip(st.session_state.DISP_folders, st.session_state.DISP_positions):
    if os.path.exists(f"{st.session_state.DISP_folder_path}/{folder}/pick/"):
        picked_modes = [int(fname.split("_M")[1].split(".")[0]) for fname in os.listdir(f"{st.session_state.DISP_folder_path}/{folder}/pick/") if fname.endswith(".pvc") and fname.startswith(f"xmid{pos}_obs_M")]
        picked_modes = sorted(picked_modes)
    else:
        picked_modes = []
    picked_modes_by_position.append(picked_modes)
    nb_picked_modes_by_position.append(len(picked_modes))
    disp_data_exists.append(os.path.exists(f"{st.session_state.DISP_folder_path}/{folder}/comp/xmid{pos}_dispersion.csv") and os.path.exists(f"{st.session_state.DISP_folder_path}/{folder}/comp/xmid{pos}_fs.csv") and os.path.exists(f"{st.session_state.DISP_folder_path}/{folder}/comp/xmid{pos}_vs.csv"))


# Tabs for selection box
tabs1 = [f"{pos} m" for pos in st.session_state.DISP_positions]
tabs2 = [f"✅" if nb > 0 else f"✘" for nb in nb_picked_modes_by_position]
tabs3 = [f"{nb} mode picked" if nb == 1 else f"{nb} modes picked" for nb in nb_picked_modes_by_position]

tabs = [
    f"{tab1} ― {tab2} {tab3}" if exist else f"{tab1} ― ⛔ Unavailable"
    for tab1, tab2, tab3, exist in zip(tabs1, tabs2, tabs3, disp_data_exists)
]

# Selection box
st.session_state.DISP_position = st.selectbox(
    "**Select a position:**",
    tabs,
    index=st.session_state.DISP_index,
    key="DISP_selected",
    on_change=lambda: update_selected_option(st.session_state.DISP_selected, tabs)
)

if st.session_state.DISP_position is not None:
            
    i = tabs.index(st.session_state.DISP_position)
    
    st.session_state.DISP_picked = nb_picked_modes_by_position[i] > 0
    
    try :
        st.session_state.DISP_FV = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/comp/xmid{st.session_state.DISP_positions[i]}_dispersion.csv", delimiter=',')
    except:
        st.text('')
        st.text('')
        st.error("⛔ No dispersion data found.")
        st.divider() # ------------------------------------------------------------------------------------------------------------------------------
        st.stop()
    try :
        st.session_state.DISP_fs = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/comp/xmid{st.session_state.DISP_positions[i]}_fs.csv", delimiter=',')
    except:
        st.text('')
        st.text('')
        st.error("⛔ No frequency data found.")
        st.divider() # ------------------------------------------------------------------------------------------------------------------------------
        st.stop()
    try :
        st.session_state.DISP_vs = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/comp/xmid{st.session_state.DISP_positions[i]}_vs.csv", delimiter=',')
    except:
        st.text('')
        st.text('')
        st.error("⛔ No velocity data found.")
        st.divider() # ------------------------------------------------------------------------------------------------------------------------------
        st.stop()

    if not st.session_state.DISP_clicked_pick:
        
        fig = plot_disp(st.session_state.DISP_FV, st.session_state.DISP_fs, st.session_state.DISP_vs, norm="Frequencies", dx=st.session_state.DISP_dx, Nx=st.session_state.DISP_Nx)
            
        if st.session_state.DISP_picked:
            picked_modes = [fname for fname in os.listdir(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick/") if fname.endswith(".pvc") and fname.startswith(f"xmid{st.session_state.DISP_positions[i]}_obs_M")]
            
            for picked_mode in picked_modes:
                pvc = np.loadtxt(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick/{picked_mode}")
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
        
        fig.update_layout(xaxis_range=[min(st.session_state.DISP_fs), max(st.session_state.DISP_fs)])
        fig.update_layout(yaxis={'range': [min(st.session_state.DISP_vs), max(st.session_state.DISP_vs)], 'autorange': False})
        st.plotly_chart(fig)
    
    elif st.session_state.DISP_clicked_pick:
        
        fs_tmp = np.copy(st.session_state.DISP_fs)
        vs_tmp = np.copy(st.session_state.DISP_vs)
        FV_tmp = np.copy(st.session_state.DISP_FV)
            
        f_grid, v_grid = np.meshgrid(fs_tmp, vs_tmp)
        f_grid = f_grid.flatten()
        v_grid = v_grid.flatten()
        
        fig_pick = plot_disp(st.session_state.DISP_FV, st.session_state.DISP_fs, st.session_state.DISP_vs, norm='Frequencies', dx=st.session_state.DISP_dx, Nx=st.session_state.DISP_Nx)
                
        event = st.plotly_chart(fig_pick, selection_mode=["lasso"], on_select=handle_picked, key='DISP_event')
    
        st.warning(f"⚠️ Picking **Mode {st.session_state.DISP_mode}**.")
        st.info("📿 Use the lasso to draw a zone on the dispersion diagram where to pick the curve.")
        st.info("👇 Or click on the button 'Cancel picking' to cancel the picking.")
        st.button("Cancel picking", use_container_width=True, on_click=cancel_picking)
            
    if not st.session_state.DISP_clicked_pick:
        columns = st.columns(2, vertical_alignment="bottom")
        with columns[0]:
            st.session_state.DISP_mode = st.number_input("**Select a mode to pick:**", value=nb_picked_modes_by_position[i], min_value=0, step=1)
        with columns[1]:
            button = st.button("Start picking", type="primary", use_container_width=True, on_click=set_clicked_pick)
        if st.session_state.DISP_mode in picked_modes_by_position[i]:
            button = st.button("Delete", on_click=lambda: clicked_clear(f"{st.session_state.DISP_folder_path}/{st.session_state.DISP_folders[i]}/pick/xmid{st.session_state.DISP_positions[i]}_obs_M{st.session_state.DISP_mode}.pvc"),  use_container_width=True, type="secondary")
        st.markdown("🛈 *The picked dispersion curve is resampled in wavelengths. It is normal for the number of points to decrease at higher frequencies, and for the picking selection zone to not be accurately represented due to this process.*")

        
    st.divider() # ------------------------------------------------------------------------------------------------------------------------------

    st.header("📊 Picked pseudo-sections")

    st.text('')
    st.text('')
            
    # Extract distinct modes and count them
    distinct_modes = {}
    for picked_modes in picked_modes_by_position:
        for picked_mode in picked_modes:
            if picked_mode not in distinct_modes.keys():
                distinct_modes[picked_mode] = 1
            else:
                distinct_modes[picked_mode] += 1
    distinct_modes = dict(sorted(distinct_modes.items()))

    # Display bars and pseudo-sections
    if distinct_modes:
        
        with st.container(border=True):
            st.markdown("**Display mode:**")
            on = st.toggle("OFF: Frequencies | ON: Wavelengths", value=False)
        
        for mode, count in distinct_modes.items():
            with st.container(border=True):
                st.subheader(f"M{mode}")
                my_bar = st.progress(count/len(st.session_state.DISP_positions), text=f'{count}/{len(st.session_state.DISP_positions)} picked')
                if on:
                    ws_per_position = []
                    vs_per_position = []
                    for j, folder in enumerate(st.session_state.DISP_folders):
                        try :
                            pvc = np.loadtxt(f"{st.session_state.DISP_folder_path}/{folder}/pick/{folder}_obs_M{mode}.pvc")
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
                    obs_v_wx = np.full((len(st.session_state.DISP_positions), len(obs_ws)), np.nan)
                    for j, (ws, vs) in enumerate(zip(ws_per_position, vs_per_position)):
                        if len(ws) > 0:
                            fi_start = np.where(obs_ws >= ws[0])[0][0]
                            fi_end = np.where(obs_ws >= ws[-1])[0][0]
                            obs_v_wx[j, fi_start:fi_end+1] = vs
                    fig = plot_pseudo_section(obs_v_wx, obs_ws, st.session_state.DISP_positions, wavelength=True)
                    st.plotly_chart(fig)
                else:
                    fs_per_position = []
                    vs_per_position = []
                    for j, folder in enumerate(st.session_state.DISP_folders):
                        try :
                            pvc = np.loadtxt(f"{st.session_state.DISP_folder_path}/{folder}/pick/{folder}_obs_M{mode}.pvc")
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
                    obs_v_fx = np.full((len(st.session_state.DISP_positions), len(obs_fs)), np.nan)
                    for j, (fs, vs) in enumerate(zip(fs_per_position, vs_per_position)):
                        if len(fs) > 0:
                            fi_start = np.where(obs_fs >= fs[0])[0][0]
                            fi_end = np.where(obs_fs >= fs[-1])[0][0]
                            obs_v_fx[j, fi_start:fi_end+1] = vs
                    fig = plot_pseudo_section(obs_v_fx, obs_fs, st.session_state.DISP_positions)
                    st.plotly_chart(fig)
                
    else:
        st.write("")
        st.write("")
        st.warning("❌ No picked dispersion data.")
        
st.divider() # --------------------------------------------------------------------------------------------------------------------------------------
### END INTERFACE------------------------------------------------------------------------------------------------------------------------------------