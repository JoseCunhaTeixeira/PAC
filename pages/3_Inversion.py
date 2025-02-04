"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import concurrent.futures
import subprocess
import json
import pandas as pd
import time
from scipy.ndimage import generic_filter

from modules.dispersion import resamp_wavelength, resamp_frequency
from modules.display import plot_pseudo_section, display_inverted_section, display_pseudo_sections
from modules.misc import arange
from Paths import output_dir, work_dir

import warnings
warnings.filterwarnings("ignore")



### FUNCTIONS ---------------------------------------------------------------------------------------------------------------------------------------
def run_script(script):
    command = ["python3"] + script.split()
    subprocess.run(command)
    
def clear_session():
    st.cache_data.clear()
    st.session_state.clear()

def initialize_session():
    for key in st.session_state:
        if 'INV' not in key:
            st.session_state.pop(key)
            
    if "INV_folders" not in st.session_state:
        st.session_state.INV_folders = None
    
    if 'INV_status' not in st.session_state:
        st.session_state.INV_status = None
        
    if 'INV_all_selected' not in st.session_state:
        st.session_state.INV_all_selected = False
        
    if 'INV_nb_layers' not in st.session_state:
        st.session_state.INV_nb_layers = None
        
    if 'INV_modes' not in st.session_state:
        st.session_state.INV_modes = []
        
    if 'INV_vs_min_1' not in st.session_state:
        st.session_state.INV_vs_min_1 = None
    
    if 'INV_n_iterations' not in st.session_state:
        st.session_state.INV_n_iterations = None
        
def mode_filter_median(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mode = np.nanmedian(values)
    return mode
### -------------------------------------------------------------------------------------------------------------------------------------------------



### HANDLERS ----------------------------------------------------------------------------------------------------------------------------------------
def update_selection():
    if st.session_state.INV_all_selected:
        for status, position in zip(st.session_state.INV_status, st.session_state.INV_positions):
            if status == False:
                st.session_state[f"INV_{position}_selected"] = True
    else:
        for position in st.session_state.INV_positions:
            st.session_state[f"INV_{position}_selected"] = False

def update_all_checkbox():
    if all([st.session_state[f"INV_{position}_selected"] for position, status in zip(st.session_state.INV_positions, st.session_state.INV_status) if status == False]):
        st.session_state.INV_all_selected = True
    else:
        st.session_state.INV_all_selected = False
                       
def set_nb_layers_INV():
    if st.session_state.INV_nb_layers is not None:
        st.session_state.INV_nb_layers = round(st.session_state.INV_nb_layers, 0)
        if st.session_state.INV_nb_layers < 2:
            st.session_state.INV_nb_layers = 2
            
def set_thickness():
    if st.session_state.INV_nb_layers is not None :
        for layer in range(1, st.session_state.INV_nb_layers+1):
            if st.session_state[f'INV_thickness_min_{layer}'] is not None :
                st.session_state[f'INV_thickness_min_{layer}'] = round(st.session_state[f'INV_thickness_min_{layer}'], 2)
                if st.session_state[f'INV_thickness_min_{layer}'] <= 0:
                    st.session_state[f'INV_thickness_min_{layer}'] = 0.01
            if st.session_state[f'INV_thickness_max_{layer}'] is not None:
                st.session_state[f'INV_thickness_max_{layer}'] = round(st.session_state[f'INV_thickness_max_{layer}'], 2)
                if st.session_state[f'INV_thickness_max_{layer}'] <= 0:
                    st.session_state[f'INV_thickness_max_{layer}'] = 0.01    
            if st.session_state[f'INV_thickness_min_{layer}'] is not None and st.session_state[f'INV_thickness_max_{layer}'] is not None:
                    if st.session_state[f'INV_thickness_max_{layer}'] <= st.session_state[f'INV_thickness_min_{layer}']:
                        st.session_state[f'INV_thickness_max_{layer}'] = st.session_state[f'INV_thickness_min_{layer}'] + 0.01
            if st.session_state[f'INV_thickness_perturb_std_{layer}'] is not None:
                st.session_state[f'INV_thickness_perturb_std_{layer}'] = round(st.session_state[f'INV_thickness_perturb_std_{layer}'], 2)
                if st.session_state[f'INV_thickness_perturb_std_{layer}'] <= 0:
                    st.session_state[f'INV_thickness_perturb_std_{layer}'] = 0.01
                
def set_vs():
    if st.session_state.INV_nb_layers is not None :
        for layer in range(1, st.session_state.INV_nb_layers+1):
            if st.session_state[f'INV_vs_min_{layer}'] is not None :
                st.session_state[f'INV_vs_min_{layer}'] = round(st.session_state[f'INV_vs_min_{layer}'], 2)
                if st.session_state[f'INV_vs_min_{layer}'] <= 0:
                    st.session_state[f'INV_vs_min_{layer}'] = 0.01   
            if st.session_state[f'INV_vs_max_{layer}'] is not None:
                st.session_state[f'INV_vs_max_{layer}'] = round(st.session_state[f'INV_vs_max_{layer}'], 2)
                if st.session_state[f'INV_vs_max_{layer}'] <= 0:
                    st.session_state[f'INV_vs_max_{layer}'] = 0.01   
            if st.session_state[f'INV_vs_min_{layer}'] is not None and st.session_state[f'INV_vs_max_{layer}'] is not None:
                    if st.session_state[f'INV_vs_max_{layer}'] <= st.session_state[f'INV_vs_min_{layer}']:
                        st.session_state[f'INV_vs_max_{layer}'] = st.session_state[f'INV_vs_min_{layer}'] + 0.01
            if st.session_state[f'INV_vs_perturb_std_{layer}'] is not None:
                st.session_state[f'INV_vs_perturb_std_{layer}'] = round(st.session_state[f'INV_vs_perturb_std_{layer}'], 2)
                if st.session_state[f'INV_vs_perturb_std_{layer}'] <= 0:
                    st.session_state[f'INV_vs_perturb_std_{layer}'] = 0.01
                    
def set_mode():
    if st.session_state.INV_positions is not None and st.session_state.INV_status is not None:
        for position, status in zip(st.session_state.INV_positions, st.session_state.INV_status):
            if st.session_state[f"INV_{position}_selected"] is not None:
                st.session_state[f"INV_{position}_selected"] = False
    if st.session_state.INV_all_selected is not None:
        st.session_state.INV_all_selected = False
        
def set_running_params():
    if st.session_state.INV_n_iterations is not None:
        st.session_state.INV_n_iterations = round(st.session_state.INV_n_iterations, 0)
        if st.session_state.INV_n_iterations < 1:
            st.session_state.INV_n_iterations = 1
            
    if st.session_state.INV_n_burnin_iterations is not None:
        st.session_state.INV_n_burnin_iterations = round(st.session_state.INV_n_burnin_iterations, 0)
        if st.session_state.INV_n_burnin_iterations < 1:
            st.session_state.INV_n_burnin_iterations = 1
    if st.session_state.INV_n_chains is not None:
        st.session_state.INV_n_chains = round(st.session_state.INV_n_chains, 0)
        if st.session_state.INV_n_chains < 1:
            st.session_state.INV_n_chains = 1

def set_algorithm():
    if st.session_state.INV_nb_layers is not None:
        st.session_state.INV_nb_layers = None
    if st.session_state.INV_vs_min_1 is not None:
        st.session_state.INV_vs_min_1 = None
    if st.session_state.INV_n_iterations is not None:
        st.session_state.INV_n_iterations = None
    pass

def handle_select_folder():
    selected_folder_tmp = st.session_state.INV_selected_folder
    clear_session()
    initialize_session()
    st.session_state.INV_selected_folder = selected_folder_tmp
    
    if st.session_state.INV_selected_folder is not None:
        st.session_state.INV_folder_path = f"{output_dir}/{st.session_state.INV_selected_folder}/"
    
        # xmids folders
        INV_folders = [folder for folder in os.listdir(st.session_state.INV_folder_path) if os.path.isdir(os.path.join(st.session_state.INV_folder_path, folder))]
        
        # Positions of xmids folders
        INV_positions = [float(folder[4:]) for folder in INV_folders]
        INV_positions, INV_folders = zip(*sorted(zip(INV_positions, INV_folders)))
        
        st.session_state.INV_folders = INV_folders
        st.session_state.INV_positions = INV_positions
### -------------------------------------------------------------------------------------------------------------------------------------------------



### START INTERFACE ---------------------------------------------------------------------------------------------------------------------------------
initialize_session()

st.set_page_config(
    layout="centered",
    page_title="Inversion",
    page_icon="📐",
    initial_sidebar_state="expanded"
)

st.title("📐 Inversion")
st.write("🛈 Surface wave dispersion inversion.")

st.divider() # --------------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Data selection")

st.text('')
st.text('')

# Folder selection
files_depth_3 = glob.glob(f"{output_dir}/*/*/*")
input_folders = filter(lambda f: os.path.isdir(f), files_depth_3)
input_folders = [os.path.relpath(folder, output_dir) for folder in input_folders]
input_folders = sorted(input_folders)
if input_folders:
    st.selectbox("**Data folder**", input_folders, key='INV_selected_folder', on_change=handle_select_folder, index=None, placeholder='Select')
else:
    st.error("❌ No output data folders found.")
    st.stop()
    
if st.session_state.INV_folders is None:
    if st.session_state.INV_modes is not None:
        st.session_state.INV_modes = []
    st.info("👆 Please select a folder containing the picked dispersion curves to be inverted.")
    st.stop()
    
st.success("👌 Dispersion curves loaded.")

st.text('')
st.text('')

st.markdown("**Summary:**")
data = {
    'Folder' : [st.session_state.INV_folder_path],
    'MASW positions [m]' : [st.session_state.INV_positions],
    'Number of positions' : [len(st.session_state.INV_positions)]
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True, use_container_width=True)

st.divider() # --------------------------------------------------------------------------------------------------------------------------------------

st.header("📊 Picked pseudo-sections")

st.text('')
st.text('')

# Read picked modes xmid folder
nb_picked_modes_by_position = []
picked_modes_by_position = []
for folder, pos in zip(st.session_state.INV_folders, st.session_state.INV_positions):
    if os.path.exists(f"{st.session_state.INV_folder_path}/{folder}/pick/"):
        picked_modes = [int(fname.split("_M")[1].split(".")[0]) for fname in os.listdir(f"{st.session_state.INV_folder_path}/{folder}/pick/") if fname.endswith(".pvc") and fname.startswith(f"xmid{pos}_obs_M")]
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

if len(distinct_modes) == 0:
    st.text('')
    st.text('')
    st.error("❌ No picked dispersion data found. Select another folder.")
    st.divider() # ------------------------------------------------------------------------------------------------------------------------------
    st.stop()

# Display bars and pseudo-sections    
with st.container(border=True):
    st.markdown("**Display mode:**")
    on = st.toggle("OFF: Frequencies | ON: Wavelengths", value=False)

for mode, count in distinct_modes.items():
    with st.container(border=True):
        st.subheader(f"M{mode}")
        my_bar = st.progress(count/len(st.session_state.INV_positions), text=f'{count}/{len(st.session_state.INV_positions)} picked')
        if on:
            ws_per_position = []
            vs_per_position = []
            for j, folder in enumerate(st.session_state.INV_folders):
                try :
                    pvc = np.loadtxt(f"{st.session_state.INV_folder_path}/{folder}/pick/{folder}_obs_M{mode}.pvc")
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
            obs_v_wx = np.full((len(st.session_state.INV_positions), len(obs_ws)), np.nan)
            for j, (ws, vs) in enumerate(zip(ws_per_position, vs_per_position)):
                if len(ws) > 0:
                    fi_start = np.where(obs_ws >= ws[0])[0][0]
                    fi_end = np.where(obs_ws >= ws[-1])[0][0]
                    obs_v_wx[j, fi_start:fi_end+1] = vs
            fig = plot_pseudo_section(obs_v_wx, obs_ws, st.session_state.INV_positions, wavelength=True)
            st.plotly_chart(fig)
        else:
            fs_per_position = []
            vs_per_position = []
            for j, folder in enumerate(st.session_state.INV_folders):
                try :
                    pvc = np.loadtxt(f"{st.session_state.INV_folder_path}/{folder}/pick/{folder}_obs_M{mode}.pvc")
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
            obs_v_fx = np.full((len(st.session_state.INV_positions), len(obs_fs)), np.nan)
            for j, (fs, vs) in enumerate(zip(fs_per_position, vs_per_position)):
                if len(fs) > 0:
                    fi_start = np.where(obs_fs >= fs[0])[0][0]
                    fi_end = np.where(obs_fs >= fs[-1])[0][0]
                    obs_v_fx[j, fi_start:fi_end+1] = vs
            fig = plot_pseudo_section(obs_v_fx, obs_fs, st.session_state.INV_positions)
            st.plotly_chart(fig)
            
st.divider() # ------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Mode to invert")
st.text('')
st.text('')

options = [f"M{mode}" for mode in distinct_modes.keys()] # To be added in futur versions
st.multiselect("Modes to invert", options, key='INV_modes', on_change=set_mode)
st.markdown("🛈 *The maximum number of selected modes will be inverted if they were picked.*")

if len(st.session_state.INV_modes) == 0:
    if st.session_state.INV_nb_layers is not None:
        st.session_state.INV_nb_layers = None
    if st.session_state.INV_vs_min_1 is not None:
        st.session_state.INV_vs_min_1 = None
    if st.session_state.INV_n_iterations is not None:
        st.session_state.INV_n_iterations = None
    st.text('')
    st.text('')
    st.info("👆 Select a mode to invert.")
    st.stop()
    
st.text('')
st.text('')
st.success("👌 Mode selected.")

st.divider() # ------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Positions to invert")
st.text('')
st.text('')

st.session_state.INV_status = []
for i, position in enumerate(st.session_state.INV_positions):
    state = False
    for mode in st.session_state.INV_modes:
        if int(mode[1:]) in picked_modes_by_position[i]:
            state = state or True
    st.session_state.INV_status.append(not state)

num_columns = 5

columns = st.columns(num_columns)

with columns[0]:
    st.checkbox("All", key="INV_all_selected", on_change=update_selection)

st.session_state.selections = []
c = 1
for i, position in enumerate(st.session_state.INV_positions):
    with columns[c]:
        st.session_state.selections.append(st.checkbox(f"{position}", key=f"INV_{position}_selected", on_change=update_all_checkbox, disabled=st.session_state.INV_status[i]))
    c += 1
    if c == num_columns:
        c = 0
                
while c < num_columns:
    with columns[c]:
        st.empty()
    c += 1

st.session_state.INV_selection = [position for position, selection in zip(st.session_state.INV_positions, st.session_state.selections) if selection]
st.session_state.INV_nb_scripts = len(st.session_state.INV_selection)
                    
if st.session_state.INV_nb_scripts == 0:
    st.text('')
    st.text('')
    if st.session_state.INV_nb_layers is not None:
        st.session_state.INV_nb_layers = None
    if st.session_state.INV_vs_min_1 is not None:
        st.session_state.INV_vs_min_1 = None
    if st.session_state.INV_n_iterations is not None:
        st.session_state.INV_n_iterations = None
    st.warning("❌ No selected positions to invert.")
    st.stop()

st.text('')
st.text('')
st.success(f"👌 {st.session_state.INV_nb_scripts} selected positions to invert.")

st.divider() # ------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Parameter space")
st.text('')
st.text('')

st.number_input("Number of layers", value=None, step=1, key='INV_nb_layers', placeholder='Enter a value', format="%i", on_change=set_nb_layers_INV)

if st.session_state.INV_nb_layers is None:
    st.text('')
    st.text('')
    if st.session_state.INV_vs_min_1 is not None:
        st.session_state.INV_vs_min_1 = None
    if st.session_state.INV_n_iterations is not None:
        st.session_state.INV_n_iterations = None
    st.info("👆 Define the number of layers.")
    st.stop()

thickness_mins = []
thickness_maxs = []
thickness_perturb_stds = []
vs_mins = []
vs_maxs = []
vs_perturb_stds = []

for i in range(1, st.session_state.INV_nb_layers+1):
    st.text('')
    with st.container(border=True):
        
        if i == st.session_state.INV_nb_layers:
            st.markdown(f"**Half-space:**")
            columns = st.columns(3)
            with columns[0]:
                thickness_min = st.number_input("Min thickness [m]", key=f'INV_thickness_min_{i}', value=None, disabled=True, placeholder="∞")
                thickness_mins.append(thickness_min)
            with columns[1]:
                thickness_max = st.number_input("Max thickness [m]", key=f'INV_thickness_max_{i}', value=None, disabled=True, placeholder="∞")
                thickness_maxs.append(thickness_max)
            with columns[2]:
                perturb_std = st.number_input("Perturbation std [m]", key=f'INV_thickness_perturb_std_{i}', value=None, disabled=True, placeholder="0")
                thickness_perturb_stds.append(perturb_std)
        else:
            st.markdown(f"**Layer {i}:**")
            columns = st.columns(3)
            with columns[0]:
                thickness_min = st.number_input("Min thickness [m]", key=f'INV_thickness_min_{i}', value=None, step=0.1, on_change=set_thickness, placeholder='Enter a value', format="%.2f")
                thickness_mins.append(thickness_min)
            with columns[1]:
                thickness_max = st.number_input("Max thickness [m]", key=f'INV_thickness_max_{i}', value=None, step=0.1, on_change=set_thickness, placeholder='Enter a value', format="%.2f")
                thickness_maxs.append(thickness_max)
            with columns[2]:
                perturb_std = st.number_input("Perturbation std [m]", key=f'INV_thickness_perturb_std_{i}', value=None, step=0.1, on_change=set_thickness, placeholder="Enter a value", format="%.2f")
                thickness_perturb_stds.append(perturb_std)
        
        st.text('')
        
        columns = st.columns(3)
        with columns[0]:
            vs_min = st.number_input("Min shear wave velocity [m/s]", key=f'INV_vs_min_{i}', value=None, step=1.0, placeholder='Enter a value', format="%.2f", on_change=set_vs)
            vs_mins.append(vs_min)
        with columns[1]:
            vs_max = st.number_input("Max shear wave velocity [m/s]", key=f'INV_vs_max_{i}', value=None, step=1.0, placeholder='Enter a value', format="%.2f", on_change=set_vs)
            vs_maxs.append(vs_max)
        with columns[2]:
            perturb_std = st.number_input("Perturbation std [m/s]", key=f'INV_vs_perturb_std_{i}', value=None, step=1.0, placeholder="Enter a value", format="%.2f", on_change=set_vs)
            vs_perturb_stds.append(perturb_std)

if None in thickness_mins[:-1] or None in thickness_maxs[:-1] or None in vs_mins or None in vs_maxs or None in thickness_perturb_stds[:-1] or None in vs_perturb_stds:
    st.text('')
    st.text('')
    if st.session_state.INV_n_iterations is not None:
        st.session_state.INV_n_iterations = None
    st.info("👆 Define the entire parameter space.")
    st.stop()
    
st.text('')
st.text('')
st.success("👌 Parameter space defined.")


st.divider() # ------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Running parameters")
st.text('')
st.text('')
    
st.number_input("Iterations [#]", key='INV_n_iterations', value=None, step=1, on_change=set_running_params, placeholder='Enter a value', format="%i")
st.number_input("Burning iterations [#]", key='INV_n_burnin_iterations', value=None, step=1, on_change=set_running_params, placeholder='Enter a value', format="%i")
st.number_input("Chains [#]", key='INV_n_chains', value=None, step=1, on_change=set_running_params, placeholder='Enter a value', format="%i")

if st.session_state.INV_n_iterations is None or st.session_state.INV_n_burnin_iterations is None or st.session_state.INV_n_chains is None:
    st.text('')
    st.text('')
    st.info("👆 Define the number of subprocesses.")
    st.stop()
          
st.text('')
st.text('')
st.success("👌 Parallelization parameters defined.")

st.divider() # ----------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Run Inversion")
st.text('')
st.text('')

if st.button("Compute", type="primary", use_container_width=True):
    st.text('')
    st.text('')
    loading_message = st.empty()
    loading_message.warning(f"⏳⚙️ Running inversion...")
    
    params = {
        "date" : time.strftime("%Y-%m-%d %H:%M:%S"),
        "folder_path" : st.session_state.INV_folder_path,
        'positions': st.session_state.INV_selection,
        'inversion': {
            'modes': sorted([int(mode.split()[0][1:]) for mode in st.session_state.INV_modes]),
            'nb_layers': st.session_state.INV_nb_layers,
            'thickness_mins': thickness_mins,
            'thickness_maxs': thickness_maxs,
            'vs_mins': vs_mins,
            'vs_maxs': vs_maxs,
        },
        "running_distribution": {},
    }
    for i, position in enumerate(st.session_state.INV_selection):
        params["running_distribution"][str(i)] = {
            "x_mid" : position,
        }
    params["inversion"]["vs_perturb_stds"] = vs_perturb_stds
    params["inversion"]["thickness_perturb_stds"] = thickness_perturb_stds
    params["inversion"]["n_iterations"] = st.session_state.INV_n_iterations
    params["inversion"]["n_burnin_iterations"] = st.session_state.INV_n_burnin_iterations
    params["inversion"]["n_chains"] = st.session_state.INV_n_chains
    with open( f"{st.session_state.INV_folder_path}/inversion_params.json", "w") as file:
        json.dump(params, file, indent=2)
    
    print("\033[1m\n\nRunning inversion...\033[0m")
    start = time.time()
    
    # Run inversion scripts
    scripts = [f"{work_dir}/scripts/run_inversion.py -ID {i} -r {st.session_state.INV_folder_path}" for i in range(st.session_state.INV_nb_scripts)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        executor.map(run_script, scripts)
    
    # Plot entire inverted vs section
    print("\033[1mPlotting inverted section...\033[0m")
    all_gm = []
    all_std = []
    for folder in st.session_state.INV_folders:
        try:
            gm = np.loadtxt(f"{st.session_state.INV_folder_path}/{folder}/inv/{folder}_median_smooth_model.gm", skiprows=1)
            std = np.loadtxt(f"{st.session_state.INV_folder_path}/{folder}/inv/{folder}_median_smooth_std.gm", skiprows=1)
        except:
            gm = None
            std = None
        all_gm.append(gm)
        all_std.append(std)
    dz = 0.1
    depth_max = np.nanmax([np.sum(gm[:,0]) for gm in all_gm if gm is not None])
    if not all(gm is None for gm in all_gm) and not all(gm is None for gm in all_std): 
        depths = arange(0, depth_max, dz)
        v_xd = np.full((len(st.session_state.INV_folders), len(depths)), np.nan)
        std_xd = np.full((len(st.session_state.INV_folders), len(depths)), np.nan)
        for j, (gm, gm_std) in enumerate(zip(all_gm, all_std)):
            if gm is not None and gm_std is not None:
                col = []
                col_std = []
                for (thick, vp, vs, rho), (thick_std, vp_std, vs_std, rho_std) in zip(gm, gm_std):
                    col += [vs] * int(thick/dz)
                    col_std += [vs_std] * int(thick/dz)
                if len(col) < len(depths):
                    col += [gm[-1,2]] * (len(depths)-len(col))
                    col_std += [gm_std[-1,2]] * (len(depths)-len(col_std))
                v_xd[j, :] = col
                std_xd[j, :] = col_std
        v_xd_smooth = generic_filter(v_xd, mode_filter_median, size=(4,1))
        v_xd_smooth = generic_filter(v_xd_smooth, mode_filter_median, size=(3,1))
        v_xd_smooth = generic_filter(v_xd_smooth, mode_filter_median, size=(2,1))
        std_xd_smooth = generic_filter(std_xd, mode_filter_median, size=(4,1))
        std_xd_smooth = generic_filter(std_xd_smooth, mode_filter_median, size=(3,1))
        std_xd_smooth = generic_filter(std_xd_smooth, mode_filter_median, size=(2,1))
        path_name = f"{st.session_state.INV_folder_path}/vs_section.svg"
        display_inverted_section(v_xd_smooth, std_xd_smooth, st.session_state.INV_positions, depths, path_name)
        
    # Plot vr sections
    print("\033[1mPlotting pseudo-sections...\033[0m")
    for mode in st.session_state.INV_modes:
        # Observed
        obs_fs_per_position = []
        obs_vs_per_position = []
        for j, folder in enumerate(st.session_state.INV_folders):
            try :
                pvc = np.loadtxt(f"{st.session_state.INV_folder_path}/{folder}/pick/{folder}_obs_{mode}.pvc")
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
        for j, folder in enumerate(st.session_state.INV_folders):
            try :
                pvc = np.loadtxt(f"{st.session_state.INV_folder_path}/{folder}/inv/{folder}_median_smooth_{mode}.pvc")
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
        pred_v_fx = np.full((len(st.session_state.INV_positions), len(fs)), np.nan)
        obs_v_fx = np.full((len(st.session_state.INV_positions), len(fs)), np.nan)
        for j, (obs_fs, obs_vs, pred_fs, pred_vs) in enumerate(zip(obs_fs_per_position, obs_vs_per_position, pred_fs_per_position, pred_vs_per_position)):
            if len(obs_fs) > 0:
                fi_start = np.where(fs >= obs_fs[0])[0][0]
                fi_end = np.where(fs >= obs_fs[-1])[0][0]
                obs_v_fx[j, fi_start:fi_end+1] = obs_vs
            if len(pred_fs) > 0:
                fi_start = np.where(fs >= pred_fs[0])[0][0]
                fi_end = np.where(fs >= pred_fs[-1])[0][0]
                pred_v_fx[j, fi_start:fi_end+1] = pred_vs
        path_name = f"{st.session_state.INV_folder_path}/{mode}_vr_pseudo-section.svg"
        display_pseudo_sections(obs_v_fx, pred_v_fx, fs, st.session_state.INV_positions, path_name)
    
    end = time.time()
    print(f"\033[1mInversion completed in {end - start:.2f} seconds.\033[0m")
        
    loading_message.empty()
    st.text('')
    st.text('')
    st.success(f"👌 Inversion completed for {st.session_state.INV_nb_scripts} MASW postions.")
    st.info(f"🕒 Inversion took {end - start:.2f} seconds.")
    
else:
    st.text('')
    st.text('')
    st.info("👆 Click on the 'Compute' button to run the inversions.")

st.divider() # --------------------------------------------------------------------------------------------------------------------------------------
### END INTERFACE------------------------------------------------------------------------------------------------------------------------------------