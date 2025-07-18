"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""

import os
import sys
import glob
import time
import pandas as pd
import streamlit as st
from obspy import read
import subprocess
import concurrent.futures
import json
import plotly.graph_objects as go

from Paths import output_dir, input_dir, work_dir

import warnings
warnings.filterwarnings("ignore")



### FUNCTIONS ---------------------------------------------------------------------------------------------------------------------------------------
def plot_MASW(geophone_positions, MASW_length_idx, MASW_step_idx, source_positions):
    dx = geophone_positions[1] - geophone_positions[0]
    
    MASW_length = round((MASW_length_idx-1) * dx, 3)
    MASW_step = round((MASW_step_idx) * dx, 3)
    
    x_mids =[]
    windows = []
    windows_idx = []
    
    x_mid = MASW_length / 2 + geophone_positions[0]
    
    if MASW_step == 0:
        x_mids.append(x_mid)  
        
        w_start = geophone_positions.index(round(x_mid - (MASW_length / 2),3))
        w_end = geophone_positions.index(round(x_mid + (MASW_length / 2),3))
        windows_idx.append([w_start, w_end])
            
        windows.append([geophone_positions[w_start], geophone_positions[w_end]])
        
    else:
        while round(x_mid + (MASW_length / 2),3) <= geophone_positions[-1]:
            x_mids.append(x_mid)
            
            w_start = geophone_positions.index(round(x_mid - (MASW_length / 2),3))
            w_end = geophone_positions.index(round(x_mid + (MASW_length / 2),3))
            windows_idx.append([w_start, w_end])
            
            windows.append([geophone_positions[w_start], geophone_positions[w_end]])
            
            x_mid += MASW_step  
    
    # Create a scatter plot
    fig = go.Figure()
    
    # Add scatter points for selected sensors in blue
    fig.add_trace(go.Scatter(
        x=geophone_positions,
        y=[2] * len(geophone_positions),
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='#1f77b4'),
        showlegend=True,
        name='Sensor',
    ))
    
    # Add scatter points for MASW windows middle positions in red
    fig.add_trace(go.Scatter(
        x=x_mids,
        y=[1] * len(x_mids),
        mode='markers',
        marker=dict(symbol='star-triangle-down', size=10, color='#d62728'),
        showlegend=True,
        name='MASW positions',
    ))

    fig.add_trace(go.Scatter(
        x=source_positions,
        y=[3] * len(source_positions),
        mode='markers',
        marker=dict(symbol='star', size=10, color='#2ca02c'),
        showlegend=True,
        name='Source positions',
    ))

    # Update layout to hide the y-axis
    fig.update_layout(
        title="Sensor and MASW positions",
        xaxis=dict(
            title="Position [m]",
            side="top",
            ticks="outside",
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis_range=[0, 4],
        xaxis_range=[min(min(geophone_positions), min(source_positions))-5, max(max(geophone_positions), max(source_positions))+5],
        height=250,
    )

    return fig, x_mids, windows_idx

def run_script(script):
    python_cmd = sys.executable
    command = [python_cmd] + script.split()
    try:
        subprocess.run(command, check=True, stderr=subprocess.PIPE)
        return 0
    except Exception as e:
        return 1
    
def clear_session():
    st.cache_data.clear()
    st.session_state.clear()

def initialize_session():
    for key in st.session_state:
        if 'ACT' not in key:
            st.session_state.pop(key)
    if 'ACT_folder_path' not in st.session_state:
        st.session_state.ACT_folder_path = None
    if 'ACT_x_start' not in st.session_state:
        st.session_state.ACT_x_start = None
    if 'ACT_x_step' not in st.session_state:
        st.session_state.ACT_x_step = None
    if 'ACT_nb_max_subproc' not in st.session_state:
        st.session_state.ACT_nb_max_subproc = None
### -------------------------------------------------------------------------------------------------------------------------------------------------



### HANDLERS ----------------------------------------------------------------------------------------------------------------------------------------
def handle_select_folder():
    selected_folder_tmp = st.session_state.ACT_selected_folder
    clear_session()
    initialize_session()
    st.session_state.ACT_selected_folder = selected_folder_tmp
    
    if st.session_state.ACT_selected_folder is not None:    
        st.session_state.ACT_folder_path = f"{input_dir}/{st.session_state.ACT_selected_folder}/"

        # List all files in the folder
        files = [file for file in os.listdir(st.session_state.ACT_folder_path) if not file.startswith('.') and not file.endswith('.json')]
        if len(files) < 1:
            st.error("❌ Selected input data folder empty.")
            clear_session()
            st.stop()
        files = sorted(files)
        st.session_state.ACT_files = files

        # Read all seismic records
        st.session_state.ACT_durations = []
        for file in st.session_state.ACT_files:
            stream = read(st.session_state.ACT_folder_path + file)
            st.session_state.ACT_durations.append(stream[0].stats.endtime - stream[0].stats.starttime)
        st.session_state.ACT_N_traces = len(stream)
        del stream

        # Read source positions json file
        source_positions_file = f"{st.session_state.ACT_folder_path}/source_positions.json"
        if os.path.exists(source_positions_file):
            with open(source_positions_file, "r") as file:
                source_positions = json.load(file)
            source_positions = dict(sorted(source_positions.items(), key=lambda item: item[0]))                
            if list(source_positions.keys()) != st.session_state.ACT_files:
                st.error("❌ Source positions file does not match seismic files.")
                clear_session()
                st.stop()
            st.session_state.ACT_source_positions = list(source_positions.values())
        else:
            st.error("❌ No source positions file found in the selected folder.")
            clear_session()
            st.stop()
    

def handle_set():
    st.session_state.ACT_clicked_set = True

def set_f():
    if st.session_state.ACT_f_min is not None:
        if st.session_state.ACT_f_min < 0:
            st.session_state.ACT_f_min = 0
    if st.session_state.ACT_f_max is not None:
        if st.session_state.ACT_f_max < 0:
            st.session_state.ACT_f_max = 0
    if st.session_state.ACT_f_min is not None and st.session_state.ACT_f_max is not None:
        if st.session_state.ACT_f_max <= st.session_state.ACT_f_min:
            st.session_state.ACT_f_max = st.session_state.ACT_f_min + 1
                
def set_v():    
    if st.session_state.ACT_v_max is not None:
        st.session_state.ACT_v_max = round(st.session_state.ACT_v_max, 2)
        if st.session_state.ACT_v_max <= 0:
            st.session_state.ACT_v_max = 0.01
        if st.session_state.ACT_v_min is not None:
            st.session_state.ACT_v_min = round(st.session_state.ACT_v_min, 2)
            if st.session_state.ACT_v_max <= st.session_state.ACT_v_min:
                st.session_state.ACT_v_max = st.session_state.ACT_v_min + 0.01
            if st.session_state.ACT_dv is not None:
                st.session_state.ACT_dv = round(st.session_state.ACT_dv, 2)
                if (round(st.session_state.ACT_v_max, 2) - round(st.session_state.ACT_v_min, 2)) < round(st.session_state.ACT_dv, 2):
                    st.session_state.ACT_v_max = round(st.session_state.ACT_v_min, 2) + round(st.session_state.ACT_dv, 2)
                
    if st.session_state.ACT_v_min is not None:
        st.session_state.ACT_v_min = round(st.session_state.ACT_v_min, 2)
        if st.session_state.ACT_v_min <= 0:
            st.session_state.ACT_v_min = 0.01

    if st.session_state.ACT_dv is not None:
        st.session_state.ACT_dv = round(st.session_state.ACT_dv, 2)
        if st.session_state.ACT_dv <= 0:
            st.session_state.ACT_dv = 0.01
        if st.session_state.ACT_v_min is not None and st.session_state.ACT_v_max is not None:
            st.session_state.ACT_v_min = round(st.session_state.ACT_v_min, 2)
            st.session_state.ACT_v_max = round(st.session_state.ACT_v_max, 2)
            if (round(st.session_state.ACT_v_max, 2) - round(st.session_state.ACT_v_min, 2)) < round(st.session_state.ACT_dv, 2):
                st.session_state.ACT_dv = round(st.session_state.ACT_v_max, 2) - round(st.session_state.ACT_v_min, 2)
            
def set_nb_max_subproc():
    if st.session_state.ACT_nb_max_subproc is not None:
        st.session_state.ACT_nb_max_subproc = round(st.session_state.ACT_nb_max_subproc, 0)
        if st.session_state.ACT_nb_max_subproc < 1:
            st.session_state.ACT_nb_max_subproc = 1
        if st.session_state.ACT_nb_scripts is not None:
            st.session_state.ACT_nb_max_subproc = round(st.session_state.ACT_nb_max_subproc, 0)
            if st.session_state.ACT_nb_max_subproc > st.session_state.ACT_nb_scripts:
                st.session_state.ACT_nb_max_subproc = st.session_state.ACT_nb_scripts
        if st.session_state.ACT_nb_cores is not None:
            if st.session_state.ACT_nb_max_subproc > st.session_state.ACT_nb_cores:
                st.session_state.ACT_nb_max_subproc = st.session_state.ACT_nb_cores
                
def set_x_start():
    if st.session_state.ACT_x_start is not None:
        st.session_state.ACT_x_start = round(st.session_state.ACT_x_start, 3)
            
def set_x_step():
    if st.session_state.ACT_x_step is not None:
        st.session_state.ACT_x_step = round(st.session_state.ACT_x_step, 3)
        if st.session_state.ACT_x_step <= 0:
            st.session_state.ACT_x_step = 0.01
            
def set_MASW():
    if st.session_state.ACT_MASW_length is not None:
        st.session_state.ACT_MASW_length = round(st.session_state.ACT_MASW_length, 0)
        if st.session_state.ACT_MASW_length < 2:
            st.session_state.ACT_MASW_length = 2
        if st.session_state.ACT_MASW_length > st.session_state.ACT_N_traces:
            st.session_state.ACT_MASW_length = st.session_state.ACT_N_traces
    if st.session_state.ACT_MASW_distance_min is not None:
        st.session_state.ACT_MASW_distance_min = round(st.session_state.ACT_MASW_distance_min, 2)
        if st.session_state.ACT_MASW_distance_min < 0:
            st.session_state.ACT_MASW_distance_min = 0
    if st.session_state.ACT_MASW_distance_max is not None:
        st.session_state.ACT_MASW_distance_max = round(st.session_state.ACT_MASW_distance_max, 2)
        if st.session_state.ACT_MASW_distance_max < 0:
            st.session_state.ACT_MASW_distance_max = 0
    if st.session_state.ACT_MASW_distance_max is not None:
        st.session_state.ACT_MASW_distance_max = round(st.session_state.ACT_MASW_distance_max, 2)
        if st.session_state.ACT_MASW_distance_max < 0:
            st.session_state.ACT_MASW_distance_max = 0.01
    if st.session_state.ACT_MASW_distance_min is not None and st.session_state.ACT_MASW_distance_max is not None:
        if st.session_state.ACT_MASW_distance_max <= st.session_state.ACT_MASW_distance_min:
            st.session_state.ACT_MASW_distance_max = st.session_state.ACT_MASW_distance_min

    if st.session_state.ACT_MASW_step is not None:
        st.session_state.ACT_MASW_step = round(st.session_state.ACT_MASW_step, 0)
        if st.session_state.ACT_MASW_step < 0:
            st.session_state.ACT_MASW_step = 0
        if st.session_state.ACT_MASW_length is not None:
            st.session_state.ACT_MASW_length = round(st.session_state.ACT_MASW_length, 0)
            if st.session_state.ACT_MASW_step > st.session_state.ACT_MASW_length:
                st.session_state.ACT_MASW_step = st.session_state.ACT_MASW_length
            if st.session_state.ACT_MASW_length + st.session_state.ACT_MASW_step > st.session_state.ACT_N_traces:
                st.session_state.ACT_MASW_step = st.session_state.ACT_N_traces - st.session_state.ACT_MASW_length
### -------------------------------------------------------------------------------------------------------------------------------------------------



### START INTERFACE ---------------------------------------------------------------------------------------------------------------------------------
initialize_session()

st.set_page_config(
    layout="centered",
    page_title="Active computing",
    page_icon="👨‍💻",
    initial_sidebar_state="expanded",
)

st.title("👨‍💻 Active computing")
st.write("🛈 Surface wave dispersion active computing.")

st.divider() # --------------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Data selection")

st.text('')
st.text('')

# Folder selection
files_depth_1 = glob.glob(f"{input_dir}/*")
input_folders = filter(lambda f: os.path.isdir(f), files_depth_1)
input_folders = [os.path.relpath(folder, input_dir) for folder in input_folders]
input_folders = sorted(input_folders)
if input_folders:
    st.selectbox("**Data folder**", input_folders, key='ACT_selected_folder', on_change=handle_select_folder, index=None, placeholder='Select')
else:
    st.error("❌ No input data folders found.")
    clear_session()
    st.stop()

if st.session_state.ACT_folder_path is None:
    st.info("👆 Select a folder containing the raw seismic files to be processed. Ensure that the folders, each corresponding to a single profile, are placed inside PAC/input/. See [supported formats](https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html).")
    if st.session_state.ACT_x_start is not None or st.session_state.ACT_x_step is not None:
        st.session_state.ACT_x_start = None
        st.session_state.ACT_x_step = None
    st.stop()

if len(st.session_state.ACT_files) < 1:
    st.error("❌ Selected input data folder empty.")
    clear_session()
    st.stop()

st.success(f"👌 Seismic files loaded.")

st.text('')
st.text('')

st.markdown("**Summary:**")
data = {
    'Folder' : [st.session_state.ACT_folder_path],
    'Number of files [#]' : [len(st.session_state.ACT_files)],
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True, use_container_width=True)
data = {
    'Files' : [st.session_state.ACT_files],
    'Durations [s]' : [st.session_state.ACT_durations],
    'Source positions [m]' : [st.session_state.ACT_source_positions],
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True, use_container_width=True)

st.divider() # ----------------------------------------------------------------------------------------------------------------------------------

st.header("🚨 Sensor positions")

st.text('')
st.text('')

st.number_input("First sensor position [m]", key='ACT_x_start', value=None, step=0.01, placeholder='Enter a value', format="%0.3f", on_change=set_x_start)
st.number_input("Sensor spacing [m]", key='ACT_x_step', value=None, step=0.01, placeholder='Enter a value', format="%0.3f", on_change=set_x_step)

if st.session_state.ACT_x_start is None or st.session_state.ACT_x_step is None:
    st.text('')
    st.text('')
    st.info("👆 Define all sensor positions.") 
    st.stop()
    
st.session_state.ACT_positions = [round(st.session_state.ACT_x_start + i*st.session_state.ACT_x_step, 3) for i in range(st.session_state.ACT_N_traces)]

st.text('')
st.text('')
st.success("👌 Sensor positions defined.")

data = {
    'Number of sensors [#]' : [st.session_state.ACT_N_traces],
    'Sensors positions [m]' : [st.session_state.ACT_positions],
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True, use_container_width=True)

st.divider() # ----------------------------------------------------------------------------------------------------------------------------------

st.header("🚨 MASW parameters")

st.text('')
st.text('')

st.number_input('MASW window length [number of sensors]', key='ACT_MASW_length', value=None, step=1, placeholder='Enter a value', format="%i", on_change=set_MASW)
st.number_input('MASW window step [number of sensors]', key='ACT_MASW_step', value=None, step=1, placeholder='Enter a value', format="%i", on_change=set_MASW)
st.number_input('Minimum distance from source [m]', key='ACT_MASW_distance_min', value=None, step=1, placeholder='Enter a value', format="%i", on_change=set_MASW)
st.number_input('Maximum distance from source [m]', key='ACT_MASW_distance_max', value=None, step=1, placeholder='Enter a value', format="%i", on_change=set_MASW)

if st.session_state.ACT_MASW_length is None or st.session_state.ACT_MASW_step is None or st.session_state.ACT_MASW_distance_min is None or st.session_state.ACT_MASW_distance_max is None:
    st.text('')
    st.text('')
    st.info("👆 Define all MASW parameters.")
    st.stop()

fig_MASW, x_mids, windows_idx = plot_MASW(st.session_state.ACT_positions, st.session_state.ACT_MASW_length, st.session_state.ACT_MASW_step, 
                                           source_positions=st.session_state.ACT_source_positions)
st.session_state.ACT_x_mids = [round(x, 2) for x in x_mids]
st.session_state.ACT_nb_scripts = len(x_mids)
st.session_state.ACT_windows_idx = windows_idx
set_nb_max_subproc()

st.text('')
st.text('')

st.success(f"👌 {st.session_state.ACT_nb_scripts} MASW positions to compute.")

st.text('')
st.text('')
st.text('')
st.text('')
st.text('')

st.plotly_chart(fig_MASW, use_container_width=True)

st.divider() # ----------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Phase-shift parameters:")
st.text('')
st.text('')

st.number_input("Min frequency [Hz]", key='ACT_f_min', value=None, step=1.0, on_change=set_f, placeholder='Enter a value')
st.number_input("Max frequency [Hz]", key='ACT_f_max', value=None, step=1.0, on_change=set_f, placeholder='Enter a value')

st.text('')

st.number_input("Phase velocity step [m/s]", key='ACT_dv', value=None, step=0.1, on_change=set_v, placeholder='Enter a value')
st.number_input("Min phase velocity [m/s]", key='ACT_v_min', value=None, step=0.1, on_change=set_v, placeholder='Enter a value')
st.number_input("Max phase velocity [m/s]", key='ACT_v_max', value=None, step=0.1, on_change=set_v, placeholder='Enter a value')

if st.session_state.ACT_f_min is None or st.session_state.ACT_f_max is None or st.session_state.ACT_v_min is None or st.session_state.ACT_v_max is None or st.session_state.ACT_dv is None:
    st.text('')
    st.text('')
    st.info("👆 Define all phase-shift parameters.")
    st.stop()
    
st.text('')
st.text('')
st.success("👌 Phase-shift parameters defined.")

st.divider() # ----------------------------------------------------------------------------------------------------------------------------------
    
st.header("🚨 Parallelization parameters")

st.text('')
st.text('')
    
st.session_state.ACT_nb_cores = os.cpu_count()
st.number_input("Number of subprocesses [#]", key='ACT_nb_max_subproc', value=None, step=1, on_change=set_nb_max_subproc, placeholder='Enter a value', format="%i")
st.markdown(f"🛈 *Distributes {st.session_state.ACT_nb_scripts} MASW computations over a maximum of {st.session_state.ACT_nb_cores} available cores.*")

if st.session_state.ACT_nb_max_subproc is None:
    st.text('')
    st.text('')
    st.info("👆 Define the number of subprocesses.")
    st.stop()
    
st.text('')
st.text('')
st.success("👌 Parallelization parameters defined.")
    
st.divider() # ----------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Run computation")
st.text('')
st.text('')

if st.button("Compute", type="primary", use_container_width=True):
    st.text('')
    st.text('')
    loading_message = st.empty()
    loading_message.warning(f"⏳⚙️ Running computation...")
    
    profile = st.session_state.ACT_folder_path.split("/")[-2]
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    output_dir = f"{output_dir}/" + profile
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    output_dir = f"{output_dir}/" + f"Active/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir = output_dir + f"W{st.session_state.ACT_MASW_length}-{st.session_state.ACT_MASW_step}-D{st.session_state.ACT_MASW_distance_min}-{st.session_state.ACT_MASW_distance_max}/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    params = {
        "folder_path": st.session_state.ACT_folder_path,
        "files": st.session_state.ACT_files,
        "positions": st.session_state.ACT_positions,
        "durations": st.session_state.ACT_durations,
        "MASW_length": st.session_state.ACT_MASW_length,
        "MASW_step": st.session_state.ACT_MASW_step,
        "source_positions": st.session_state.ACT_source_positions,
        "distance_min": st.session_state.ACT_MASW_distance_min,
        "distance_max": st.session_state.ACT_MASW_distance_max,
        "x_mids": st.session_state.ACT_x_mids,
        "windows_idx": st.session_state.ACT_windows_idx,
        "f_min": st.session_state.ACT_f_min,
        "f_max": st.session_state.ACT_f_max,
        "v_min": st.session_state.ACT_v_min,
        "v_max": st.session_state.ACT_v_max,
        "dv": st.session_state.ACT_dv,
        "nb_cores": st.session_state.ACT_nb_max_subproc,
        "running_distribution": {}
    }
    for i, idx in enumerate(st.session_state.ACT_windows_idx):
        params["running_distribution"][str(i)] = {
            "start": idx[0],
            "end": idx[1],
            "x_mid" : st.session_state.ACT_x_mids[i],
        }
    with open(f"{output_dir}/computing_params.json", "w") as file:
        json.dump(params, file, indent=2)
    
    print(f"\033[1m\n\nRunning computation...\033[0m")
    start = time.time()
    
    scripts = [f"{work_dir}/scripts/run_active-MASW.py -ID {i} -r {output_dir}" for i in range(st.session_state.ACT_nb_scripts)]
    Executor = concurrent.futures.ThreadPoolExecutor
    with Executor(max_workers=st.session_state.ACT_nb_max_subproc) as executor:
        results = list(executor.map(run_script, scripts))
        
    nb_success = results.count(0)
    
    end = time.time()
    print(f"\033[1mComputation ended in {end - start:.2f} seconds.\033[0m")
        
    loading_message.empty()
    st.text('')
    st.text('')
    if nb_success == st.session_state.ACT_nb_scripts:
        st.success(f"👌 Computation completed for all {st.session_state.ACT_nb_scripts} MASW postions.")
    elif nb_success == 0:
        st.error(f"❌ Computation failed for all {st.session_state.ACT_nb_scripts} MASW postions.")
    else:
        st.warning(f"⚠️ Computation only completed for {nb_success} over {st.session_state.ACT_nb_scripts} MASW postions.")
    st.info(f"🕒 Computation took {end - start:.2f} seconds.")

else:
    st.text('')
    st.text('')
    st.info("👆 Click on the 'Compute' button to run the computations.")

st.divider() # --------------------------------------------------------------------------------------------------------------------------------------
### END INTERFACE------------------------------------------------------------------------------------------------------------------------------------
