"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""

import os
import glob
import time
import pandas as pd
import streamlit as st
from obspy import read
import subprocess
import concurrent.futures
import json
import plotly.graph_objects as go

from Paths import output_dir, input_dir

import warnings
warnings.filterwarnings("ignore")



### FUNCTIONS ---------------------------------------------------------------------------------------------------------------------------------------
def plot_MASW(geophone_positions, MASW_length_idx, MASW_step_idx):
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
    
    # Add scatter points for selected geophones in blue
    fig.add_trace(go.Scatter(
        x=geophone_positions,
        y=[5] * len(geophone_positions),
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='#1f77b4'),
        showlegend=True,
        name='Geophones',
    ))
    
    # Add scatter points for MASW windows middle positions in red
    fig.add_trace(go.Scatter(
        x=x_mids,
        y=[-5] * len(x_mids),
        mode='markers',
        marker=dict(symbol='star-triangle-down', size=10, color='#d62728'),
        showlegend=True,
        name='MASW positions',
    ))

    # Update layout to hide the y-axis
    fig.update_layout(
        title="Geophones and MASW positions",
        xaxis=dict(
            title="Position [m]",
            side="top",
            ticks="outside",
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis_range=[-10, 10],
        xaxis_range=[min(geophone_positions)-1, max(geophone_positions)+1],
        height=250,
    )

    return fig, x_mids, windows_idx

def run_script(script):
    command = ["python3"] + script.split()
    subprocess.run(command)
    
def clear_session():
    st.cache_data.clear()
    st.session_state.clear()

def initialize_session():
    for key in st.session_state:
        if 'COMP' not in key:
            st.session_state.pop(key)
    if 'COMP_folder_path' not in st.session_state:
        st.session_state.COMP_folder_path = None
    if 'COMP_x_start' not in st.session_state:
        st.session_state.COMP_x_start = None
    if 'COMP_x_step' not in st.session_state:
        st.session_state.COMP_x_step = None
### -------------------------------------------------------------------------------------------------------------------------------------------------



### HANDLERS ----------------------------------------------------------------------------------------------------------------------------------------
def handle_select_folder():
    selected_folder_tmp = st.session_state.COMP_selected_folder
    clear_session()
    initialize_session()
    st.session_state.COMP_selected_folder = selected_folder_tmp
    
    if st.session_state.COMP_selected_folder is not None:    
        st.session_state.COMP_folder_path = f"{input_dir}/{st.session_state.COMP_selected_folder}/"

        # List all files in the folder
        files = [file for file in os.listdir(st.session_state.COMP_folder_path)]
        files = sorted(files)
        st.session_state.COMP_files = files

        # Read all seismic records
        streams = []
        for file in st.session_state.COMP_files:
            stream = read(st.session_state.COMP_folder_path + file)
            streams.append(stream)
        st.session_state.COMP_streams = streams

        # Compute the duration of each seismic record
        st.session_state.COMP_durations = [stream[0].stats.endtime - stream[0].stats.starttime for stream in st.session_state.COMP_streams]
    

def handle_set():
    st.session_state.COMP_clicked_set = True

def set_f():
    if st.session_state.COMP_f_min is not None:
        if st.session_state.COMP_f_min < 0:
            st.session_state.COMP_f_min = 0
    if st.session_state.COMP_f_max is not None:
        if st.session_state.COMP_f_max < 0:
            st.session_state.COMP_f_max = 0
    if st.session_state.COMP_f_min is not None and st.session_state.COMP_f_max is not None:
        if st.session_state.COMP_f_max <= st.session_state.COMP_f_min:
            st.session_state.COMP_f_max = st.session_state.COMP_f_min + 1
                
def set_v():    
    if st.session_state.COMP_v_max is not None:
        st.session_state.COMP_v_max = round(st.session_state.COMP_v_max, 2)
        if st.session_state.COMP_v_max <= 0:
            st.session_state.COMP_v_max = 0.01
        if st.session_state.COMP_v_min is not None:
            st.session_state.COMP_v_min = round(st.session_state.COMP_v_min, 2)
            if st.session_state.COMP_v_max <= st.session_state.COMP_v_min:
                st.session_state.COMP_v_max = st.session_state.COMP_v_min + 0.01
            if st.session_state.COMP_dv is not None:
                st.session_state.COMP_dv = round(st.session_state.COMP_dv, 2)
                if (round(st.session_state.COMP_v_max, 2) - round(st.session_state.COMP_v_min, 2)) < round(st.session_state.COMP_dv, 2):
                    st.session_state.COMP_v_max = round(st.session_state.COMP_v_min, 2) + round(st.session_state.COMP_dv, 2)
                
    if st.session_state.COMP_v_min is not None:
        st.session_state.COMP_v_min = round(st.session_state.COMP_v_min, 2)
        if st.session_state.COMP_v_min <= 0:
            st.session_state.COMP_v_min = 0.01

    if st.session_state.COMP_dv is not None:
        st.session_state.COMP_dv = round(st.session_state.COMP_dv, 2)
        if st.session_state.COMP_dv <= 0:
            st.session_state.COMP_dv = 0.01
        if st.session_state.COMP_v_min is not None and st.session_state.COMP_v_max is not None:
            st.session_state.COMP_v_min = round(st.session_state.COMP_v_min, 2)
            st.session_state.COMP_v_max = round(st.session_state.COMP_v_max, 2)
            if (round(st.session_state.COMP_v_max, 2) - round(st.session_state.COMP_v_min, 2)) < round(st.session_state.COMP_dv, 2):
                st.session_state.COMP_dv = round(st.session_state.COMP_v_max, 2) - round(st.session_state.COMP_v_min, 2)
            
def set_segment():
    if st.session_state.COMP_segment_length is not None:
        st.session_state.COMP_segment_length = round(st.session_state.COMP_segment_length, 3)
        if st.session_state.COMP_segment_length <= 0:
            st.session_state.COMP_segment_length = 0.001
        if st.session_state.COMP_segment_length >= min(st.session_state.COMP_durations):
            st.session_state.COMP_segment_length = min(st.session_state.COMP_durations)
    if st.session_state.COMP_segment_step is not None:
        st.session_state.COMP_segment_step = round(st.session_state.COMP_segment_step, 3)
        if st.session_state.COMP_segment_step <= 0:
            st.session_state.COMP_segment_step = 0.001
        if st.session_state.COMP_segment_step >= min(st.session_state.COMP_durations):
            st.session_state.COMP_segment_step = min(st.session_state.COMP_durations)
    if st.session_state.COMP_segment_length is not None and st.session_state.COMP_segment_step is not None:
        if st.session_state.COMP_segment_step > st.session_state.COMP_segment_length:
            st.session_state.COMP_segment_step = st.session_state.COMP_segment_length
        if st.session_state.COMP_segment_length + st.session_state.COMP_segment_step > min(st.session_state.COMP_durations):
            st.session_state.COMP_segment_step = max(0, min(st.session_state.COMP_durations) - st.session_state.COMP_segment_length)
                
def set_FK_ratio_threshold():
    if st.session_state.COMP_FK_ratio_threshold is not None:
        st.session_state.COMP_FK_ratio_threshold = round(st.session_state.COMP_FK_ratio_threshold, 1)
        if st.session_state.COMP_FK_ratio_threshold < 0:
            st.session_state.COMP_FK_ratio_threshold = 0
        if st.session_state.COMP_FK_ratio_threshold > 1:
            st.session_state.COMP_FK_ratio_threshold = 1
            
def set_pws_nu():
    if st.session_state.COMP_pws_nu is not None:
        st.session_state.COMP_pws_nu = round(st.session_state.COMP_pws_nu, 0)
        if st.session_state.COMP_pws_nu < 0:
            st.session_state.COMP_pws_nu = 0
            
def set_nb_max_subproc():
    if st.session_state.COMP_nb_max_subproc is not None:
        st.session_state.COMP_nb_max_subproc = round(st.session_state.COMP_nb_max_subproc, 0)
        if st.session_state.COMP_nb_max_subproc < 1:
            st.session_state.COMP_nb_max_subproc = 1
        if st.session_state.COMP_nb_scripts is not None:
            st.session_state.COMP_nb_max_subproc = round(st.session_state.COMP_nb_max_subproc, 0)
            if st.session_state.COMP_nb_max_subproc > st.session_state.COMP_nb_scripts:
                st.session_state.COMP_nb_max_subproc = st.session_state.COMP_nb_scripts
        if st.session_state.COMP_nb_cores is not None:
            if st.session_state.COMP_nb_max_subproc > st.session_state.COMP_nb_cores:
                st.session_state.COMP_nb_max_subproc = st.session_state.COMP_nb_cores
                
def set_x_start():
    if st.session_state.COMP_x_start is not None:
        st.session_state.COMP_x_start = round(st.session_state.COMP_x_start, 3)
            
def set_x_step():
    if st.session_state.COMP_x_step is not None:
        st.session_state.COMP_x_step = round(st.session_state.COMP_x_step, 3)
        if st.session_state.COMP_x_step <= 0:
            st.session_state.COMP_x_step = 0.01
            
def set_MASW():
    if st.session_state.COMP_MASW_length is not None:
        st.session_state.COMP_MASW_length = round(st.session_state.COMP_MASW_length, 0)
        if st.session_state.COMP_MASW_length < 2:
            st.session_state.COMP_MASW_length = 2
        if st.session_state.COMP_MASW_length > len(st.session_state.COMP_streams[0]):
            st.session_state.COMP_MASW_length = len(st.session_state.COMP_streams[0])
            
    if st.session_state.COMP_MASW_step is not None:
        st.session_state.COMP_MASW_step = round(st.session_state.COMP_MASW_step, 0)
        if st.session_state.COMP_MASW_step < 0:
            st.session_state.COMP_MASW_step = 0
        if st.session_state.COMP_MASW_length is not None:
            st.session_state.COMP_MASW_length = round(st.session_state.COMP_MASW_length, 0)
            if st.session_state.COMP_MASW_step > st.session_state.COMP_MASW_length:
                st.session_state.COMP_MASW_step = st.session_state.COMP_MASW_length
            if st.session_state.COMP_MASW_length + st.session_state.COMP_MASW_step > len(st.session_state.COMP_streams[0]):
                st.session_state.COMP_MASW_step = len(st.session_state.COMP_streams[0]) - st.session_state.COMP_MASW_length
### -------------------------------------------------------------------------------------------------------------------------------------------------



### START INTERFACE ---------------------------------------------------------------------------------------------------------------------------------
initialize_session()

st.set_page_config(
    layout="centered",
    page_title="Computing",
    page_icon="👨‍💻",
    initial_sidebar_state="expanded",
)

st.title("👨‍💻 Computing")
st.write("🛈 Surface wave dispersion computing.")

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
    st.selectbox("**Data folder**", input_folders, key='COMP_selected_folder', on_change=handle_select_folder, index=None, placeholder='Select')
else:
    st.error("❌ No input data folders found.")
    st.stop()

if st.session_state.COMP_folder_path is None:
    st.info("👆 Select a folder containing the raw seismic files to be processed.")
    if st.session_state.COMP_x_start is not None or st.session_state.COMP_x_step is not None:
        st.session_state.COMP_x_start = None
        st.session_state.COMP_x_step = None
    st.stop()

st.success(f"👌 Seismic files loaded.")

st.text('')
st.text('')

st.markdown("**Summary:**")
data = {
    'Folder' : [st.session_state.COMP_folder_path],
    'Number of files [#]' : [len(st.session_state.COMP_files)],
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True, use_container_width=True)
data = {
    'Files' : [st.session_state.COMP_files],
    'Durations [s]' : [st.session_state.COMP_durations],
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True, use_container_width=True)

st.divider() # ----------------------------------------------------------------------------------------------------------------------------------

st.header("🚨 Geophone positions")

st.text('')
st.text('')

st.number_input("First geophone position [m]", key='COMP_x_start', value=None, step=0.01, placeholder='Enter a value', format="%0.3f", on_change=set_x_start)
st.number_input("Geophone spacing [m]", key='COMP_x_step', value=None, step=0.01, placeholder='Enter a value', format="%0.3f", on_change=set_x_step)

if st.session_state.COMP_x_start is None or st.session_state.COMP_x_step is None:
    st.text('')
    st.text('')
    st.info("👆 Define all geophone positions.") 
    st.stop()
    
st.session_state.COMP_positions = [round(st.session_state.COMP_x_start + i*st.session_state.COMP_x_step, 3) for i in range(len(st.session_state.COMP_streams[0]))]
   
st.text('')
st.text('')
st.success("👌 Geophone positions defined.")

data = {
    'Number of geophones [#]' : [len(st.session_state.COMP_streams[0])],
    'Geophone positions [m]' : [st.session_state.COMP_positions],
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True, use_container_width=True)

st.divider() # ----------------------------------------------------------------------------------------------------------------------------------

st.header("🚨 MASW parameters")

st.text('')
st.text('')

st.number_input('MASW window length [number of geophones]', key='COMP_MASW_length', value=None, step=1, placeholder='Enter a value', format="%i", on_change=set_MASW)
st.number_input('MASW window step [number of geophones]', key='COMP_MASW_step', value=None, step=1, placeholder='Enter a value', format="%i", on_change=set_MASW)

if st.session_state.COMP_MASW_length is None or st.session_state.COMP_MASW_step is None:
    st.text('')
    st.text('')
    st.info("👆 Define all MASW parameters.")
    st.stop()

fig_MASW, x_mids, windows_idx = plot_MASW(st.session_state.COMP_positions, st.session_state.COMP_MASW_length, st.session_state.COMP_MASW_step)
st.session_state.COMP_x_mids = [round(x, 2) for x in x_mids]
st.session_state.COMP_nb_scripts = len(x_mids)
st.session_state.COMP_windows_idx = windows_idx

st.text('')
st.text('')

st.success(f"👌 {st.session_state.COMP_nb_scripts} MASW positions to compute.")

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

st.number_input("Min frequency [Hz]", key='COMP_f_min', value=None, step=1.0, on_change=set_f, placeholder='Enter a value')
st.number_input("Max frequency [Hz]", key='COMP_f_max', value=None, step=1.0, on_change=set_f, placeholder='Enter a value')

st.text('')

st.number_input("Phase velocity step [m/s]", key='COMP_dv', value=None, step=0.1, on_change=set_v, placeholder='Enter a value')
st.number_input("Min phase velocity [m/s]", key='COMP_v_min', value=None, step=0.1, on_change=set_v, placeholder='Enter a value')
st.number_input("Max phase velocity [m/s]", key='COMP_v_max', value=None, step=0.1, on_change=set_v, placeholder='Enter a value')

if st.session_state.COMP_f_min is None or st.session_state.COMP_f_max is None or st.session_state.COMP_v_min is None or st.session_state.COMP_v_max is None or st.session_state.COMP_dv is None:
    st.text('')
    st.text('')
    st.info("👆 Define all phase-shift parameters.")
    st.stop()
    
st.text('')
st.text('')
st.success("👌 Phase-shift parameters defined.")

st.divider() # ----------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Segment parameters")
st.text('')
st.text('')

st.number_input('Segment window length [s]', key='COMP_segment_length', value=None, step=0.001, on_change=set_segment, placeholder='Enter a value', format="%0.3f")
st.number_input('Segment window step [s]', key='COMP_segment_step', value=None, step=0.001, on_change=set_segment, placeholder='Enter a value', format="%0.3f")

st.text('')

st.number_input("Minimum FK ration threshold [-]", key="COMP_FK_ratio_threshold", value=None, step=0.1, on_change=set_FK_ratio_threshold, placeholder='Enter a value', format="%0.1f")

if st.session_state.COMP_segment_length is None or st.session_state.COMP_segment_step is None or st.session_state.COMP_FK_ratio_threshold is None:
    st.text('')
    st.text('')
    st.info("👆 Define all segment parameters.")
    st.stop()

st.text('')
st.text('')
st.success("👌 Segment parameters defined.")

st.divider() # ----------------------------------------------------------------------------------------------------------------------------------
st.header("🚨 Stacking parameters")
st.text('')
st.text('')

st.number_input("Phase weighted stack order [-]", key='COMP_pws_nu', value=None, step=1, on_change=set_pws_nu, placeholder='Enter a value', format="%i")
st.markdown("🛈 *Order 0 corresponds to a linear stack.*")

if st.session_state.COMP_pws_nu is None:
    st.text('')
    st.text('')
    st.info("👆 Define the stacking order.")
    st.stop()
    
st.text('')
st.text('')

st.success("👌 Stacking parameters defined.")

st.divider() # ----------------------------------------------------------------------------------------------------------------------------------
    
st.header("🚨 Parallelization parameters")

st.text('')
st.text('')
    
st.session_state.COMP_nb_cores = os.cpu_count()
st.number_input("Number of subprocesses [#]", key='COMP_nb_max_subproc', value=None, step=1, on_change=set_nb_max_subproc, placeholder='Enter a value', format="%i")
st.markdown(f"🛈 *Distributes {st.session_state.COMP_nb_scripts} MASW computations over a maximum of {st.session_state.COMP_nb_cores} available cores.*")

if st.session_state.COMP_nb_max_subproc is None:
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
    
    profile = st.session_state.COMP_folder_path.split("/")[-2]
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    output_dir = f"{output_dir}/" + profile
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    output_dir = f"{output_dir}/" + f"Sl{st.session_state.COMP_segment_length:.3f}-Ss{st.session_state.COMP_segment_step:.3f}-FK{st.session_state.COMP_FK_ratio_threshold:.2f}-PWS{st.session_state.COMP_pws_nu:.1f}/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir = output_dir + f"W{st.session_state.COMP_MASW_length}-{st.session_state.COMP_MASW_step}/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    params = {
        "folder_path": st.session_state.COMP_folder_path,
        "files": st.session_state.COMP_files,
        "positions": st.session_state.COMP_positions,
        "durations": st.session_state.COMP_durations,
        "MASW_length": st.session_state.COMP_MASW_length,
        "MASW_step": st.session_state.COMP_MASW_step,
        "x_mids": st.session_state.COMP_x_mids,
        "windows_idx": st.session_state.COMP_windows_idx,
        "f_min": st.session_state.COMP_f_min,
        "f_max": st.session_state.COMP_f_max,
        "v_min": st.session_state.COMP_v_min,
        "v_max": st.session_state.COMP_v_max,
        "dv": st.session_state.COMP_dv,
        "segment_length": st.session_state.COMP_segment_length,
        "segment_step": st.session_state.COMP_segment_step,
        "FK_ratio_threshold": st.session_state.COMP_FK_ratio_threshold,
        "pws_nu": st.session_state.COMP_pws_nu,
        "nb_cores": st.session_state.COMP_nb_max_subproc,
        "running_distribution": {}
    }
    for i, idx in enumerate(st.session_state.COMP_windows_idx):
        params["running_distribution"][str(i)] = {
            "start": idx[0],
            "end": idx[1],
            "x_mid" : st.session_state.COMP_x_mids[i],
        }
    with open(f"{output_dir}/computing_params.json", "w") as file:
        json.dump(params, file, indent=2)
    
    print(f"\033[1m\n\nRunning computation...\033[0m")
    start = time.time()
    
    scripts = [f"/home/jteixeira/Desktop/passive-MASW/src/scripts/run_passive-MASW.py -ID {i} -r {output_dir}" for i in range(st.session_state.COMP_nb_scripts)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=st.session_state.COMP_nb_max_subproc) as executor:
        executor.map(run_script, scripts)
    
    end = time.time()
    print(f"\033[1mComputation completed in {end - start:.2f} seconds.\033[0m")
        
    loading_message.empty()
    st.text('')
    st.text('')
    st.success(f"👌 Computation completed for {st.session_state.COMP_nb_scripts} MASW postions.")
    st.info(f"🕒 Computation took {end - start:.2f} seconds.")

else:
    st.text('')
    st.text('')
    st.info("👆 Click on the 'Compute' button to run the computations.")

st.divider() # --------------------------------------------------------------------------------------------------------------------------------------
### END INTERFACE------------------------------------------------------------------------------------------------------------------------------------