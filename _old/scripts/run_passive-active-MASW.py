"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""

import sys
import argparse
import json
import numpy as np
from os import mkdir, path, system
from time import time
from obspy import read
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import correlate, filtfilt, iirnotch
from scipy.signal.windows import tukey
from math import isclose

sys.path.append("./modules/")
from misc import arange
from signal_processing import normalize, whiten, cut
from display import display_dispersion_img, display_spectrum_img_fromArray, display_seismic_wiggle_fromStream
from obspy2numpy import array_to_stream, stream_to_array
from dispersion import phase_shift

# Do not display warnings
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore")


tic = time()

### ARGUMENTS -------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Process an ID argument.")
parser.add_argument("-ID", type=int, required=True, help="ID of the script")
parser.add_argument("-r", type=str, required=True, help="Path to the folder containing the data")
args = parser.parse_args()
output_dir = args.r
ID = f"{int(args.ID)}"
### -----------------------------------------------------------------------------------------------




### READ-----------------------------------------------------------------------------
with open(f"{output_dir}/computing_params.json", "r") as file:
    params = json.load(file)

folder_path = params["folder_path"]
files = params["files"]

f_min = params["f_min"]
f_max = params["f_max"]
v_min = params["v_min"]
v_max = params["v_max"]
dv = params["dv"]
pws_nu = params["pws_nu"]

x_mid = np.round(params["running_distribution"][ID]["x_mid"], 6)
start = np.round(params["running_distribution"][ID]["start"], 6)
end = np.round(params["running_distribution"][ID]["end"], 6)
N_sensors = int(params["MASW_length"])
MASW_step = int(params["MASW_step"])
positions = np.round(np.array(params["positions"][start:end+1]), 6)
d_position = np.round(positions[1] - positions[0], 6)
source_positions = np.round(params["source_positions"], 6)
distance_min = np.round(params["distance_min"], 6)
distance_max = np.round(params["distance_max"], 6)
### -----------------------------------------------------------------------------------------------




### FK ratio results file -------------------------------------------------------------------------
output_dir = f"{output_dir}/xmid{x_mid}/"
if not path.exists(output_dir):
    mkdir(output_dir)

output_dir = f"{output_dir}/comp/"
if not path.exists(output_dir):
    mkdir(output_dir)

log_file = open(f"{output_dir}" + f"xmid{x_mid}_output.log", "w")
sys.stdout = log_file
sys.stderr = log_file
### -----------------------------------------------------------------------------------------------




### RUN -------------------------------------------------------------------------------------------
print(f'ID {ID} | x_mid {x_mid} | Running computation')
sys.stdout = sys.__stdout__
print(f'ID {ID} | x_mid {x_mid} | Running computation')
sys.stdout = log_file


VSGs = []

for i_segment, (file, source_position) in enumerate(zip(files, source_positions)):

    ### INITIALIZATION -----------------------------------------------------------------------
    # If source position is in the range of positions, skip
    if (source_position >= positions[0] and source_position <= positions[-1]):
        continue

    # Check if source at the left or right of the window
    if source_position < positions[0]:
        origin = positions[0]
        source_dir = "L"
    elif source_position > positions[-1]:
        origin = positions[-1]
        source_dir = "R"

    # If the source is too far from the positions, skip
    if abs(source_position - origin) > distance_max:
        continue

    # If the source position is too close to the positions, skip
    if abs(source_position - origin) < distance_min: 
        continue


    ### READ FILE ---------------------------------------------------------------------------------
    stream = read(folder_path + file)
    stream = stream[start:end+1]
    Nt = stream[0].stats.npts
    dt = stream[0].stats.delta
    f_ech = stream[0].stats.sampling_rate


    ### DEMAEN AND DETREND -----------------------------------------------------------------------
    stream.detrend('demean')
    stream.detrend("linear")


    ### ARRAY FORMAT ------------------------------------------------------------------------------
    TX_raw = stream_to_array(stream, len(stream), Nt)


    ### FILTERING (optional) ----------------------------------------------------------------------
    # Can help to remove some noise induced by the electrical frequency of the railway (50 Hz in France)
    # f0 = 50 # Frequency to remove
    # Q = 5  # Quality factor (higher = narrower notch)
    # f_nyq = f_ech / 2 # Nyquist frequency
    # while f0 <= f_nyq:
    #     b, a = iirnotch(f0, Q, f_ech)
    #     for i, trace in enumerate(TX_raw.T):
    #         TX_raw[:,i] = filtfilt(b, a, trace)
    #     f0 += 50


    ### WHITENING -----------------------------------------------------------------------------
    TX = whiten(TX_raw, f_ech, 0, f_max)


    ### INTERFEROMETRY ------------------------------------------------------------------------
    for virtual_source in [1, N_sensors]:
        t_s = TX.T[virtual_source-1]

        VSG = []

        for t_r in TX.T:
            correl = correlate(t_s, t_r)
            tmp = np.copy(correl)
            tmp_0 = tmp[int(len(correl)//2)]
            tmp_del_tmp_0 = np.delete(tmp, int(len(correl)//2))
            acausal, causal = np.hsplit(tmp_del_tmp_0, 2)
            acausal = np.flip(acausal)

            if source_dir == "R" :
                if virtual_source == 1:
                    trace = causal
                elif virtual_source == N_sensors:
                    trace = acausal

            elif source_dir == "L" :
                if virtual_source == 1:
                    trace = acausal
                elif virtual_source == N_sensors:
                    trace = causal

            trace = np.insert(trace, 0, tmp_0)
            trace = trace[0:int(TX_raw.shape[0]/dt)+1]

            for i, (correl_val, tukey_val)  in enumerate(zip(trace, tukey(len(trace)))):
                if i >= len(trace)//2:
                    trace[i] = correl_val * tukey_val

            VSG.append(trace)

        VSG = np.array(VSG)
        if virtual_source == N_sensors:
            VSG = np.flip(VSG, axis=0)

        VSGs.append(VSG)

VSGs = np.array(VSGs)


### STACK INTERFEROMETRY --------------------------------------------------------------------------
VSG = np.zeros((VSGs.shape[1], VSGs.shape[2]))

for i_r in range(VSGs.shape[1]):
    traces_to_stack = []
    for VSG in VSGs:
        traces_to_stack.append(VSG[i_r, :])

    traces_to_stack = np.array(traces_to_stack)
    stream = array_to_stream(traces_to_stack.T, dt, range(len(traces_to_stack)))
    stream = stream.stack(stack_type=("pw", pws_nu))
    VSG[i_r, :] = stream[0].data


# Stream format ---
st_interf = array_to_stream(VSG.T, dt, positions)
name_path = output_dir +  f"xmid{x_mid}_VSG.segy"
try:
    st_interf.write(name_path, format="SEGY", data_encoding=1, byteorder=sys.byteorder)
except:
    print(f"\033[93mID {ID} | x_mid {x_mid} | Warning : Unable to write the SEGY file. Maybe because obspy can't write traces with more than 32767 samples.\033[0m")

name_path = output_dir + f"xmid{x_mid}_VSG.svg"
display_seismic_wiggle_fromStream(st_interf, positions, path=name_path, norm_method="trace")

# Spectrum ---
name_path1 = output_dir + f"xmid{x_mid}_spectrogram.svg"
name_path2 = output_dir + f"xmid{x_mid}_spectrogramFirstLastTrace.svg"
display_spectrum_img_fromArray(VSG.T, dt, positions, path1=name_path1, path2=name_path2, norm_method="trace", f_min=f_min, f_max=f_max)


### SLANT STACK -------------------------------------------------------------------------------
offsets = np.abs(positions - positions[0])
(fs, vs, FV) = phase_shift(VSG, dt, offsets, v_min, v_max, dv, f_min, f_max)

name_path = output_dir + f"xmid{x_mid}_dispersion.svg"
display_dispersion_img(FV, fs, vs, path=name_path, normalization='Frequency', dx=positions[1]-positions[0])

name_path = output_dir + f"xmid{x_mid}_dispersion.csv"
np.savetxt(name_path, FV, delimiter=",")

name_path = output_dir + f"xmid{x_mid}_fs.csv"
np.savetxt(name_path, fs, delimiter=",")

name_path = output_dir + f"xmid{x_mid}_vs.csv"
np.savetxt(name_path, vs, delimiter=",")
### -----------------------------------------------------------------------------------------------



### END -------------------------------------------------------------------------------------------
toc = time()
print(f"ID {ID} | x_mid {x_mid} | Computation completed in {toc-tic:.1f} s")
sys.stdout = sys.__stdout__
print(f"\033[92mID {ID} | x_mid {x_mid} | Computation completed in {toc-tic:.1f} s\033[0m")
sys.stdout = log_file
### -----------------------------------------------------------------------------------------------
