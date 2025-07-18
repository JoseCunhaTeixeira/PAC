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
from os import mkdir, path
from time import time
from obspy import read
from scipy.signal import filtfilt, iirnotch

sys.path.append("./modules/")
from display import display_dispersion_img
from obspy2numpy import stream_to_array
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


for file, source_position in zip(files, source_positions):

    ### INITIALIZATION -----------------------------------------------------------------------
    # If source position is in the range of positions, skip
    if (source_position >= positions[0] and source_position <= positions[-1]):
        continue

    # Check if source at the left or right of the window
    if source_position < positions[0]:
        origin = positions[0]
    elif source_position > positions[-1]:
        origin = positions[-1]

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


    ### SLANT STACK -------------------------------------------------------------------------------
    offsets = np.abs(positions - source_position)
    (fs, vs, FV) = phase_shift(TX_raw.T, dt, offsets, v_min, v_max, dv, f_min, f_max)
    if 'FV_stacked' not in locals():
        FV_stacked = FV
    else:
        FV_stacked += FV
    ### -------------------------------------------------------------------------------------------




### SAVE RESULTS ----------------------------------------------------------------------------------
if 'FV_stacked' not in locals():
    print(f"ID {ID} | x_mid {x_mid} | No valid data found for this x_mid")
    sys.stdout = sys.__stdout__
    print(f"\033[93mID {ID} | x_mid {x_mid} | No valid data found for this x_mid\033[0m")
    sys.stdout = log_file
    log_file.close()
    sys.exit()

name_path = output_dir + f"xmid{x_mid}_dispersion.svg"
display_dispersion_img(FV_stacked, fs, vs, path=name_path, normalization='Frequency', dx=positions[1]-positions[0])

name_path = output_dir + f"xmid{x_mid}_dispersion.csv"
np.savetxt(name_path, FV_stacked, delimiter=",")

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
