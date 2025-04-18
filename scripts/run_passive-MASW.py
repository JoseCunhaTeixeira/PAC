"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""

import os
import sys
import argparse
import json
import numpy as np
from os import mkdir, path, system
from time import time
from obspy import read
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import butter, correlate, filtfilt, iirnotch
from scipy.signal.windows import tukey
from math import isclose

sys.path.append("./modules/")
from misc import arange
from signal_processing import normalize, whiten, cut
from display import display_dispersion_img, display_spectrum_img_fromArray, display_seismic_wiggle_fromStream
from obspy2numpy import array_to_stream, stream_to_array
from dispersion import phase_shift

import matplotlib.pyplot as plt

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
ID = f"{int(args.ID)}"
output_dir = f"{args.r}"
### -----------------------------------------------------------------------------------------------




### READ JSON -------------------------------------------------------------------------------------
with open(f"{output_dir}/computing_params.json", "r") as file:
    params = json.load(file)

folder_path = params["folder_path"]
profile = params["folder_path"].split("/")[-2]

f_min = params["f_min"]
f_max = params["f_max"]
v_min = params["v_min"]
v_max = params["v_max"]
dv = params["dv"]

segment_length = np.round(params["segment_length"], 6)
segment_step = np.round(params["segment_step"], 6)
FK_ratio_threshold = np.round(params["FK_ratio_threshold"], 6)
pws_nu = int(params["pws_nu"])

x_mid = np.round(params["running_distribution"][ID]["x_mid"], 6)
start = np.round(params["running_distribution"][ID]["start"], 6)
end = np.round(params["running_distribution"][ID]["end"], 6)
N_sensors = int(params["MASW_length"])
MASW_step = int(params["MASW_step"])
positions = np.round(np.array(params["positions"][start:end+1]), 6)
### -----------------------------------------------------------------------------------------------




### READ FILES ------------------------------------------------------------------------------------
files = [file for file in os.listdir(folder_path) if not file.startswith(".")]
files = sorted(files)

streams = [read(folder_path + file) for file in files]
streams = [stream[start:end+1] for stream in streams]

durations = np.round([stream[0].stats.endtime - stream[0].stats.starttime for stream in streams], 6)

dt = streams[0][0].stats.delta
### -----------------------------------------------------------------------------------------------




### INITIALISATION --------------------------------------------------------------------------------
virtual_sources = [1, N_sensors]
N_segments = 0
for file, duration in zip(files, durations) :
    if segment_length < duration:
        cuts = list(arange(0, duration-segment_length, segment_step))
    else:
        cuts = [0]
    N_segments += len(cuts)

interf_db = np.zeros((len(virtual_sources), N_sensors, N_segments, int(segment_length/dt)+1))
interf_db_stack = np.zeros((N_sensors, int(segment_length/dt)+1))
FK_ratios = []
### -----------------------------------------------------------------------------------------------




### FK ratio results file -------------------------------------------------------------------------
output_dir = f"{output_dir}/xmid{x_mid}/"
if not path.exists(output_dir):
    mkdir(output_dir)

output_dir = f"{output_dir}/comp/"
if not path.exists(output_dir):
    mkdir(output_dir)

file_path = output_dir + f"xmid{x_mid}_FK_ratios.log"
if path.exists(file_path):
    system(f"rm {file_path}")
file_FK_ratios = open(file_path, "a")

log_file = open(f"{output_dir}" + f"xmid{x_mid}_output.log", "w")
sys.stdout = log_file
sys.stderr = log_file
### -----------------------------------------------------------------------------------------------




### RUN -------------------------------------------------------------------------------------------
print(f'ID {ID} | x_mid {x_mid} | Running computation')
sys.stdout = sys.__stdout__
print(f'ID {ID} | x_mid {x_mid} | Running computation')
sys.stdout = log_file

i_segment = 0
to_del = []
for file, stream, duration in zip(files, streams, durations):


    ### DEMAEN AND DETREND ------------------------------------------------------------------------
    stream.detrend('demean')
    stream.detrend("linear")


    ### DECIMATE ---------------------------------------------------------------------------------
    if not isclose(stream[0].stats.delta, dt, rel_tol=1e-9):
        ratio = int(dt/ stream[0].stats.delta)
        stream.decimate(ratio)
    Nt = stream[0].stats.npts
    dt = stream[0].stats.delta
    f_ech = stream[0].stats.sampling_rate
    ts_raw = arange(0, Nt*dt-dt, dt)


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


    ### LOOP ON SEGMENTS --------------------------------------------------------------------------
    cut_start = 0
    while cut_start < duration - segment_length or isclose(cut_start, duration - segment_length, rel_tol=1e-6):
        ### CUT SEGMENT ---------------------------------------------------------------------------
        TX, _ = cut(TX_raw, ts_raw, cut_start, cut_start+segment_length)

        ### FK SELECTION --------------------------------------------------------------------------
        # Zero padding for better resolution if length is less than 2000 samples [can improve results but takes more time to compute]
        # TX_tmp = np.copy(TX)
        # nb_zeros_for_f = max(0, 2000-TX_tmp.shape[0])
        # nb_zeros_for_k = max(0, 2000-TX_tmp.shape[1])
        # TX_tmp = np.pad(TX_tmp, ((0,nb_zeros_for_f), (0,nb_zeros_for_k)), mode='constant', constant_values=0)
        # If not needed, comment the lines above and uncomment the line below
        TX_tmp = np.copy(TX)

        # Compute the FK diagram
        fs = rfftfreq(TX_tmp.shape[0], dt)
        try :
            fimax = np.where(fs >= f_max)[0][0]
        except :
            fimax = len(fs)-1
        try :
            fimin = np.where(fs >= f_min)[0][0]
        except :
            fimin = 0
        fs = fs[fimin:fimax+1]

        ks = fftfreq(TX_tmp.shape[1], positions[1]-positions[0])

        FX = rfft(TX_tmp, axis=0)
        FX = FX[fimin:fimax+1,:]
        n_k = FX.shape[1]

        FK = fft(FX, axis=1)

        FK = np.abs(FK)

        # Separate positive and negative wavenumbers
        if n_k%2 == 0:
            K_pos = FK[:, 1:int(n_k/2)]
            ks_pos = ks[1:int(n_k/2)]

            K_neg = FK[:, int(n_k/2)+1:]
            ks_neg = ks[int(n_k/2)+1:]
        else:
            K_pos = FK[:, 1:int(n_k/2)+1]
            ks_pos = ks[1:int(n_k/2)+1]

            K_neg = FK[:, int(n_k/2)+1:]
            ks_neg = ks[int(n_k/2)+1:]

        # Limit the FK diagrams to the velocity range as in Cheng et al. (2018) [Users should avoid wavenumber=frequency/velocity > 1/(2*dx) by limiting frequencies in accordance to the velocities to study]
        fs_min_lim_neg = v_min * abs(ks_neg)
        fs_max_lim_neg = v_max * abs(ks_neg)
        for i_k in range(len(ks_neg)):
            if fs_max_lim_neg[i_k] >= fs[0] <= fs_max_lim_neg[i_k] <= fs[-1]:
                i_flim = np.where(fs >= fs_max_lim_neg[i_k])[0][0]
                K_neg[i_flim:, i_k] = 0
            if fs_min_lim_neg[i_k] >= fs[0] and fs_min_lim_neg[i_k] <= fs[-1]:
                i_flim = np.where(fs >= fs_min_lim_neg[i_k])[0][0]
                K_neg[:i_flim+1, i_k] = 0

        fs_min_lim_pos = v_min * abs(ks_pos)
        fs_max_lim_pos = v_max * abs(ks_pos)
        for i_k in range(len(ks_pos)):
            if fs_max_lim_pos[i_k] >= fs[0] <= fs_max_lim_pos[i_k] <= fs[-1]:
                i_flim = np.where(fs >= fs_max_lim_pos[i_k])[0][0]
                K_pos[i_flim:, i_k] = 0
            if fs_min_lim_pos[i_k] >= fs[0] and fs_min_lim_pos[i_k] <= fs[-1]:
                i_flim = np.where(fs >= fs_min_lim_pos[i_k])[0][0]
                K_pos[:i_flim+1, i_k] = 0

        # Compute the FK ratio
        FK_ratio = 0
        source_position = None

        if np.sum(K_pos) > np.sum(K_neg):
            FK_ratio = np.sum(K_pos)/np.sum(K_neg) - 1
            if FK_ratio > FK_ratio_threshold :
                source_position = "R"
                FK_ratios.append([i_segment, abs(FK_ratio)])

        elif np.sum(K_neg) > np.sum(K_pos):
            FK_ratio = - np.sum(K_neg)/np.sum(K_pos) + 1
            if FK_ratio < -FK_ratio_threshold :
                source_position = "L"
                FK_ratios.append([i_segment, abs(FK_ratio)])

        # Uncomment to plot FK diagrams
        # import matplotlib.pyplot as plt
        # plt.rcParams.update({'font.size': 14})
        # CM = 1/2.54
        # vmin = np.min(FK)
        # vmax = np.max(FK)
        # fig, axs = plt.subplots(1, 2, figsize=(18*CM,18*CM), dpi=300)
        # axs[0].pcolormesh(ks_neg, fs, K_neg, vmin=vmin, vmax=vmax, cmap="gray_r")
        # axs[0].plot(ks_neg, fs_min_lim_neg, color='red', linestyle='--')
        # axs[0].plot(ks_neg, fs_max_lim_neg, color='red', linestyle='--')
        # axs[0].set_ylim(0, fs[-1])
        # axs[0].set_xlabel(r"Wavenumber [$m^{-1}$]")
        # axs[0].set_ylabel("Frequency [Hz]")
        # axs[1].pcolormesh(ks_pos, fs, K_pos, vmin=vmin, vmax=vmax, cmap="gray_r")
        # axs[1].plot(ks_pos, fs_min_lim_pos, color='red', linestyle='--')
        # axs[1].plot(ks_pos, fs_max_lim_pos, color='red', linestyle='--')
        # axs[1].set_ylim(fs[0], fs[-1])
        # axs[1].set_xlabel(r"Wavenumber [$m^{-1}$]")
        # axs[1].set_ylabel("Frequency [Hz]")
        # axs[1].yaxis.tick_right()
        # axs[1].yaxis.set_label_position("right")
        # fig.suptitle(f"FK ratio : {FK_ratio:.2f}")
        # name_path = output_dir + f"FK_file{file[:-4]}_cut{cut_start:.2f}.png"
        # fig.savefig(name_path)
        # plt.tight_layout()
        # plt.close()

        file_FK_ratios.write(f"{file} - {cut_start:.2f} - {source_position} - {FK_ratio}\n")

        if source_position == "L" or source_position == "R" :

            ### TEMPORAL NORMALIZATION (optional) -----------------------------------------------------
            # Not useful when using trains as sources
            # TX = normalize(TX, dt, clip_factor=1.0, norm_method="clipping")
            # TX = normalize(TX, dt, norm_win=0.01, norm_method="ramn")


            ### WHITENING -----------------------------------------------------------------------------
            TX = whiten(TX, f_ech, 0, f_max)

            ### INTERFEROMETRY ------------------------------------------------------------------------
            for i_s, virtual_source in enumerate(virtual_sources) :
                t_s = TX.T[virtual_source-1]

                for i_r, t_r in enumerate(TX.T):
                    correl = correlate(t_s, t_r)
                    tmp = np.copy(correl)
                    tmp_0 = tmp[int(len(correl)//2)]
                    tmp_del_tmp_0 = np.delete(tmp, int(len(correl)//2))
                    acausal, causal = np.hsplit(tmp_del_tmp_0, 2)
                    acausal = np.flip(acausal)

                    if source_position == "R" :
                        if virtual_source == 1:
                            correl_sym = causal
                        elif virtual_source == N_sensors:
                            correl_sym = acausal

                    elif source_position == "L" :
                        if virtual_source == 1:
                            correl_sym = acausal
                        elif virtual_source == N_sensors:
                            correl_sym = causal

                    correl_sym = np.insert(correl_sym, 0, tmp_0)
                    correl_sym = correl_sym[0:int(segment_length/dt)+1]

                    for i, (correl_val, tukey_val)  in enumerate(zip(correl_sym, tukey(len(correl_sym)))):
                        if i >= len(correl_sym)//2:
                            correl_sym[i] = correl_val * tukey_val

                    interf_db[i_s, i_r, i_segment, :] = correl_sym

        else :
            to_del.append(i_segment)

        i_segment += 1

        if segment_step > 0:
            cut_start = cut_start + segment_step
        else:
            break




### STACK INTERFEROMETRY --------------------------------------------------------------------------
FK_ratios = np.array(FK_ratios)
if FK_ratios.size > 0:
    FK_ratios = FK_ratios[FK_ratios[:, 1].argsort()]
else:
    print(f"\033[91mID {ID} | x_mid {x_mid} | No segment with FK ratio above {FK_ratio_threshold}\033[0m")
    raise Exception("No segment selected")

interf_db = np.delete(interf_db, to_del, 2)

for i_r in range(N_sensors) :
    if len(virtual_sources) == 2 :
        arr1 = np.copy(interf_db[0, i_r, :, :])
        arr2 = np.copy(interf_db[1, N_sensors-1-i_r, :, :])
        arr = np.vstack((arr1, arr2))

    elif len(virtual_sources) == 1:
        if virtual_sources[0] == 1:
            arr = np.copy(interf_db[0, i_r, :, :])
        elif virtual_sources[0] == N_sensors:
            arr = np.copy(interf_db[0, N_sensors-1-i_r, :, :])

    index = []
    for i, tr in enumerate(arr):
        if not np.any(tr):
            index.append(i)
    arr = np.delete(arr, index, 0)

    # PWS
    st = array_to_stream(arr.T, dt, range(arr.shape[0]))
    st = st.stack(stack_type=("pw", pws_nu))
    interf_db_stack[i_r, 0:int(segment_length/dt)+1] = st[0].data

# Stream format ---
offsets = np.abs(positions - positions[0])
st_interf = array_to_stream(interf_db_stack.T, dt, offsets)
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
display_spectrum_img_fromArray(interf_db_stack.T, dt, positions, path1=name_path1, path2=name_path2, norm_method="trace", f_min=f_min, f_max=f_max)
### -----------------------------------------------------------------------------------------------




### SLANT STACK -----------------------------------------------------------------------------------
arr = np.copy(interf_db_stack)

offsets = np.abs(positions - positions[0])
(fs, vs, FV) = phase_shift(arr, dt, offsets, v_min, v_max, dv, f_max)
# Uncomment to use the raw seismic data instead of the result of interferometry
# (fs, vs, FV) = phase_shift(TX_raw.T, dt, offsets, v_min, v_max, dv, f_max)

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
