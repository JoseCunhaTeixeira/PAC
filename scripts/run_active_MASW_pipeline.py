import argparse
import json
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
from sigproc.base import Acquisition, Coordinate
from sigproc.transformers import Detrend, Dispersion, Load, Plot, Save, Stack

warnings.filterwarnings("error")
warnings.filterwarnings("ignore")


tic = time()


### ARGUMENTS -------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Process an ID argument.")
parser.add_argument("-ID", type=int, required=True, help="ID of the script")
parser.add_argument(
    "-r", type=str, required=True, help="Path to the folder containing the data"
)
args = parser.parse_args()
output_dir = Path(args.r)
ID = f"{int(args.ID)}"
### -----------------------------------------------------------------------------------------------


### READ PARAMS -----------------------------------------------------------------------------------
with open(output_dir / "computing_params.json", "r") as file:
    params = json.load(file)

    input_dir = Path(params["folder_path"])
    files = params["files"]

    fmin = params["f_min"]
    fmax = params["f_max"]
    vmin = params["v_min"]
    vmax = params["v_max"]
    dv = params["dv"]

    xmid = np.round(params["running_distribution"][ID]["x_mid"], 6)
    start = np.round(params["running_distribution"][ID]["start"], 6)
    end = np.round(params["running_distribution"][ID]["end"], 6)
    N_sensors = int(params["MASW_length"])
    MASW_step = int(params["MASW_step"])
    positions = np.round(np.array(params["positions"][start : end + 1]), 6)
    d_position = np.round(positions[1] - positions[0], 6)
    source_positions = np.round(params["source_positions"], 6)
    distance_min = np.round(params["distance_min"], 6)
    distance_max = np.round(params["distance_max"], 6)
### -----------------------------------------------------------------------------------------------


### OUTPUT FOLDER ---------------------------------------------------------------------------------
output_dir = output_dir / f"xmid{xmid}" / "comp"
output_dir.mkdir(parents=True, exist_ok=True)
log_file = (output_dir / f"xmid{xmid}_output.log").open("w")
sys.stdout = log_file
sys.stderr = log_file
### -----------------------------------------------------------------------------------------------


### FILTER FILES ----------------------------------------------------------------------------------
file_paths = []
acquisitions = []
receivers = tuple(Coordinate(x=position, y=0, z=0) for position in positions)
for file, source_position in zip(files, source_positions, strict=True):
    if positions[0] <= source_position <= positions[-1]:
        continue

    origin = positions[0] if source_position < positions[0] else positions[-1]
    distance = abs(source_position - origin)

    if distance_min <= distance <= distance_max:
        file_paths.append(input_dir / file)
        source = Coordinate(x=source_position, y=0, z=0)
        acquisitions.append(Acquisition(source=source, receivers=receivers))
### -----------------------------------------------------------------------------------------------


### RUN -------------------------------------------------------------------------------------------
print(f"ID {ID} | xmid {xmid} | Running computation")
sys.stdout = sys.__stdout__
print(f"ID {ID} | xmid {xmid} | Running computation")
sys.stdout = log_file

pipeline = (
    Load(
        file_paths=file_paths,
        data_type="segy",
        acquisitions=acquisitions,
        receivers_to_load=range(start, end + 1),
    )
    >> Detrend(method="constant")
    >> Detrend(method="linear")
    >> Dispersion(method="phase", fmin=fmin, fmax=fmax, vmin=vmin, vmax=vmax)
    >> Stack(method="linear")
    >> Save(folder_path=output_dir, file_name=f"xmid{xmid}_dispersion")
    >> Plot(folder_path=output_dir, file_name=f"xmid{xmid}_dispersion")
)

pipeline.run()
### -----------------------------------------------------------------------------------------------


### END -------------------------------------------------------------------------------------------
toc = time()
print(f"ID {ID} | x_mid {xmid} | Computation completed in {toc - tic:.1f} s")
sys.stdout = sys.__stdout__
print(
    f"\033[92mID {ID} | x_mid {xmid} | Computation completed in {toc - tic:.1f} s\033[0m"
)
sys.stdout = log_file
### -----------------------------------------------------------------------------------------------
