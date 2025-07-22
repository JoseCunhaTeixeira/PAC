"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""

import os
import sys
import json
import argparse
import numpy as np
from time import time
from disba import PhaseDispersion
import bayesbay as bb
from bayesbay import State
from bayesbay._state import ParameterSpaceState
from bayesbay.likelihood import LogLikelihood
from scipy.interpolate import interp1d
from disba import DispersionError

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.colors as colors

sys.path.append("./modules/")
from misc import arange
from display import display_dispersion_img



FONT_SIZE = 9
plt.rcParams.update({'font.size': FONT_SIZE})
CM = 1/2.54

VP_VS = 1.77

tic = time()



### FUNCTIONS -------------------------------------------------------------------------------------
class CustomParametrization(bb.parameterization.Parameterization):
    def __init__(self, param_space, modes, fs_per_mode):
        super().__init__(param_space)
        self.modes = modes
        self.fs_per_mode = fs_per_mode
    
    def initialize(self):
        param_values = dict()
        for ps_name, ps in self.parameter_spaces.items():
            param_values[ps_name] = self.initialize_param_space(ps)
        return State(param_values)

    def initialize_param_space(self, param_space):
        unstable = True
        while unstable:
            vs_vals = []
            thick_vals = []
            for name, param in param_space.parameters.items():    
                vmin, vmax = param.get_vmin_vmax(None)
                if 'vs' in name:
                    vs_vals.append(np.random.uniform(vmin, vmax))
                elif 'thick' in name:
                    thick_vals.append(np.random.uniform(vmin, vmax))
            vs_vals = np.sort(vs_vals)
            thick_vals = np.sort(thick_vals)
            vp_vals = vs_vals * VP_VS
            rho_vals = 0.32 * vp_vals + 0.77*1000
            velocity_model = np.column_stack((np.append(thick_vals, 1000), vp_vals, vs_vals, rho_vals))
            velocity_model /= 1000 # m to km and kg/m^3 to g/cm^3
            try:
                for mode, fs in zip(self.modes, self.fs_per_mode):
                    pd = PhaseDispersion(*velocity_model.T)
                    periods = 1 / fs[::-1]
                    d_pred = pd(periods, mode=mode, wave="rayleigh").velocity
                    if d_pred.shape[0] != periods.shape[0]: # Test if the dispersion curve is too short - It is often the case for low velocities (i.e. high periods) on superior modes
                        raise DispersionError(f"Dispersion curve length for mode {mode} is not the same as the observed one")
                unstable = False
                print(f'ID {ID} | x_mid {x_mid} | Found stable initialisation')
            except DispersionError:
                unstable = True
        vals = np.concatenate((vs_vals, thick_vals))
        param_values = dict()
        for i, name in enumerate(param_space.parameters.keys()):
            param_values[name] = np.array([vals[i]])
        return ParameterSpaceState(1, param_values)

def forward_model_disba(thick_vals, vs_vals, mode, fs):
    vp_vals = vs_vals * VP_VS
    rho_vals = 0.32 * vp_vals + 0.77*1000
    velocity_model = np.column_stack((thick_vals, vp_vals, vs_vals, rho_vals))
    velocity_model /= 1000 # m to km and kg/m^3 to g/cm^3
    pd = PhaseDispersion(*velocity_model.T)# dc=0.000005)
    periods = 1 / fs[::-1] # Hz to s and reverse
    pd = pd(periods, mode=mode, wave="rayleigh")
    vr = pd.velocity
    if pd.period.shape[0] < periods.shape[0]: # If the dispersion curve is too short - It is often the case for low velocities (i.e. high periods) on superior modes
        vr = np.append(vr, [np.nan]*(periods.shape[0] - pd.period.shape[0]))
    vr = vr[::-1]*1000 # km/s to m/s and over frequencies
    return vr

def fwd_function(state, mode, fs):
    vs_vals = [state["space"][f"vs{i+1}"][0] for i in range(nb_layers)]
    thick_vals = [state["space"][f"thick{i+1}"][0] for i in range(nb_layers-1)]
    thick_vals.append(1000)
    vs_vals = np.array(vs_vals)
    thick_vals = np.array(thick_vals)
    vr = forward_model_disba(thick_vals, vs_vals, mode, fs)
    return vr
### -----------------------------------------------------------------------------------------------



### ARGUMENTS -------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Process an ID argument.")
parser.add_argument("-ID", type=int, required=True, help="ID of the script")
parser.add_argument("-r", type=str, required=True, help="Path to the folder containing the data")
args = parser.parse_args()
ID = f"{int(args.ID)}"
folder_path = args.r
### -----------------------------------------------------------------------------------------------



### READ JSON -------------------------------------------------------------------------------------
with open(f"{folder_path}/inversion_params.json", "r") as file:
    inversion_params = json.load(file)

x_mid = inversion_params["running_distribution"][ID]["x_mid"]

modes = inversion_params["inversion"]["modes"]
modes = sorted([int(mode) for mode in modes])

nb_layers = inversion_params["inversion"]["nb_layers"]

thickness_mins = np.array(inversion_params["inversion"]["thickness_mins"], dtype=np.float64)
thickness_maxs = np.array(inversion_params["inversion"]["thickness_maxs"], dtype=np.float64)
thickness_perturb_stds = np.array(inversion_params["inversion"]["thickness_perturb_stds"])

vs_mins = np.array(inversion_params["inversion"]["vs_mins"], dtype=np.float64)
vs_maxs = np.array(inversion_params["inversion"]["vs_maxs"], dtype=np.float64)
vs_perturb_stds = np.array(inversion_params["inversion"]["vs_perturb_stds"])

n_iterations = inversion_params["inversion"]["n_iterations"]
n_burnin_iterations = inversion_params["inversion"]["n_burnin_iterations"]
n_chains = inversion_params["inversion"]["n_chains"]


with open(f"{folder_path}/computing_params.json", "r") as f:
    computing_params = json.load(f)
    Nx = computing_params["MASW_length"]
    dx = computing_params["positions"][1] - computing_params["positions"][0]
### -----------------------------------------------------------------------------------------------



### READ DISPERSION CURVE -------------------------------------------------------------------------
fs_obs_per_mode = [] 
vr_obs_per_mode = []
err_obs_per_mode = []
existing_modes = []
for mode in modes:
    try:
        data_obs = np.loadtxt(f"{folder_path}/xmid{x_mid}/pick/xmid{x_mid}_obs_M{mode}.pvc")
    except:
        continue
    fs_obs_per_mode.append(data_obs[:,0])
    vr_obs_per_mode.append(data_obs[:,1])
    err_obs_per_mode.append(data_obs[:,2])
    existing_modes.append(mode)
modes = existing_modes
### -----------------------------------------------------------------------------------------------



### OUTPUT DIRECTORY ------------------------------------------------------------------------------
output_dir = f"{folder_path}/xmid{x_mid}/inv/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
log_file = open(f"{output_dir}/xmid{x_mid}_output.log", "w")
sys.stdout = log_file
sys.stderr = log_file

print(f'ID {ID} | x_mid {x_mid} | Parameters loaded')
### -----------------------------------------------------------------------------------------------



### INVERSION -------------------------------------------------------------------------------------
# Targets
targets = []
for mode, err_obs, vr_obs in zip(modes, err_obs_per_mode, vr_obs_per_mode):
    covariance_mat_inv = np.diag(1/err_obs**2)
    target = bb.likelihood.Target(
        name=f"rayleigh_M{mode}", 
        dobs=vr_obs,
        covariance_mat_inv=covariance_mat_inv,
    )
    targets.append(target)
    
# Forward functions
fwd_functions = []
for mode, fs_obs in zip(modes, fs_obs_per_mode):
    fwd_functions.append(lambda state, mode=mode, fs=fs_obs: fwd_function(state, mode, fs))

# Log-likelihood
log_likelihood = LogLikelihood(targets=targets, fwd_functions=fwd_functions)

# Priors
priors = []
for i in range(nb_layers):
    priors.append(bb.prior.UniformPrior(name=f'vs{i+1}',
                                           vmin=vs_mins[i],
                                           vmax=vs_maxs[i],
                                           perturb_std=vs_perturb_stds[i]))
for i in range(nb_layers-1):
    priors.append(bb.prior.UniformPrior(name=f'thick{i+1}',
                                            vmin=thickness_mins[i],
                                            vmax=thickness_maxs[i],
                                            perturb_std=thickness_perturb_stds[i]))

# Parameter space
param_space = bb.parameterization.ParameterSpace(
    name="space", 
    n_dimensions=1, 
    parameters=priors, 
)

# Parameterization
parameterization = CustomParametrization(param_space, modes, fs_obs_per_mode)

# Inversion
inversion = bb.BayesianInversion(
    log_likelihood=log_likelihood, 
    parameterization=parameterization, 
    n_chains=n_chains,
)

print(f'ID {ID} | x_mid {x_mid} | Running inversion')
sys.stdout = sys.__stdout__
print(f'ID {ID} | x_mid {x_mid} | Running inversion')
sys.stdout = log_file

# Run inversion
inversion.run(
    n_iterations=n_iterations,
    burnin_iterations=n_burnin_iterations,
    save_every=150,
    verbose=False,
)

# Print statistics
print(f'ID {ID} | x_mid {x_mid} | Inversion statistics:')

for chain in inversion.chains:
    chain.print_statistics()
### -----------------------------------------------------------------------------------------------
  


### PLOT RESULTS ----------------------------------------------------------------------------------
depth_max = np.nansum(thickness_maxs) + 1 
dz = 0.01

results = inversion.get_results(concatenate_chains=True)

# Extract sampled models
all_sampled_vs = []
for i in range(nb_layers):
    all_sampled_vs.append(np.array(results[f'space.vs{i+1}']).reshape(-1))
all_sampled_vs = np.array(all_sampled_vs)

all_sampled_thick = []
for i in range(nb_layers-1):
    all_sampled_thick.append(np.array(results[f'space.thick{i+1}']).reshape(-1))
all_sampled_thick.append(np.ones_like(all_sampled_vs[-1]))
all_sampled_thick = np.array(all_sampled_thick)

all_sampled_gm = []
for vs_vals, thick_vals in zip(all_sampled_vs.T, all_sampled_thick.T):
    gm = np.column_stack((thick_vals, vs_vals*VP_VS, vs_vals, 0.32*vs_vals*VP_VS + 0.77*1000))
    all_sampled_gm.append(gm)
all_sampled_gm = np.array(all_sampled_gm)



# Misfits
misfits = np.zeros(len(results[f'rayleigh_M{modes[0]}.dpred']))
for mode, vr_obs in zip(modes, vr_obs_per_mode):
    d_pred = np.array(results[f'rayleigh_M{mode}.dpred'])
    for i, vr_pred in enumerate(d_pred):
        misfits[i] += np.sum((vr_obs - vr_pred)**2)
misfits /= np.sum([len(vr_obs) for vr_obs in vr_obs_per_mode])
misfits = np.sqrt(misfits)


# Best layered model
idx = np.argmin(misfits)
best_layered_gm = all_sampled_gm[idx]


# Smooth best layered model
thick_vals = best_layered_gm[:,0]
vp_vals = best_layered_gm[:,1]
vs_vals = best_layered_gm[:,2]
rho_vals = best_layered_gm[:,3]

best_layered_gm[-1,0] = (depth_max - np.sum(best_layered_gm[:-1,0]))/2

depth_vals = [0]
vp_vals = [vp_vals[0]]
vs_vals = [vs_vals[0]]
rho_vals = [rho_vals[0]]

for i, (thick, vp, vs, rho) in enumerate(zip(best_layered_gm[:,0], best_layered_gm[:,1], best_layered_gm[:,2], best_layered_gm[:,3])):
    current_thick = best_layered_gm[i,0]
    current_vp = best_layered_gm[i,1]
    current_vs = best_layered_gm[i,2]
    current_rho = best_layered_gm[i,3]

    depth_vals.append(depth_vals[-1] + current_thick/2)
    vp_vals.append(current_vp)
    vs_vals.append(current_vs)
    rho_vals.append(current_rho)

    if i == len(best_layered_gm)-1:
        break

    next_vp = best_layered_gm[i+1,1]
    next_vs = best_layered_gm[i+1,2]
    next_rho = best_layered_gm[i+1,3]

    depth_vals.append(depth_vals[-1] + current_thick/2)
    vp_vals.append((current_vp + next_vp)/2)
    vs_vals.append((current_vs + next_vs)/2)
    rho_vals.append((current_rho + next_rho)/2)

depth = depth_vals[-1] + dz
depth = np.round(depth, 2)
while depth < depth_max:
    depth_vals = np.append(depth_vals, depth)
    vp_vals = np.append(vp_vals, vp_vals[-1])
    vs_vals = np.append(vs_vals, vs_vals[-1])
    rho_vals = np.append(rho_vals, rho_vals[-1])
    depth += dz
depth_vals = np.append(depth_vals, depth)
vp_vals = np.append(vp_vals, vp_vals[-1])
vs_vals = np.append(vs_vals, vs_vals[-1])
rho_vals = np.append(rho_vals, rho_vals[-1])

depth_vals = np.round(depth_vals, 2)

depth_vals = np.array(depth_vals)
vp_vals = np.array(vp_vals)
vs_vals = np.array(vs_vals)
rho_vals = np.array(rho_vals)

depth_vals_smooth = arange(min(depth_vals), max(depth_vals), dz)
depth_vals_smooth = np.round(depth_vals_smooth, 2)

f = interp1d(depth_vals, vp_vals, kind='cubic')
vp_smooth = f(depth_vals_smooth)

f = interp1d(depth_vals, vs_vals, kind='cubic')
vs_smooth = f(depth_vals_smooth)

f = interp1d(depth_vals, rho_vals, kind='cubic')
rho_smooth = f(depth_vals_smooth)

vp_smooth = vp_smooth[:-1]
vs_smooth = vs_smooth[:-1]
rho_smooth = rho_smooth[:-1]

smooth_best_layered_gm = np.column_stack((np.full_like(vp_smooth, dz), vp_smooth, vs_smooth, rho_smooth))


# Median layered model
median_layered_gm = np.median(all_sampled_gm, axis=0)
median_layered_std = np.std(all_sampled_gm, axis=0)


# Smooth median layered model
thick_vals = median_layered_gm[:,0]
vp_vals = median_layered_gm[:,1]
vs_vals = median_layered_gm[:,2]
rho_vals = median_layered_gm[:,3]
std_vp_vals = median_layered_std[:,1]
std_vs_vals = median_layered_std[:,2]
std_rho_vals = median_layered_std[:,3]

median_layered_gm[-1,0] = (depth_max - np.sum(median_layered_gm[:-1,0]))/2

depth_vals = [0]
vp_vals = [vp_vals[0]]
vs_vals = [vs_vals[0]]
rho_vals = [rho_vals[0]]
std_vp_vals = [std_vp_vals[0]]
std_vs_vals = [std_vs_vals[0]]
std_rho_vals = [std_rho_vals[0]]

for i, (thick, vp, vs, rho, std_vp, std_vs, std_rho) in enumerate(zip(median_layered_gm[:,0], median_layered_gm[:,1], median_layered_gm[:,2], median_layered_gm[:,3], median_layered_std[:,1], median_layered_std[:,2], median_layered_std[:,3])):
    current_thick = median_layered_gm[i,0]
    current_vp = median_layered_gm[i,1]
    current_vs = median_layered_gm[i,2]
    current_rho = median_layered_gm[i,3]
    current_std_vp = median_layered_std[i,1]
    current_std_vs = median_layered_std[i,2]
    current_std_rho = median_layered_std[i,3]

    depth_vals.append(depth_vals[-1] + current_thick/2)
    vp_vals.append(current_vp)
    vs_vals.append(current_vs)
    rho_vals.append(current_rho)
    std_vp_vals.append(current_std_vp)
    std_vs_vals.append(current_std_vs)
    std_rho_vals.append(current_std_rho)

    if i == len(median_layered_gm)-1:
        break

    next_vp = median_layered_gm[i+1,1]
    next_vs = median_layered_gm[i+1,2]
    next_rho = median_layered_gm[i+1,3]
    next_std_vp = median_layered_std[i+1,1]
    next_std_vs = median_layered_std[i+1,2]
    next_std_rho = median_layered_std[i+1,3]

    depth_vals.append(depth_vals[-1] + current_thick/2)
    vp_vals.append((current_vp + next_vp)/2)
    vs_vals.append((current_vs + next_vs)/2)
    rho_vals.append((current_rho + next_rho)/2)
    std_vp_vals.append((current_std_vp + next_std_vp)/2)
    std_vs_vals.append((current_std_vs + next_std_vs)/2)
    std_rho_vals.append((current_std_rho + next_std_rho)/2)

depth = depth_vals[-1] + dz
depth = np.round(depth, 2)
while depth < depth_max:
    depth_vals = np.append(depth_vals, depth)
    vp_vals = np.append(vp_vals, vp_vals[-1])
    vs_vals = np.append(vs_vals, vs_vals[-1])
    rho_vals = np.append(rho_vals, rho_vals[-1])
    std_vp_vals = np.append(std_vp_vals, std_vp_vals[-1])
    std_vs_vals = np.append(std_vs_vals, std_vs_vals[-1])
    std_rho_vals = np.append(std_rho_vals, std_rho_vals[-1])
    depth += dz
depth_vals = np.append(depth_vals, depth)
vp_vals = np.append(vp_vals, vp_vals[-1])
vs_vals = np.append(vs_vals, vs_vals[-1])
rho_vals = np.append(rho_vals, rho_vals[-1])
std_vp_vals = np.append(std_vp_vals, std_vp_vals[-1])
std_vs_vals = np.append(std_vs_vals, std_vs_vals[-1])
std_rho_vals = np.append(std_rho_vals, std_rho_vals[-1])

depth_vals = np.round(depth_vals, 2)

depth_vals = np.array(depth_vals)
vp_vals = np.array(vp_vals)
vs_vals = np.array(vs_vals)
rho_vals = np.array(rho_vals)
std_vp_vals = np.array(std_vp_vals)
std_vs_vals = np.array(std_vs_vals) 
std_rho_vals = np.array(std_rho_vals)

depth_vals_smooth = arange(min(depth_vals), max(depth_vals), dz)
depth_vals_smooth = np.round(depth_vals_smooth, 2)

f = interp1d(depth_vals, vp_vals, kind='cubic')
vp_smooth = f(depth_vals_smooth)

f = interp1d(depth_vals, vs_vals, kind='cubic')
vs_smooth = f(depth_vals_smooth)

f = interp1d(depth_vals, rho_vals, kind='cubic')
rho_smooth = f(depth_vals_smooth)

f = interp1d(depth_vals, std_vp_vals, kind='cubic')
std_vp_smooth = f(depth_vals_smooth)

f = interp1d(depth_vals, std_vs_vals, kind='cubic')
std_vs_smooth = f(depth_vals_smooth)

f = interp1d(depth_vals, std_rho_vals, kind='cubic')
std_rho_smooth = f(depth_vals_smooth)

vp_smooth = vp_smooth[:-1]
vs_smooth = vs_smooth[:-1]
rho_smooth = rho_smooth[:-1]
std_vp_smooth = std_vp_smooth[:-1]
std_vs_smooth = std_vs_smooth[:-1]
std_rho_smooth = std_rho_smooth[:-1]

smooth_median_layered_gm = np.column_stack((np.full_like(vp_smooth, dz), vp_smooth, vs_smooth, rho_smooth))
smooth_median_layered_std = np.column_stack((np.full_like(vp_smooth, dz), std_vp_smooth, std_vs_smooth, std_rho_smooth))



# Median ensemble model
all_ensemble_gm = []
for gm in all_sampled_gm:
    ensemble_gm = []
    for thick, vp, vs, rho in gm:
        ensemble_gm += [[dz, vp, vs, rho]]*int(thick/dz)
    if len(ensemble_gm) < depth_max/dz:
        ensemble_gm += [[dz, vp, vs, rho]]*int(depth_max/dz - len(ensemble_gm))
    all_ensemble_gm.append(ensemble_gm)
all_ensemble_gm = np.array(all_ensemble_gm)

median_ensemble_gm = np.median(all_ensemble_gm, axis=0)
median_ensemble_std = np.std(all_ensemble_gm, axis=0)


median_layered_gm[-1,0] = 1


# Save models
print(f'ID {ID} | x_mid {x_mid} | Saving results in {output_dir}')

# Save best layered model
with open(f"{output_dir}/xmid{x_mid}_best_layered_model.gm", "w") as f:
    f.write(f"{len(best_layered_gm)}\n")
    np.savetxt(f, best_layered_gm, fmt="%.4f")
# Save best layered dispersion
for mode, fs_obs in zip(modes, fs_obs_per_mode):
    best_layered_dc = forward_model_disba(best_layered_gm[:,0], best_layered_gm[:,2], mode, fs_obs)
    with open(f"{output_dir}/xmid{x_mid}_best_layered_M{mode}.pvc", "w") as f:
        np.savetxt(f, np.column_stack((fs_obs, best_layered_dc)), fmt="%.4f")

# Save smooth best layered model
with open(f"{output_dir}/xmid{x_mid}_smooth_best_layered_model.gm", "w") as f:
    f.write(f"{len(smooth_best_layered_gm)}\n")
    np.savetxt(f, smooth_best_layered_gm, fmt="%.4f")
# Save smooth best layered dispersion
for mode, fs_obs in zip(modes, fs_obs_per_mode):
    smooth_best_layered_dc = forward_model_disba(smooth_best_layered_gm[:,0], smooth_best_layered_gm[:,2], mode, fs_obs)
    with open(f"{output_dir}/xmid{x_mid}_smooth_best_layered_M{mode}.pvc", "w") as f:
        np.savetxt(f, np.column_stack((fs_obs, smooth_best_layered_dc)), fmt="%.4f")

# Save median layered model
with open(f"{output_dir}/xmid{x_mid}_median_layered_model.gm", "w") as f:
    f.write(f"{len(median_layered_gm)}\n")
    np.savetxt(f, median_layered_gm, fmt="%.4f")
# Save median layered std
with open(f"{output_dir}/xmid{x_mid}_median_layered_std.gm", "w") as f:
    f.write(f"{len(median_layered_std)}\n")
    np.savetxt(f, median_layered_std, fmt="%.4f")
# Save median layered dispersion
for mode, fs_obs in zip(modes, fs_obs_per_mode):
    median_layered_dc = forward_model_disba(median_layered_gm[:,0], median_layered_gm[:,2], mode, fs_obs)
    with open(f"{output_dir}/xmid{x_mid}_median_layered_M{mode}.pvc", "w") as f:
        np.savetxt(f, np.column_stack((fs_obs, median_layered_dc)), fmt="%.4f")

# Save smooth median layered model
with open(f"{output_dir}/xmid{x_mid}_smooth_median_layered_model.gm", "w") as f:
    f.write(f"{len(smooth_median_layered_gm)}\n")
    np.savetxt(f, smooth_median_layered_gm, fmt="%.4f")
# Save smooth median layered std
with open(f"{output_dir}/xmid{x_mid}_smooth_median_layered_std.gm", "w") as f:
    f.write(f"{len(smooth_median_layered_std)}\n")
    np.savetxt(f, smooth_median_layered_std, fmt="%.4f")
# Save smooth median layered dispersion
for mode, fs_obs in zip(modes, fs_obs_per_mode):
    smooth_median_layered_dc = forward_model_disba(smooth_median_layered_gm[:,0], smooth_median_layered_gm[:,2], mode, fs_obs)
    with open(f"{output_dir}/xmid{x_mid}_smooth_median_layered_M{mode}.pvc", "w") as f:
        np.savetxt(f, np.column_stack((fs_obs, smooth_median_layered_dc)), fmt="%.4f")

# Save median ensemble model
with open(f"{output_dir}/xmid{x_mid}_median_ensemble_model.gm", "w") as f:
    f.write(f"{len(median_ensemble_gm)}\n")
    np.savetxt(f, median_ensemble_gm, fmt="%.4f")
# Save median ensemble std
with open(f"{output_dir}/xmid{x_mid}_median_ensemble_std.gm", "w") as f:
    f.write(f"{len(median_ensemble_std)}\n")
    np.savetxt(f, median_ensemble_std, fmt="%.4f")
# Save median ensemble dispersion
for mode, fs_obs in zip(modes, fs_obs_per_mode):
    median_ensemble_dc = forward_model_disba(median_ensemble_gm[:,0], median_ensemble_gm[:,2], mode, fs_obs)
    with open(f"{output_dir}/xmid{x_mid}_median_ensemble_M{mode}.pvc", "w") as f:
        np.savetxt(f, np.column_stack((fs_obs, median_ensemble_dc)), fmt="%.4f")
    
    
    
print(f'ID {ID} | x_mid {x_mid} | Plotting results in {output_dir}')

# Update dispersion image with all obs and pred modes
FV = np.loadtxt(f"{folder_path}/xmid{x_mid}/comp/xmid{x_mid}_dispersion.csv", delimiter=",")
fs = np.loadtxt(f"{folder_path}/xmid{x_mid}/comp/xmid{x_mid}_fs.csv", delimiter=",")
vs = np.loadtxt(f"{folder_path}/xmid{x_mid}/comp/xmid{x_mid}_vs.csv", delimiter=",")

obs_modes = []
for mode in modes:
    pvc = np.loadtxt(f"{folder_path}/xmid{x_mid}/pick/xmid{x_mid}_obs_M{mode}.pvc")
    obs_modes.append(pvc)
    
pred_modes = []
for mode, fs_obs in zip(modes, fs_obs_per_mode):
    pred_modes.append(np.column_stack((fs_obs, forward_model_disba(median_layered_gm[:,0], median_layered_gm[:,2], mode, fs_obs))))

full_pred_modes = []
inRange = True
mode = 0
velocity_model = median_layered_gm/1000
pd = PhaseDispersion(*velocity_model.T)
periods = 1 / fs[fs>0]
periods = periods[::-1]
while inRange:
    data = pd(periods, mode=mode, wave="rayleigh")
    if data.period.shape[0] == 0:
        inRange = False
    mode+=1
    full_pred_modes.append(np.column_stack((1/data.period[::-1], data.velocity[::-1]*1000)))

name_path = f"{output_dir}/xmid{x_mid}_dispersion.svg"
display_dispersion_img(FV, fs, vs, obs_modes=obs_modes, pred_modes=pred_modes, full_pred_modes=full_pred_modes, path=name_path, normalization='Frequency', dx=dx)



# Plot inversion results
fig, axs = plt.subplots(1, 2, figsize=(18*CM, 12*CM), dpi=300)

ax = axs[0]

for i, (fs_obs, vr_obs, err_obs, mode) in enumerate(zip(fs_obs_per_mode, vr_obs_per_mode, err_obs_per_mode, modes)):
    # Plot inferred data
    d_pred = np.array(results[f'rayleigh_M{mode}.dpred'])
    percentiles = np.percentile(d_pred, (10, 50, 90), axis=0)
    label = '10th-90th percentiles' if i == 0 else '_nolegend_'
    ax.fill_between(fs_obs, percentiles[0], percentiles[2], color='k', alpha=0.2, label=label, zorder=1)
    label = '50th percentile' if i == 0 else '_nolegend_'
    ax.plot(fs_obs, percentiles[1], color='k', label=label, linewidth=0.2, linestyle='--', zorder=2)

    # Plot observed data
    label = 'Observed Data' if i == 0 else '_nolegend_'
    ax.errorbar(fs_obs, vr_obs, yerr=err_obs, fmt='o', color='tab:blue', label=label, markersize=1.5, capsize=0, elinewidth=0.3, zorder=3)

    # Plot best layered model
    label = 'Best layered model' if i == 0 else '_nolegend_'
    ax.plot(fs_obs, forward_model_disba(best_layered_gm[:,0], best_layered_gm[:,2], mode, fs_obs), 'o', color='tab:green', label=label, markersize=1.5, zorder=4)

    # Plot smooth best layered model
    label = 'Smooth best layered model' if i == 0 else '_nolegend_'
    ax.plot(fs_obs, forward_model_disba(smooth_best_layered_gm[:,0], smooth_best_layered_gm[:,2], mode, fs_obs), 'o', color='green', label=label, markersize=1.5, zorder=6)

    # Plot median layered model
    label = 'Median layered model' if i == 0 else '_nolegend_'
    ax.plot(fs_obs, forward_model_disba(median_layered_gm[:,0], median_layered_gm[:,2], mode, fs_obs), 'o', color='orange', label=label, markersize=1.5, zorder=4)

    # Plot smooth median layered model
    label = 'Smooth median layered model' if i == 0 else '_nolegend_'
    ax.plot(fs_obs, forward_model_disba(smooth_median_layered_gm[:,0], smooth_median_layered_gm[:,2], mode, fs_obs), 'o', color='tab:orange', label=label, markersize=1.5, zorder=6)

    # Plot median ensemble model
    label = 'Median ensemble model' if i == 0 else '_nolegend_'
    ax.plot(fs_obs, forward_model_disba(median_ensemble_gm[:,0], median_ensemble_gm[:,2], mode, fs_obs), 'o', color='tab:red', label=label, markersize=1.5, zorder=5)

ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Phase velocity $v_{R}$ [m/s]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))

ax = axs[1]

# Plot all models
cmap = plt.get_cmap('Greys_r')
def _forward(x):
    return np.log(x)
def _inverse(x):
    return np.exp(x)
norm = colors.FuncNorm((_forward, _inverse), vmin=np.min(misfits), vmax=np.max(misfits))
sorted_idx = np.argsort(misfits)[::-1]
all_sampled_gm = all_sampled_gm[sorted_idx]
misfits = misfits[sorted_idx]
for gm, misfit in zip(all_sampled_gm, misfits):
    thick_vals = np.copy(gm[:,0])
    vs_vals = np.copy(gm[:,2])
    thick_vals[-1] = depth_max - np.sum(thick_vals[:-2])
    depth_vals = np.cumsum(thick_vals)
    depth_vals = np.insert(depth_vals, 0, 0)
    vs_vals = np.append(vs_vals, vs_vals[-1])
    ax.step(vs_vals, depth_vals, color=cmap(norm(misfit)), label='_nolegend_', linewidth=0.5)

# Plot best layered model
thick_vals = np.copy(best_layered_gm[:,0])
vs_vals = np.copy(best_layered_gm[:,2])
thick_vals[-1] = depth_max - np.sum(thick_vals[:-2])
depth_vals = np.cumsum(thick_vals)
depth_vals = np.insert(depth_vals, 0, 0)
vs_vals = np.append(vs_vals, vs_vals[-1])
ax.step(vs_vals, depth_vals, color='tab:green', label='Best layered model', linewidth=1)

# Plot smooth best layered model
thick_vals = np.copy(smooth_best_layered_gm[:,0])
vs_vals = np.copy(smooth_best_layered_gm[:,2])
thick_vals[-1] = depth_max - np.sum(thick_vals[:-2])
depth_vals = np.cumsum(thick_vals)
depth_vals = np.insert(depth_vals, 0, 0)
vs_vals = np.append(vs_vals, vs_vals[-1])
ax.step(vs_vals, depth_vals, color='green', label='Smooth best layered model', linewidth=1)

# Plot median layered model
thick_vals = np.copy(median_layered_gm[:,0])
vs_vals = np.copy(median_layered_gm[:,2])
thick_vals[-1] = depth_max - np.sum(thick_vals[:-2])
depth_vals = np.cumsum(thick_vals)
depth_vals = np.insert(depth_vals, 0, 0)
vs_vals = np.append(vs_vals, vs_vals[-1])
ax.step(vs_vals, depth_vals, color='orange', label='Median layered model', linewidth=1)

# Plot smooth median layered model
thick_vals = np.copy(smooth_median_layered_gm[:,0])
vs_vals = np.copy(smooth_median_layered_gm[:,2])
thick_vals[-1] = depth_max - np.sum(thick_vals[:-2])
depth_vals = np.cumsum(thick_vals)
depth_vals = np.insert(depth_vals, 0, 0)
vs_vals = np.append(vs_vals, vs_vals[-1])
ax.step(vs_vals, depth_vals, color='tab:orange', label='Smooth median layered model', linewidth=1)
# Plot std smooth model
std_vs_vals = np.copy(smooth_median_layered_std[:,2])
std_vs_vals = np.append(std_vs_vals, std_vs_vals[-1])
ax.step(vs_vals-std_vs_vals, depth_vals, color='tab:orange', label='Standard deviation', linewidth=1, linestyle='dotted')
ax.step(vs_vals+std_vs_vals, depth_vals, color='tab:orange', label='_nolegend_', linewidth=1, linestyle='dotted')

# Plot median ensemble model
thick_vals = np.copy(median_ensemble_gm[:,0])
vs_vals = np.copy(median_ensemble_gm[:,2])
thick_vals[-1] = depth_max - np.sum(thick_vals[:-2])
depth_vals = np.cumsum(thick_vals)
depth_vals = np.insert(depth_vals, 0, 0)
vs_vals = np.append(vs_vals, vs_vals[-1])
ax.step(vs_vals, depth_vals, color='tab:red', label='Median ensemble model', linewidth=1)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
ax.invert_yaxis()
ax.set_ylim(depth_max, 0)
ax.set_xlabel('Shear wave velocity $v_{S}$ [m/s]')
ax.set_ylabel('Depth [m]')

plt.tight_layout()
fig.savefig(f"{output_dir}/xmid{x_mid}_density_curves.png")



print(f'ID {ID} | x_mid {x_mid} | Plotting marginals in {output_dir}')

results = inversion.get_results()
samples = {f'$v_{{s{i+1}}}$ [m/s]': np.array(results[f'space.vs{i+1}']).T for i in range(nb_layers)}
samples.update({f'$H_{{{i+1}}}$ [m]': np.array(results[f'space.thick{i+1}']).T for i in range(nb_layers-1)})

rows = nb_layers*2-1
cols = rows
width = 2.6 * CM * cols
height = 2.4 * CM * rows
az.rcParams["plot.max_subplots"] = rows*cols
fig, axs = plt.subplots(rows, cols, figsize=(width, height), dpi=300)
_ = az.plot_pair(
    samples,
    marginals=True,
    kind='kde',
    kde_kwargs={
        'hdi_probs': [0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
        'contourf_kwargs': {'cmap': 'Blues'},
        },
    ax=axs,
    textsize=FONT_SIZE,
    colorbar=True,
    )

for i in range(rows):
    axs[i,i].set_ylabel('Probability')
    axs[i,i].yaxis.set_label_position("right")
    axs[i,i].yaxis.tick_right()
    if i==0:
        legend = 'a priori'
    else:
        legend = '_nolegend_'
plt.savefig(f"{output_dir}/xmid{x_mid}_marginals.png")

toc = time()

print(f'ID {ID} | x_mid {x_mid} | Inversion completed in {toc-tic:.1f} s')
sys.stdout = sys.__stdout__
print(f'\033[92mID {ID} | x_mid {x_mid} | Inversion completed in {toc-tic:.1f} s\033[0m')
sys.stdout = log_file