"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""

import sys
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from matplotlib.path import Path

sys.path.append("./modules/")
from misc import arange



### -----------------------------------------------------------------------------------------------
def phase_shift(XT, si, offsets, vmin, vmax, dv, f_min, fmax):
    """
    Constructs a FV dispersion diagram with the phase-shift method from Park et al. (1999)
    args :
        XT (numpy array) : data
        si (float) : sampling interval in seconds
        offsets (numpy array) : offsets in meter
        vmin, vmax (float) : velocities to scan in m/s
        dv (float) : velocity step in m/s
        fmax (float) : maximum frequency computed
    returns :
        fs : frequency axis
        vs : velocity axis
        FV: dispersion plot
    """   
    Nt = XT.shape[1]
    XF = rfft(XT, axis=(1), n=Nt)

    fs = rfftfreq(Nt, si)
    try:
        fimin = np.where(fs >= f_min)[0][0]
    except:
        fimin = 0
    try :
        fimax = np.where(fs >= fmax)[0][0]
    except :
        fimax = len(fs)-1
    fs = fs[fimin:fimax+1]
    XF = XF[: , fimin:fimax+1]

    vs = arange(vmin, vmax, dv)

    # Vecrorized version (otpimized)
    FV = np.zeros((len(fs), len(vs)))
    for v_i, v in enumerate(vs):
        dphi = 2 * np.pi * offsets[..., None] * fs / v
        FV[:, v_i] = np.abs(np.sum(XF/np.abs(XF)*np.exp(1j*dphi), axis=0))   
    
    # Loop version (not optimized)
    # FV = np.zeros((len(fs), len(vs)))
    # for j, v in enumerate(vs):
    #     for i, f in enumerate(fs):   
    #         sum_exp = 0
    #         for k in range(len(offsets)):
    #             dphi = 2 * np.pi * offsets[k] * f / v
    #             exp_term = np.exp(1j * dphi) * (XF[k, i] / abs(XF[k, i]))
    #             sum_exp += exp_term
    #             FV[i, j] = abs(sum_exp)

    return fs, vs, FV
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def extract_curve(FV, fs, vs, poly_coords, smooth):
    """
    Extracts f-v dispersion curve from f-v dispersion diagram by aiming maximums

    args :
        FV (2D numpy array) : dispersion diagram
        fs (1D numpy array) : frequency axis
        vs (1D numpy array) : velocity axis
        start (tuple of floats) : starting coordinates (f,v) values
        end (tuple of floats) : ending coordinates (f,v) values

    returns :
        curve (1D numpy array[velocity]) : f-v dispersion curve
    """
    
    FV = np.copy(FV)
    for i in range(FV.shape[0]):
        FV[i,:] = FV[i,:] / np.max(FV[i,:])

    df = fs[1] - fs[0]
    dv = vs[1] - vs[0]
    idx = np.zeros((len(poly_coords), 2), dtype=int)
    for i, (f,v) in enumerate(poly_coords):
        idx[i][0] = int(f/df)
        idx[i][1] = int(v/dv)
        
    # Make the low frequency limit of the polygon vertical to avoid the picking to follow the polygon limit at low frequencies
    idx[-1][0] = idx[0][0]
    
    poly_path = Path(idx)
    x,y = np.mgrid[:FV.shape[0], :FV.shape[1]]
    coors = np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))

    mask = poly_path.contains_points(coors)
    mask = mask.reshape(FV.shape)

    FV_masked = FV * mask
    
    f_picked = []
    v_picked =[]

    f_start_i = np.min(idx[:, 0])
    f_end_i = np.max(idx[:, 0])
    v_start_i = np.min(idx[:, 1])
    v_end_i = np.max(idx[:, 1])

    FV_tmp = FV_masked[f_start_i:f_end_i, v_start_i+1:v_end_i]

    for i, FV_f in enumerate(FV_tmp): # FV_f is a vector of velocities for a frequency f
        v_max_i = np.where(FV_f == FV_f.max())[0][0]
        v_max = vs[v_max_i+v_start_i]
        if v_max_i+v_start_i == v_end_i-1 and i != 0:
            v_picked.append(v_picked[-1])
        else:
            v_picked.append(v_max)
        f_picked.append(fs[i+f_start_i])

    f_picked = np.array(f_picked)
    v_picked = np.array(v_picked)

    if not smooth:
        return f_picked[1:], v_picked[1:]
           
    if (len(v_picked)/2) % 2 == 0:
        wl = len(v_picked)//2 + 1
    else:
        wl = len(v_picked)//2
    v_picked = savgol_filter(v_picked, window_length=wl, polyorder=5)
    return f_picked[1:], v_picked[1:]
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def resamp_wavelength(f, v):
    w = v / f
    func_v = interp1d(w, v)
    w_resamp = arange(np.ceil(min(w)), np.floor(max(w)), 1)
    v_resamp = func_v(w_resamp)
    return w_resamp, v_resamp[::-1]
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def resamp_frequency(f, v):
    func_v = interp1d(f, v)
    f_resamp = arange(np.ceil(min(f)), np.floor(max(f)), 1)
    v_resamp = func_v(f_resamp)
    return f_resamp, v_resamp[::-1]
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def resamp(f, v, err, wmax=None):
    w = v / f
    min_w = np.ceil(min(w))
    max_w = np.floor(max(w))
    if min_w < max_w:
        func_v = interp1d(w, v, kind='linear')
        func_err = interp1d(w, err, kind='linear', fill_value='extrapolate')
        w_resamp = arange(min_w, max_w, 1)
        v_resamp = func_v(w_resamp)
        err_resamp = func_err(w_resamp)
        if wmax is not None:
            if max(w_resamp) > wmax:
                try:
                    idx = np.where(w_resamp >= wmax)[0][0]
                except:
                    idx = len(w_resamp)-1
                w_resamp = w_resamp[:idx+1]
                v_resamp = v_resamp[:idx+1]
                err_resamp = err_resamp[:idx+1]
        f_resamp = v_resamp/w_resamp
        f_resamp, v_resamp, err_resamp = zip(*sorted(zip(f_resamp, v_resamp, err_resamp)))
    else : 
        f_resamp = [f[0]]
        v_resamp = [v[0]]
        err_resamp = [err[0]]
    return f_resamp, v_resamp, err_resamp
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def lorentzian_error(v_picked, f_picked, dx, Nx, a=0.5):
    # Factor to adapt error depending on window size
    fac = 10**(1/np.sqrt(Nx*dx))
    
    # Resolution
    Dc_left = 1 / (1/v_picked - 1/(2*f_picked*Nx*fac*dx))
    Dc_right = 1 / (1/v_picked + 1/(2*f_picked*Nx*fac*dx))
    Dc = np.abs(Dc_left - Dc_right)
    
    # Absolute uncertainty
    dc = (10**-a) * Dc

    for i, (err, v) in enumerate(zip(dc, v_picked)):
        if err > 0.4*v :
            dc[i] = 0.4*v
        if err < 5 :
            dc[i] = 5

    return dc
### -----------------------------------------------------------------------------------------------