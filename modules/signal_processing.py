"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""




import numpy as np
from scipy.signal.windows import tukey




### -----------------------------------------------------------------------------------------------
def normalize(array, dt, clip_factor=6, clip_weight=10, norm_win=10, norm_method="1bit"):
    """
    Temporal normalization.
    """
    array2 = np.copy(array)
    for tr in array2.T:
        if norm_method == 'clipping':
            lim = clip_factor * np.std(tr.data)
            tr[tr > lim] = lim
            tr[tr < -lim] = -lim
        elif norm_method == "clipping_iter":
            lim = clip_factor * np.std(np.abs(tr))
            # as long as still values left above the waterlevel, clip_weight
            while tr[np.abs(tr) > lim].size > 0:
                tr[tr > lim] /= clip_weight
                tr[tr < -lim] /= clip_weight
        elif norm_method == 'ramn':
            lwin = int(norm_win/dt)
            st = 0                                               # starting point
            N = lwin                                             # ending point
            while N < len(tr):
                win = tr[st:N]
                w = np.mean(np.abs(win)) / (2. * lwin + 1)
                # weight center of window
                tr[int(st + lwin / 2)] /= w
                # shift window
                st += 1
                N += 1
            # taper edges
            taper = get_window(len(tr))
            tr *= taper
        elif norm_method == "1bit":
            tr = np.sign(tr)
            tr = np.float32(tr)
    return array2
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def get_window(N, alpha=0.2):
    window = np.ones(N)
    x = np.linspace(-1., 1., N)
    ind1 = (abs(x) > 1 - alpha) * (x < 0)
    ind2 = (abs(x) > 1 - alpha) * (x > 0)
    window[ind1] = 0.5 * (1 - np.cos(np.pi * (x[ind1] + 1) / alpha))
    window[ind2] = 0.5 * (1 - np.cos(np.pi * (x[ind2] - 1) / alpha))
    return window
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def whiten(array, f_ech, freqmin, freqmax):

    array2 = np.empty(array.shape)
    nsamp = f_ech

    for i, tr in enumerate(array.T):
        
        n = len(tr)
        frange = float(freqmax) - float(freqmin)
        nsmo = int(np.fix(min(0.01, 0.5 * (frange)) * float(n) / nsamp))
        f = np.arange(n) * nsamp / (n - 1.)
        JJ = ((f > float(freqmin)) & (f<float(freqmax))).nonzero()[0]
            
        # signal FFT
        FFTs = np.fft.fft(tr)
        FFTsW = np.zeros(n) + 1j * np.zeros(n)

        # Apodization to the left with cos^2 (to smooth the discontinuities)
        smo1 = (np.cos(np.linspace(np.pi / 2, np.pi, nsmo+1))**2)
        FFTsW[JJ[0]:JJ[0]+nsmo+1] = smo1 * np.exp(1j * np.angle(FFTs[JJ[0]:JJ[0]+nsmo+1]))

        # boxcar
        FFTsW[JJ[0]+nsmo+1:JJ[-1]-nsmo] = np.ones(len(JJ) - 2 * (nsmo+1))\
        * np.exp(1j * np.angle(FFTs[JJ[0]+nsmo+1:JJ[-1]-nsmo]))

        # Apodization to the right with cos^2 (to smooth the discontinuities)
        smo2 = (np.cos(np.linspace(0., np.pi/2., nsmo+1))**2.)
        espo = np.exp(1j * np.angle(FFTs[JJ[-1]-nsmo:JJ[-1]+1]))
        FFTsW[JJ[-1]-nsmo:JJ[-1]+1] = smo2 * espo

        whitedata = 2. * np.fft.ifft(FFTsW).real
        
        array2[:, i] = np.require(whitedata, dtype="float32")

    return array2
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def cut(TX_raw, ts, t_cut_start, t_cut_end):
    t_cut_start = np.round(t_cut_start, 3)
    t_cut_end = np.round(t_cut_end, 3)
    ts = np.round(ts, 6)
    TX_raw = TX_raw.copy()
    i_cut_start = np.where(ts >= t_cut_start)[0][0]
    i_cut2 = np.where(ts >= t_cut_end)[0][0]
    ts_cut = np.linspace(0, t_cut_end-t_cut_start, i_cut2-i_cut_start+1)
    TX = TX_raw[i_cut_start:i_cut2+1, ::]    
    for i, data in enumerate(np.transpose(TX)):
        TX[:,i] = np.transpose(data * tukey(len(ts_cut)))
    return TX, ts_cut
### -----------------------------------------------------------------------------------------------