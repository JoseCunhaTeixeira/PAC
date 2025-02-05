"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from modules.misc import verify_expected
import matplotlib.colors as mcolors

from modules.misc import arange


plt.rcParams.update({'font.size': 14})
_DPI = 300
CM = 1/2.54



### -----------------------------------------------------------------------------------------------
def plot_wiggle(XT, positions, dt, norm=None):
    Nt = XT.shape[1]
    ts = arange(0, Nt*dt-dt, dt)
    Nx = XT.shape[0]
    
    if norm == 'trace':
        for i in range(Nx):
            XT[i, :] = 2*(XT[i, :] - np.nanmin(XT[i, :])) / (np.nanmax(XT[i, :]) - np.nanmin(XT[i, :])) - 1
    elif norm == 'global':
        XT = 2*(XT - np.nanmin(XT)) / (np.nanmax(XT) - np.nanmin(XT)) - 1
        
    fig = px.imshow(XT.T,
                    labels=dict(x="Trace [#]", y="Time [s]", color="Amplitude"),
                    x=positions,
                    y=ts,
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    )
    
    fig.update_layout(xaxis=dict(side='top'))
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(
                text="Amplitude",  # Colorbar label
                side="right"       # Position vertically on the right
            )
        )
    )
    return fig
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def plot_spectrum(XT, positions, dt, norm='trace'):
    Nt = XT.shape[1]
    Nx = XT.shape[0]
    
    XF = fft(XT, axis=(1), n=Nt)
    XF = np.abs(XF[:, :Nt//2+1])
    fs = fftfreq(Nt, dt)
    fs = np.abs(fs[:Nt//2+1])
    
    if norm == 'trace':
        for i in range(Nx):
            XF[i, :] = XF[i, :] / np.nanmax(XF[i, :])
    elif norm == 'global':
        XF = XF / np.nanmax(XF)
        
    fig = px.imshow(XF.T,
                    labels=dict(x="Position [m]", y="Frequency [Hz]", color="Amplitude"),
                    x=positions,
                    y=fs,
                    aspect='auto',
                    color_continuous_scale='gray_r',
                    )
    
    fig.update_layout(xaxis=dict(side='top'))
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(
                text="Amplitude",  # Colorbar label
                side="right"       # Position vertically on the right
            )
        )
    )
    return fig
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def plot_disp(FV, fs, vs, dx=None, Nx=None, norm=None):
    FV = np.copy(FV)
    if norm == "Frequencies":
        for i, f in enumerate(fs):
            FV[i, :] = FV[i, :] / np.nanmax(FV[i, :])
    elif norm == "Velocities":
        for i, v in enumerate(vs):
            FV[:, i] = FV[:, i] / np.nanmax(FV[:, i])
    elif norm == 'Global':
        FV /= np.nanmax(FV) 
    
    cmap = plt.get_cmap('gist_stern_r')
    colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]
                                    
    fig = px.imshow(FV.T**2,
                    labels=dict(x="Frequency [Hz]", y="Phase velocity [m/s]", color="Amplitude"),
                    x=fs,
                    y=vs,
                    aspect='auto',
                    color_continuous_scale=colorscale,
                    origin='lower',
                    )
    if dx is not None:
        lambda_min = 2*dx
        vs_min = fs * lambda_min
        vs_min[vs_min > np.max(vs)] = np.nan
        vs_min[vs_min < np.min(vs)] = np.nan
        fig.add_trace(go.Scatter(x=fs, y=vs_min, mode='lines', name="&#955;<sub>nyq</sub>", line=dict(color='grey', width=2, dash='dash')))
        if Nx is not None:
            lambda_max = (Nx-1)*dx
            vs_max = fs * lambda_max
            vs_max[vs_max > np.max(vs)] = np.nan
            vs_max[vs_max < np.min(vs)] = np.nan
            fig.add_trace(go.Scatter(x=fs, y=vs_max, mode='lines', name='6<i>L</i><sub>window</sub>', line=dict(color='grey', width=2, dash='dash')))
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(
                text="Amplitude",  # Colorbar label
                side="right"       # Position vertically on the right
            )
        ),
    )
    return fig
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def plot_pseudo_section(FX, fs, xs, wavelength=False):
    FX = np.copy(FX)
    fs = np.copy(fs)
    xs = np.copy(xs)
    
    if not wavelength:
        fig = px.imshow(np.flipud(FX.T),
                        labels=dict(x="Position [m]", y="Frequency [Hz]", color="Phase velocity [m/s]"),
                        x=xs,
                        y=fs,
                        aspect='auto',
                        color_continuous_scale='cividis',
                        origin='lower',
                        )
    else:
        fig = px.imshow(np.flipud(FX.T),
                        labels=dict(x="Position [m]", y="Wavelength [m]", color="Phase velocity [m/s]"),
                        x=xs,
                        y=fs,
                        aspect='auto',
                        color_continuous_scale='cividis',
                        origin='upper',
                        )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(
                text="Phase velocity [m/s]",  # Colorbar label
                side="right"       # Position vertically on the right
            )
        )
    )
    return fig
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def plot_inverted_section(section, positions, depths, zmin=None, zmax=None):
    section = np.copy(section)
    positions = np.copy(positions)
    depths = np.copy(depths)
    
    if zmin is not None:
        section[section < zmin] = zmin
    if zmax is not None:
        section[section > zmax] = zmax
    
    cmap = plt.get_cmap('terrain')
    colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]
    
    fig = px.imshow(section.T,
                    labels=dict(x="Position [m]", y="Depth [m]", color="Shear wave velocity [m/s]"),
                    x=positions,
                    y=depths,
                    aspect='auto',
                    color_continuous_scale=colorscale,
        )    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(
                text="Shear wave velocity [m/s]",  # Colorbar label
                side="right"       # Position vertically on the right
            )
        )
    )
    return fig
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def plot_std_section(section, positions, depths, zmin=None, zmax=None):
    section = np.copy(section)
    positions = np.copy(positions)
    depths = np.copy(depths)
    
    if zmin is not None:
        section[section < zmin] = zmin
    if zmax is not None:
        section[section > zmax] = zmax

    cmap = plt.get_cmap('afmhot_r')
    colorscale = [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]

    fig = px.imshow(section.T,
                    labels=dict(x="Position [m]", y="Depth [m]", color="Stardard deviation [m/s]"),
                    x=positions,
                    y=depths,
                    aspect='auto',
                    color_continuous_scale=colorscale,
                    zmin=0,
        )    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(
                text="Stardard deviation [m/s]",  # Colorbar label
                side="right"       # Position vertically on the right
            )
        )
    )
    return fig
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def plot_geophones(selected_geophone_positions, geophone_positions, source_position):    
    non_selected_geophones = [x for x in geophone_positions if x not in selected_geophone_positions]

    # Create a scatter plot
    fig = go.Figure()
    
    # Add scatter points for non-selected geophones in grey
    fig.add_trace(go.Scatter(
        x=non_selected_geophones,
        y=[0] * len(non_selected_geophones),
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='grey'),  # Grey color
        showlegend=False,
    ))
    
    # Add scatter points for selected geophones in blue
    fig.add_trace(go.Scatter(
        x=selected_geophone_positions,
        y=[0] * len(selected_geophone_positions),
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='blue'),  # Blue color
        showlegend=False,
    ))
    
    # Add scatter points for source in red
    fig.add_trace(go.Scatter(
        x=[source_position],
        y=[0],
        mode='markers',
        marker=dict(symbol='star', size=10, color='red'),  # Blue color
        showlegend=False,
    ))

    # Update layout to hide the y-axis
    fig.update_layout(
        title="Geophones",
        xaxis_title="Position [m]",
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis_range=[-1, 1],  # Optional: Control the vertical space
        xaxis_range=[min(source_position, min(geophone_positions))-1, max(source_position, max(geophone_positions))+1],
        height=200,
    )

    return fig
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def display_seismic_wiggle_fromStream(stream, x_sensors, path, scale=1.0, norm_method=None, **kwargs):
    for i, tr in enumerate(stream) :
        tr.stats.distance = x_sensors[i]*1000
    fig = plt.figure(figsize=(18*CM, 8*CM), dpi=_DPI)
    if len(kwargs) == 0:
        stream.plot(type='section', time_down=True, scale=scale, fig=fig, fillcolors=("black",None), norm_method=norm_method, linewidth=0.5)
    if len(kwargs) == 2:
        verify_expected(kwargs, ["recordstart", "recordlength"])
        recordstart=kwargs["recordstart"]
        recordlength=kwargs["recordlength"]
        stream.plot(type='section', time_down=True, scale=scale, fig=fig, fillcolors=("black",None), norm_method=norm_method, linewidth=0.5, recordstart=recordstart, recordlength=recordlength)
    plt.tight_layout()
    plt.xlabel("Position [m]")
    plt.ylabel("Time [s]")

    plt.savefig(path, format='png', dpi='figure', bbox_inches='tight')
    plt.close()
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def display_spectrum_img_fromArray(array, dt, x_sensors, path1, path2, norm_method=None):
    Nt, Nx = array.shape
    SP = np.fft.rfft(array, axis=0)
    fs = np.fft.rfftfreq(Nt, dt)

    if norm_method == "stream":
        SP = np.abs(SP)/np.max(np.abs(SP))
    elif norm_method == "trace":
        for i, col in enumerate(SP.T):
            SP[:,i] = np.abs(col) / np.max(np.abs(col))

    extent = [x_sensors[0], x_sensors[-1], fs[0], fs[-1]]
    
    plt.figure(figsize=(18*CM, 8*CM), dpi=_DPI)
    plt.imshow(np.flipud(np.abs(SP)), cmap='gray_r', extent=extent, aspect="auto")
    plt.xlabel("Position [m]")
    plt.ylabel("Frequency [Hz]")
    cbar = plt.colorbar(shrink=0.5, aspect=15)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])
    cbar.set_label("Amplitude [-]")
    plt.tight_layout()
    plt.savefig(path1, format='svg', dpi='figure', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(18*CM, 8*CM), dpi=_DPI)
    plt.plot(fs, abs(SP[:,0]), fs, abs(SP[:, Nx//2]), fs, abs(SP[:,-1]), linewidth=0.5)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [-]")
    plt.legend(["Trace 1", f"Trace {Nx//2}", f"Trace {Nx}"], loc="upper right")
    plt.savefig(path2, format='svg', dpi='figure', bbox_inches='tight')
    plt.close()
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def display_dispersion_img(FV_arr, fs, vs, path, obs_modes=[], pred_modes=[], full_pred_modes=[], Nx=None, dx=None, normalization=None, errbars=True):
    """
    obs_modes = [mode0, mode1, ...]
    """
    FV = np.copy(FV_arr)
    
    if normalization == "Frequency":
        for i, col in enumerate(FV):
            FV[i, :] /= np.max(col)
    elif normalization == "Global":
        FV = FV / np.max(FV)
        
    plt.figure(figsize=(18*CM, 8*CM), dpi=_DPI)
    extent = [fs[0], fs[-1], vs[0], vs[-1]]
    plt.imshow(np.flipud(FV.T**2), cmap='gist_stern_r', extent=extent, aspect="auto")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("$v_{R}$ [m/s]")
    plt.tight_layout()
    
    #add colorbar at the right
    cbar = plt.colorbar(shrink=0.5, aspect=15)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])
    cbar.set_label("Amplitude [-]")
    
    if len(obs_modes) > 0 :
        for mode in obs_modes:
            if len(mode.shape) == 1:
                mode = mode.reshape(1,-1)
            fs_mode = mode[:,0]
            vs_mode = mode[:,1]
            dc_mode = mode[:,2]
            if errbars == True:
                plt.errorbar(fs_mode[::1], vs_mode[::1], dc_mode[::1], fmt=f"o-", ecolor='white', elinewidth=0.5, ms=1, color='white', linewidth=0.5)
            elif errbars == False:
                plt.errorbar(fs_mode[::1], vs_mode[::1], fmt=f"o-", elinewidth=0.5, ms=1.5, color='white', linewidth=0.5)
                
    if len(pred_modes) > 0 :
        for mode in pred_modes:
            if len(mode.shape) == 1:
                mode = mode.reshape(1,-1)
            fs_mode = mode[:,0]
            vs_mode = mode[:,1]
            plt.errorbar(fs_mode, vs_mode, fmt=f"o", elinewidth=0.5, ms=2, color='tab:orange', linewidth=1.0)
            
    if len(full_pred_modes) > 0 :
        for mode in full_pred_modes:
            if len(mode.shape) == 1:
                mode = mode.reshape(1,-1)
            fs_mode = mode[:,0]
            vs_mode = mode[:,1]
            plt.plot(fs_mode, vs_mode, color='tab:orange', linewidth=1.0)
            
    if dx is not None:
        lambda_min = 2*dx
        vs_min = fs * lambda_min
        vs_min[vs_min > np.max(vs)] = np.nan
        vs_min[vs_min < np.min(vs)] = np.nan
        plt.plot(fs, vs_min, color='grey', linestyle='--', linewidth=1.0)
        if Nx is not None:
            lambda_max = 6*(Nx-1)*dx
            vs_max = fs * lambda_max
            vs_max[vs_max > np.max(vs)] = np.nan
            vs_max[vs_max < np.min(vs)] = np.nan
            plt.plot(fs, vs_max, color='grey', linestyle='--', linewidth=1.0)
            
    plt.ylim((vs[0], vs[-1]))
    plt.xlim((fs[0], fs[-1]))
                        
    plt.savefig(path, format='svg', dpi='figure', bbox_inches='tight')
    plt.close()
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def display_inverted_section(vs_section, std_section, positions, depths, path, zmin=None, zmax=None, cmap='terrain'):
    vs_section = np.copy(vs_section)
    std_section = np.copy(std_section)
    positions = np.copy(positions)
    depths = np.copy(depths)
    
    if zmin is not None:
        vs_section[vs_section < zmin] = zmin
    if zmax is not None:
        vs_section[vs_section > zmax] = zmax
        
    if zmin is None:
        zmin = np.nanmin(vs_section)
    if zmax is None:
        zmax = np.nanmax(vs_section)
    
    cmap = cmap.lower()
    
    fig, axs = plt.subplots(2, 1, figsize=(18*CM, 10*CM), dpi=_DPI)
    
    im0 = axs[0].pcolormesh(positions, depths, vs_section.T, cmap=cmap, rasterized=True, vmin=zmin, vmax=zmax)
    axs[0].set_ylabel("Depth [m]")
    axs[0].invert_yaxis()
    axs[0].minorticks_on()
    axs[0].set_xticklabels([])
    axs[0].set_xlabel("")
    axs[0].xaxis.set_ticks_position('both')
    axs[0].yaxis.set_ticks_position('both')
    cbar0 = fig.colorbar(im0, ax=axs[0], orientation='vertical', pad=0.025)
    cbar0.set_label("$v_{S}$ [m/s]")
    cbar0.ax.minorticks_on()
    
    vmax = np.max(std_section)
    im1 = axs[1].pcolormesh(positions, depths, std_section.T, cmap='afmhot_r', rasterized=True, vmin=0, vmax=vmax)
    axs[1].set_xlabel("Position [m]")
    axs[1].set_ylabel("Depth [m]")
    axs[1].invert_yaxis()
    axs[1].minorticks_on()
    axs[1].xaxis.set_ticks_position('both')
    axs[1].yaxis.set_ticks_position('both')
    cbar1 = fig.colorbar(im1, ax=axs[1], orientation='vertical', pad=0.025)
    cbar1.set_label("Std [m/s]")
    cbar1.ax.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(path, format='svg', dpi='figure', bbox_inches='tight')
    plt.close()
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def display_pseudo_sections(obs_section, pred_section, fs, positions, path, zmin=None, zmax=None):
    obs_section = np.copy(obs_section)
    pred_section = np.copy(pred_section)
    fs = np.copy(fs)
    positions = np.copy(positions)
        
    if zmin is not None:
        obs_section[obs_section < zmin] = zmin
        pred_section[pred_section < zmin] = zmin
    if zmax is not None:
        obs_section[obs_section > zmax] = zmax
        pred_section[pred_section > zmax] = zmax
        
    if zmin is None:
        zmin = np.nanmin([np.nanmin(obs_section), np.nanmin(pred_section)])
    if zmax is None:
        zmax = np.nanmax([np.nanmax(obs_section), np.nanmax(pred_section)])

    fig, axs = plt.subplots(3, 1, figsize=(18*CM, 15*CM), dpi=_DPI)
    
    im0 = axs[0].pcolormesh(positions, fs, np.flipud(obs_section.T), cmap='viridis', rasterized=True, vmin=zmin, vmax=zmax)
    axs[0].set_ylabel("Frequency [Hz]")
    axs[0].minorticks_on()
    axs[0].set_xticklabels([])
    axs[0].set_xlabel("")
    axs[0].xaxis.set_ticks_position('both')
    axs[0].yaxis.set_ticks_position('both')
    cbar0 = fig.colorbar(im0, ax=axs[0], orientation='vertical', pad=0.025)
    cbar0.set_label("Obs $v_{R}$ [m/s]")
    cbar0.ax.minorticks_on()
    
    im1 = axs[1].pcolormesh(positions, fs, np.flipud(pred_section.T), cmap='viridis', rasterized=True, vmin=zmin, vmax=zmax)
    axs[1].set_ylabel("Frequency [Hz]")
    axs[1].minorticks_on()
    axs[1].set_xticklabels([])
    axs[1].set_xlabel("")
    axs[1].xaxis.set_ticks_position('both')
    axs[1].yaxis.set_ticks_position('both')
    cbar1 = fig.colorbar(im1, ax=axs[1], orientation='vertical', pad=0.025)
    cbar1.set_label("Pred $v_{R}$ [m/s]")
    cbar1.ax.minorticks_on()
    
    if np.all(np.isnan(pred_section)) or np.all(np.isnan(obs_section)):
        res_section = np.full_like(obs_section, np.nan)
        lim = 0
    else:
        res_section = (pred_section - obs_section) / pred_section * 100
        vmin = np.nanmin(res_section)
        vmax = np.nanmax(res_section)
        lim = max(abs(vmin), abs(vmax))
    im2 = axs[2].pcolormesh(positions, fs, np.flipud(res_section.T), cmap='bwr', rasterized=True, vmin=-lim, vmax=lim)
    axs[2].set_xlabel("Position [m]")
    axs[2].set_ylabel("Frequency [Hz]")
    axs[2].minorticks_on()
    axs[2].xaxis.set_ticks_position('both')
    axs[2].yaxis.set_ticks_position('both')
    cbar2 = fig.colorbar(im2, ax=axs[2], orientation='vertical', pad=0.025)
    cbar2.set_label("Residuals [%]")
    cbar2.ax.minorticks_on()
    
    fig.savefig(path, format='svg', dpi='figure', bbox_inches='tight')
    plt.close()
### -----------------------------------------------------------------------------------------------