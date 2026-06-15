"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""

import numpy as np
from obspy.core import AttribDict
from obspy.core.stream import Stream
from obspy.core.trace import Trace, Stats
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.segy.segy import SEGYBinaryFileHeader



### -----------------------------------------------------------------------------------------------
def stream_to_array(stream, Nx, Nt):
    """
    Transform stream from obspy (Stream object) to numpy array in order to plot them
    """
    array = np.zeros((Nt,Nx)) # le champ d'onde est une matrice (Nx*Nt)
    for i, trace in enumerate(stream):
        array[:,i] = trace.data
    return array
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def array_to_stream(array, dt, offsets):
    """
    Fills a stream with an array
    """
    startTime = ["2023", "01", "01", "00", "00", "-2", "000000"] # list [yyyy, mm, dd, hh, mm, ss, ssssss]
    Nt,_ = array.shape
    stream = Stream()
    str_startTime = f"{startTime[0]}-{startTime[1]}-{startTime[2]}T{startTime[3]}:{startTime[4]}:{startTime[5]}.{startTime[6]}Z"
    for i, col in enumerate(array.T):
        stats = Stats()
        stats.starttime = UTCDateTime(str_startTime)
        stats.sampling_rate = 1/dt
        stats.delta = dt
        stats.npts = Nt
        stats.calib = 4.2704e-05
        stats.distance = offsets[i]
        trace = Trace(data=np.require(col, dtype=np.float32), header=stats)
        stream.append(trace)
        stream.stats = AttribDict()
    stream.stats.textual_file_header = 'Textual Header!'
    stream.stats.binary_file_header = SEGYBinaryFileHeader()
    stream.stats.binary_file_header.trace_sorting_code = 5
    return stream
### -----------------------------------------------------------------------------------------------