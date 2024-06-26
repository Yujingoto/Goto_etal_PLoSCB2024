import cupy as cp
import numpy as np
from scipy.signal import bilinear, lfilter

def A_weighting(signal, fs):
    """Design of an A-weighting filter.
    b, a = A_weighting(fs) designs a digital A-weighting filter for
    sampling frequency `fs`. Usage: y = scipy.signal.lfilter(b, a, x).
    Warning: `fs` should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.
    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2*np.pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    DENs = np.polymul([1, 4*np.pi * f4, (2*np.pi * f4)**2],
                   [1, 4*np.pi * f1, (2*np.pi * f1)**2])
    DENs = np.polymul(np.polymul(DENs, [1, 2*np.pi * f3]),
                                 [1, 2*np.pi * f2])

    b,a = bilinear(NUMs, DENs, fs)
    return cp.array(lfilter(b,a,cp.asnumpy(signal)))