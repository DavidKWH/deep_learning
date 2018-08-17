# [DONE] include support for QAM and PSK

from numpy import sqrt
from numpy.random import randn

def crandn(*args):
    samps = sqrt(0.5) * (randn(*args) + randn(*args) * 1j)
    return samps

import numpy as np
import numpy.random as rnd
import commpy as comm
from commpy import utilities as util
import matplotlib.pyplot as plt

def scatterplot(x_cplx):
    return plt.scatter(x_cplx.real, x_cplx.imag, s=10)

def get_const_symbols(mode='QAM', m=16):
    ''' Implement either QAM or PSK '''

    # m is given
    nbps = np.log2(m)

    # simplify constellations to 2^k alphabets
    if nbps.is_integer():
        nbps = nbps.astype(int)
    else:
        raise Exception('QAM: m must be a power of 2')

    # switch between modes
    if mode == 'QAM':

        blist = [util.dec2bitarray(x,nbps) for x in range(m)]
        bits = np.concatenate(blist)

        mod = comm.modulation.QAMModem(m)
        syms = mod.modulate(bits) / np.sqrt(10)
        return syms

    elif mode == 'PSK':

        thetas = [x/m * 2 * np.pi for x in range(m)]
        syms = np.exp(1j*np.asarray(thetas))
        return syms

    else:
        raise Exception('Invalid initialization!')

def get_noisy_syms(nsyms, std=.01):
    nbps = 4
    m = 2**nbps
    nbits = nsyms * nbps
    scale = std

    mod = comm.modulation.QAMModem(m)

    bits = rnd.randint(2, size=nbits)
    syms = mod.modulate(bits) / np.sqrt(10)

    # add noise
    syms_n = syms + scale * crandn(*syms.shape)
    return syms_n
