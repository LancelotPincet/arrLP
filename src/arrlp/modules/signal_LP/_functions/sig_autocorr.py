#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def _sig_autocorr(self, out, array) :
    array  = array.copy()
    array  -= array.mean(axis=self.axes, keepdims=True)
    
    array  = self.scipyx.fft.rfft(array,  axis=self.axes[0]) # release previous array
    array  = self.xp.abs(array)**2
    array = self.scipyx.fft.irfft(array, axis=self.axes[0]) # release previous array    
    return self.scipyx.fft.fftshift(array, axes=self.axes)

def par_sig_autocorr(self, out, array) :
    array  = array.copy()
    array  -= array.mean(axis=self.axes, keepdims=True)
    
    array  = self.scipyx.fft.rfft(array,  axis=self.axes[0], workers=self.parallel) # release previous array
    array  = self.xp.abs(array)**2
    array = self.scipyx.fft.irfft(array, axis=self.axes[0], workers=self.parallel) # release previous array    
    return self.scipyx.fft.fftshift(array, axes=self.axes)



# Main function
sig_autocorr = FunctionArray(
    
    # Mandatory
    ndims = 1,
    cpu_function = _sig_autocorr,
    par_function = par_sig_autocorr,
    gpu_function = _sig_autocorr,
    out_function = None,
    ini_function = None,

    # Loops
    use_joblib = False, # If True, arguments of parallel function should not have "out".

    # Performances
    remove_parallel = False,
    remove_cuda = False,

)



if __name__ == '__main__' :
    from arrlp import debug_array
    func = sig_autocorr

    # Arguments
    kwargs = dict(
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
