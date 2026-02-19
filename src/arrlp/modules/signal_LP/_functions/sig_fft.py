#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def _sig_fft(self, out, array, **kwargs) :
    return self.scipyx.fft.fftshift(self.scipyx.fft.fft(array, axis=self.axes[0], **kwargs), axes=self.axes)

def par_sig_fft(self, out, array, **kwargs) :
    return self.scipyx.fft.fftshift(self.scipyx.fft.fft(array, axis=self.axes[0], workers=-1, **kwargs), axes=self.axes)



# Main function
sig_fft = FunctionArray(
    
    # Mandatory
    ndims = 1,
    cpu_function = _sig_fft,
    par_function = par_sig_fft,
    gpu_function = _sig_fft,
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
    func = sig_fft

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
