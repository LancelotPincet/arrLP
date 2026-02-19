#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def _vol_fft(self, out, array, **kwargs) :
    return self.scipyx.fft.fftshift(self.scipyx.fft.fftn(array, axes=self.axes, **kwargs), axes=self.axes)

def par_vol_fft(self, out, array, **kwargs) :
    return self.scipyx.fft.fftshift(self.scipyx.fft.fftn(array, axes=self.axes, workers=-1, **kwargs), axes=self.axes)



# Main function
vol_fft = FunctionArray(
    
    # Mandatory
    ndims = 3,
    cpu_function = _vol_fft,
    par_function = par_vol_fft,
    gpu_function = _vol_fft,
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
    func = vol_fft

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
