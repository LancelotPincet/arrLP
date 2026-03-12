#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def _vol_autocorr(self, out, array) :
    array  = array.copy()
    array  -= array.mean(axis=self.axes, keepdims=True)
    
    array = self.scipyx.fft.rfftn(array,  axes=self.axes) # release previous array
    array  = self.xp.abs(array)**2
    array = self.scipyx.fft.irfftn(array, axes=self.axes) # release previous array    
    return self.scipyx.fft.fftshift(array, axes=self.axes)

def par_vol_autocorr(self, out, array) :
    array  = array.copy()
    array  -= array.mean(axis=self.axes, keepdims=True)
    
    array = self.scipyx.fft.rfftn(array,  axes=self.axes, workers=self.parallel) # release previous array
    array  = self.xp.abs(array)**2
    array = self.scipyx.fft.irfftn(array, axes=self.axes, workers=self.parallel) # release previous array    
    return self.scipyx.fft.fftshift(array, axes=self.axes)



# Main function
vol_autocorr = FunctionArray(
    
    # Mandatory
    ndims = 3,
    cpu_function = _vol_autocorr,
    par_function = par_vol_autocorr,
    gpu_function = _vol_autocorr,
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
    func = vol_autocorr

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
