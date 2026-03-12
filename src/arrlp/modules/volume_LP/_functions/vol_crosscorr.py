#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def _vol_crosscorr(self, out, array, array2) :
    array  = array.copy()
    array2 = array2.copy()
    
    array  -= array.mean(axis=self.axes, keepdims=True)
    array2 -= array2.mean(axis=self.axes, keepdims=True)
    array  /= array.std(axis=self.axes, keepdims=True)
    array2 /= array2.std(axis=self.axes, keepdims=True)
    
    array  = self.scipyx.fft.rfftn(array,  axes=self.axes) # release previous array
    array2 = self.scipyx.fft.rfftn(array2, axes=self.axes) # release previous array2
    self.xp.conj(array2, out=array2)
    array *= array2
    del(array2) # release previous array 2
    array = self.scipyx.fft.irfftn(array, axes=self.axes) # release previous array    
    return self.scipyx.fft.fftshift(array, axes=self.axes)

def par_vol_crosscorr(self, out, array, array2) :
    array  = array.copy()
    array2 = array2.copy()
    
    array  -= array.mean(axis=self.axes, keepdims=True)
    array2 -= array2.mean(axis=self.axes, keepdims=True)
    array  /= array.std(axis=self.axes, keepdims=True)
    array2 /= array2.std(axis=self.axes, keepdims=True)
    
    array  = self.scipyx.fft.rfftn(array,  axes=self.axes, workers=self.parallel) # release previous array
    array2 = self.scipyx.fft.rfftn(array2, axes=self.axes, workers=self.parallel) # release previous array2
    self.xp.conj(array2, out=array2)
    array *= array2
    del(array2) # release previous array 2
    array = self.scipyx.fft.irfftn(array, axes=self.axes, workers=self.parallel) # release previous array    
    return self.scipyx.fft.fftshift(array, axes=self.axes)



# Main function
vol_crosscorr = FunctionArray(
    
    # Mandatory
    ndims = 3,
    cpu_function = _vol_crosscorr,
    par_function = par_vol_crosscorr,
    gpu_function = _vol_crosscorr,
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
    func = vol_crosscorr

    # Arguments
    kwargs = dict(
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, narrays=2, **kwargs)
