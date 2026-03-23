#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_vol_minifilter(self, array, *, footprint, **kwargs) :
    return dict(
        footprint=self.xp.asarray(footprint),
    )

def out_vol_minifilter(self, array, **kwargs) :
    return self.xp.empty_like(array, dtype=self.xp.float32)

def _vol_minifilter(self, out, array, *, footprint, mode='nearest', **kwargs) :
    return self.ndimage.minimum_filter(array, footprint=footprint, axes=self.axes, output=out, mode=mode, **kwargs)

def par_vol_minifilter(self, array, *, footprint, mode='nearest', **kwargs) :
    return self.ndimage.minimum_filter(array, footprint=footprint, mode=mode, **kwargs)



# Main function
vol_minifilter = FunctionArray(
    
    # Mandatory
    ndims = 3,
    cpu_function = _vol_minifilter,
    par_function = par_vol_minifilter,
    gpu_function = _vol_minifilter,
    out_function = out_vol_minifilter,
    ini_function = ini_vol_minifilter,

    # Loops
    par_loop = True,
    use_joblib = True, # If True, arguments of parallel function should not have "out".

    # Performances
    remove_parallel = False,
    remove_cuda = False,

)



if __name__ == '__main__' :
    from arrlp import debug_array
    from arrlp import kernel
    func = vol_minifilter

    # Arguments
    kwargs = dict(
        footprint=kernel(ndims=3, pixel=1, window=7),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
