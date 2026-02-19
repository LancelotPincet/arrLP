#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_vol_correlate(self, array, kernel, **kwargs) :
    return dict(
        kernel=self.xp.asarray(kernel),
    )

def out_vol_correlate(self, array, **kwargs) :
    return self.xp.empty_like(array)

def _vol_correlate(self, out, array, kernel, mode='nearest', **kwargs) :
    return self.ndimage.correlate(array, weights=kernel, axes=self.axes, output=out, mode=mode, **kwargs)

def par_vol_correlate(self, array, kernel, mode='nearest', **kwargs) :
    return self.ndimage.correlate(array, weights=kernel, mode=mode, **kwargs)



# Main function
vol_correlate = FunctionArray(
    
    # Mandatory
    ndims = 3,
    cpu_function = _vol_correlate,
    par_function = par_vol_correlate,
    gpu_function = _vol_correlate,
    out_function = out_vol_correlate,
    ini_function = ini_vol_correlate,

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
    func = vol_correlate

    # Arguments
    kwargs = dict(
        kernel=kernel(ndims=3, pixel=1, sigma=3),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
