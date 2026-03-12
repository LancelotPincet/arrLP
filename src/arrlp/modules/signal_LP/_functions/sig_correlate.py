#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_sig_correlate(self, array, kernel, **kwargs) :
    return dict(
        kernel=self.xp.asarray(kernel),
    )

def out_sig_correlate(self, array, **kwargs) :
    return self.xp.empty_like(array, dtype=self.xp.float32)

def _sig_correlate(self, out, array, kernel, mode='nearest', **kwargs) :
    return self.ndimage.correlate1d(array, weights=kernel, axis=self.axes[0], output=out, mode=mode, **kwargs)

def par_sig_correlate(self, array, kernel, mode='nearest', **kwargs) :
    return self.ndimage.correlate1d(array, weights=kernel, mode=mode, **kwargs)



# Main function
sig_correlate = FunctionArray(
    
    # Mandatory
    ndims = 1,
    cpu_function = _sig_correlate,
    par_function = par_sig_correlate,
    gpu_function = _sig_correlate,
    out_function = out_sig_correlate,
    ini_function = ini_sig_correlate,

    # Loops
    par_loop = True,
    use_joblib = True, # If True, arguments of parallel function should not have "out".

    # Performances
    remove_parallel = True,
    remove_cuda = False,

)



if __name__ == '__main__' :
    from arrlp import debug_array
    from arrlp import kernel
    func = sig_correlate

    # Arguments
    kwargs = dict(
        kernel=kernel(ndims=1, pixel=1, sigma=3),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
