#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def out_sig_minifilter(self, array, **kwargs) :
    return self.xp.empty_like(array, dtype=self.xp.float32)

def _sig_minifilter(self, out, array, *, size, mode='nearest', **kwargs) :
    return self.ndimage.minimum_filter1d(array, size=size, axis=self.axes[0], output=out, mode=mode, **kwargs)

def par_sig_minifilter(self, array, *, size, mode='nearest', **kwargs) :
    return self.ndimage.minimum_filter1d(array, size=size, mode=mode, **kwargs)



# Main function
sig_minifilter = FunctionArray(
    
    # Mandatory
    ndims = 1,
    cpu_function = _sig_minifilter,
    par_function = par_sig_minifilter,
    gpu_function = _sig_minifilter,
    out_function = out_sig_minifilter,
    ini_function = None,

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
    func = sig_minifilter

    # Arguments
    kwargs = dict(
        size=7,
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
