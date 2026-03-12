#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_img_correlate(self, array, kernel, **kwargs) :
    return dict(
        kernel=self.xp.asarray(kernel),
    )

def out_img_correlate(self, array, **kwargs) :
    return self.xp.empty_like(array, dtype=self.xp.float32)

def _img_correlate(self, out, array, kernel, mode='nearest', **kwargs) :
    return self.ndimage.correlate(array, weights=kernel, axes=self.axes, output=out, mode=mode, **kwargs)

def par_img_correlate(self, array, kernel, mode='nearest', **kwargs) :
    return self.ndimage.correlate(array, weights=kernel, mode=mode, **kwargs)



# Main function
img_correlate = FunctionArray(
    
    # Mandatory
    ndims = 2,
    cpu_function = _img_correlate,
    par_function = par_img_correlate,
    gpu_function = _img_correlate,
    out_function = out_img_correlate,
    ini_function = ini_img_correlate,

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
    func = img_correlate

    # Arguments
    kwargs = dict(
        kernel=kernel(ndims=2, pixel=1, sigma=3),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
