#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_img_maxifilter(self, array, *, footprint, **kwargs) :
    return dict(
        footprint=self.xp.asarray(footprint),
    )

def out_img_maxifilter(self, array, **kwargs) :
    return self.xp.empty_like(array, dtype=self.xp.float32)

def _img_maxifilter(self, out, array, *, footprint, mode='nearest', **kwargs) :
    return self.ndimage.maximum_filter(array, footprint=footprint, axes=self.axes, output=out, mode=mode, **kwargs)

def par_img_maxifilter(self, array, *, footprint, mode='nearest', **kwargs) :
    return self.ndimage.maximum_filter(array, footprint=footprint, mode=mode, **kwargs)



# Main function
img_maxifilter = FunctionArray(
    
    # Mandatory
    ndims = 2,
    cpu_function = _img_maxifilter,
    par_function = par_img_maxifilter,
    gpu_function = _img_maxifilter,
    out_function = out_img_maxifilter,
    ini_function = ini_img_maxifilter,

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
    func = img_maxifilter

    # Arguments
    kwargs = dict(
        footprint=kernel(ndims=2, pixel=1, window=7),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
