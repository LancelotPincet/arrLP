#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import vol_correlate
import numpy as np
try :
    import cupy as cp
except ImportError :
    cp = None



# %% Function
def vol_convolve(array, *, kernel, out=None, # Arrays
        stacks=False, channels=False, parallel=False, cuda=False, test=False, iterator=range, # Modes
        **kwargs) :

    xp = cp if cuda and cp is not None else np
    for dim in range(3) :
        kernel = xp.flip(kernel, axis=dim)
        
    return vol_correlate(array, kernel=kernel, out=out, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, test=test, iterator=iterator, **kwargs)
vol_convolve.ndims = 3



if __name__ == '__main__' :
    from arrlp import debug_array
    from arrlp import kernel
    func = vol_convolve

    # Arguments
    kwargs = dict(
        kernel=kernel(ndims=3, sigma=3),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
