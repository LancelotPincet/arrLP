#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import arr_function
import numpy as np
try :
    import cupy as cp
except ImportError :
    cp = None



# %% Function
def arr_wrapper(array, *, out=None, # Arrays
        stacks=False, channels=False, parallel=False, cuda=False, test=False, iterator=range, # Modes
        **kwargs) :

    xp = cp if cuda and cp is not None else np
        
    return arr_function(array, out=out, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, test=test, iterator=iterator, **kwargs)
arr_wrapper.ndims = 2



if __name__ == '__main__' :
    from arrlp import debug_array
    func = arr_wrapper

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
