#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import img_correlate, get_xp



# %% Function
def img_convolve(array, *, kernel, out=None, # Arrays
        stacks=False, channels=False, parallel=False, cuda=False, test=False, iterator=range, # Modes
        **kwargs) :

    xp = get_xp(cuda)
    for dim in range(2) :
        kernel = xp.flip(kernel, axis=dim)
        
    return img_correlate(array, kernel=kernel, out=out, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, test=test, iterator=iterator, **kwargs)
img_convolve.ndims = 2



if __name__ == '__main__' :
    from arrlp import debug_array
    from arrlp import kernel
    func = img_convolve

    # Arguments
    kwargs = dict(
        kernel=kernel(ndims=2, sigma=3),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
