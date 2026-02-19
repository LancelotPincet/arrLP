#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import kernel, sig_correlate



# %% Function
def sig_gaussianfilter(array, *, sigma:float, pixel:float=1., out=None, # Arrays
        stacks=False, channels=False, parallel=False, cuda=False, test=False, iterator=range, # Modes
        **kwargs) :

    k = kernel(ndims=1, pixel=pixel, sigma=sigma)
    return sig_correlate(array, kernel=k, out=out, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, test=test, iterator=iterator, **kwargs)
sig_gaussianfilter.ndims = 1



if __name__ == '__main__' :
    from arrlp import debug_array
    func = sig_gaussianfilter

    # Arguments
    kwargs = dict(
        sigma=3.,
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
