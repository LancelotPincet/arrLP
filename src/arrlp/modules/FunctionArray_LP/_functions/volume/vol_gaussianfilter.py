#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import kernel, vol_correlate1d



# %% Function
def vol_gaussianfilter(array, *, sigma:float, pixel:float=1., out=None, # Arrays
        stacks=False, channels=False, parallel=False, cuda=False, test=False, iterator=range, # Modes
        **kwargs) :

    try :
        if len(pixel) != 3 :
            raise ValueError('Argument does not have good number of dimensions')
    except TypeError :
        pixel = (pixel, pixel, pixel)
    try :
        if len(sigma) != 3 :
            raise ValueError('Argument does not have good number of dimensions')
    except TypeError :
        sigma = (sigma, sigma, sigma)
    k = kernel(ndims=1, pixel=pixel[0], sigma=sigma[0]), kernel(ndims=1, pixel=pixel[1], sigma=sigma[1]), kernel(ndims=1, pixel=pixel[2], sigma=sigma[2])
    return vol_correlate1d(array, kernel=k, out=out, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, test=test, iterator=iterator, **kwargs)
vol_gaussianfilter.ndims = 3



if __name__ == '__main__' :
    from arrlp import debug_array
    func = vol_gaussianfilter

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
