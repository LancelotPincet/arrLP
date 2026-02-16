#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import kernel, img_correlate1d



# %% Function
def img_gaussianfilter(array, sigma:float, pixel:float=1., out=None, # Arrays
        stacks=False, channels=False, parallel=False, cuda=False, test=False, iterator=range, # Modes
        **kwargs) :

    try :
        if len(pixel) != 2 :
            raise ValueError('Argument does not have good number of dimensions')
    except TypeError :
        pixel = (pixel, pixel)
    try :
        if len(sigma) != 2 :
            raise ValueError('Argument does not have good number of dimensions')
    except TypeError :
        sigma = (sigma, sigma)
    k = kernel(ndims=1, pixel=pixel[0], sigma=sigma[0]), kernel(ndims=1, pixel=pixel[1], sigma=sigma[1])
    return img_correlate1d(array, out=out, kernel=k, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, test=test, iterator=iterator, **kwargs)
img_gaussianfilter.ndims = 2



if __name__ == '__main__' :
    from arrlp import debug_array
    func = img_gaussianfilter

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
