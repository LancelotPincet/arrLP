#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import xp, scipyx, axes



# %% Function
def img_fft(array, *,
        stacks=False, channels=False, parallel=False, cuda=False, print=None,
        **kwargs) :
    '''
    Fast Fourier Transform on images.
    '''

    # Checks
    if parallel and not stacks and not channels :
        raise ValueError('Normal array (no stack of channel) cannot be calculated in parallel')
    if parallel and cuda :
        raise SyntaxError('Cuda and Parallel cannot be True at the same time')
    # if parallel :
    #    import warnings
    #    warnings.warn('Parallel optimization is not effective in this function')
    # if cuda :
    #    import warnings
    #    warnings.warn('Cuda optimization is not effective in this function')
        

    # Init
    _xp = xp(cuda)
    _scipyx = scipyx(cuda)
    array = _xp.asarray(array)

    # Function info [update here]
    if parallel :
        func = lambda array, *args, axes=None, **kwargs : _scipyx.fft.fftshift(_scipyx.fft.fft2(array, *args, axes=axes, workers=-1, **kwargs), axes=axes)
    else :
        func = lambda array, *args, axes=None, **kwargs : _scipyx.fft.fftshift(_scipyx.fft.fft2(array, *args, axes=axes, **kwargs), axes=axes)
    ndims = 2

    # Looping on axes
    _axes = axes(ndims, stacks)
    return func(array, axes=_axes, **kwargs)
        




if __name__ == '__main__' :
    from arrlp import debug_array
    func = img_fft

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
