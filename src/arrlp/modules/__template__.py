#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import xp, scipyx, ndimage, axes, parallel_array



# %% Function
def temp(array, *,
        stacks=False, channels=False, parallel=False, cuda=False, print=None,
        **kwargs) :
    '''
    TODO for signals/images/volumes.
    '''

    # Checks
    if parallel and not stacks and not channels :
        raise ValueError('Normal array (no stack of channel) cannot be calculated in parallel')
    if parallel and cuda :
        raise SyntaxError('Cuda and Parallel cannot be True at the same time')
    # if parallel :
    #     import warnings
    #     warnings.warn('Parallel optimization is not effective in this function')
    # if cuda :
    #     import warnings
    #     warnings.warn('Cuda optimization is not effective in this function')
        

    # Init
    ndims = 2
    _xp = xp(cuda)
    _scipyx = scipyx(cuda)
    _ndimage = ndimage(cuda)
    _axes = axes(ndims)
    array = _xp.asarray(array)
    if out is None: out = _xp.empty_like(array)

    # Function info
    func = lambda array, *args, axes=None, **kwargs : _scipyx.fft.fftshift(_scipyx.fft.fft2(array, *args, axes=axes, workers=-1, **kwargs), axes=axes)
    ndims = 2
    ins = (array,)
    outs = (None,)

    # Init
    nstacks, nchannels = array.shape[0], array.shape[-1]
    iterator = print.clock if print is not None else range

    # Looping on axes
    if cuda or not parallel :
        _axes = axes(ndims, stacks)
        return func(array, axes=_axes, **kwargs) # TODO
    
        match (stacks, channels) :
            case (False, False) :
                func(array, out=out)
            case (True, False) :
                for i in iterator(nstacks) :
                    func(array[i], out=out[i])
            case (False, True) :
                for j in iterator(nchannels) :
                    func(array[..., j], out=out[..., j])
            case (True, True) :
                for i in iterator(nstacks) :
                    a, o = array[i], out[i]
                    for j in range(nchannels) :
                        func(a[..., j], out=o[..., j])
        return out

    # Parallel

    match (stacks, channels) :

        case (True, False) :
            return parallel_array(func, *ins, outs=outs, stacks=True, **kwargs)

        case (False, True) :
            return parallel_array(func, *ins, outs=outs, stacks=False, **kwargs)

        case (True, True) :
            _axes = axes(ndims, False)
            return parallel_array(func, *ins, outs=outs, stacks=True, axes=_axes, **kwargs)
    




if __name__ == '__main__' :
    from arrlp import debug_array
    func = TODO

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
