#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import xp, parallel_array, img_fft
from arrlp import kernel as _kernel



# %% Function
def img_wiener(array, *,
        stacks=False, channels=False, parallel=False, cuda=False, print=None,
        kernel=None, power=2, balance=10**2, **kernel_kwargs) :
    '''
    Makes Wiener deconvolution on images.
    '''

    # Checks
    if parallel and not stacks and not channels :
        raise ValueError('Normal array (no stack of channel) cannot be calculated in parallel')
    if parallel and cuda :
        raise SyntaxError('Cuda and Parallel cannot be True at the same time')
    if parallel :
        import warnings
        warnings.warn('Parallel optimization is not effective in this function')
    # if cuda :
    #     import warnings
    #     warnings.warn('Cuda optimization is not effective in this function')
        

    # Init
    _xp = xp(cuda)
    array = _xp.asarray(array)
    out = _xp.empty_like(array)

    # Function info [update here]
    ndims = 2
    ins = (array,)
    outs = (out,)

    # Init
    nstacks, nchannels = array.shape[0], array.shape[-1]
    iterator = print.clock if print is not None else range

    # Wiener
    y, x = array.shape[int(stacks)], array.shape[int(stacks) + 1]
    if kernel is None : kernel = _kernel(ndims, shape=(y, x), cuda=cuda, **kernel_kwargs)
    fft_kernel = _xp.abs(img_fft(kernel, cuda=cuda))
    W = fft_kernel**(power-1) / (fft_kernel**power + fft_kernel.max() / balance)
    def func(array) :
        dtype = array.dtype
        fft = _xp.fft.fftshift(_xp.fft.fft2(array))
        return _xp.real(_xp.fft.ifft2(_xp.fft.ifftshift(fft * W))).astype(dtype)


    # Looping on axes
    if cuda or not parallel :

        match (stacks, channels) :
            case (False, False) :
                out[:] = func(array)
            case (True, False) :
                for i in iterator(nstacks) :
                    out[i] = func(array[i])
            case (False, True) :
                for j in iterator(nchannels) :
                    out[..., j] = func(array[..., j])
            case (True, True) :
                for i in iterator(nstacks) :
                    a, o = array[i], out[i]
                    for j in range(nchannels) :
                        o[..., j] = func(a[..., j])
        return out
    
    # Parallel
    if print is not None : print('Looping in parallel... [ETA not available]')

    match (stacks, channels) :

        case (True, False) :
            return parallel_array(func, *ins, outs=outs, stacks=True)

        case (False, True) :
            return parallel_array(func, *ins, outs=outs, stacks=False)

        case (True, True) :
            def function(array) :
                out = _xp.empty_like(array)
                for j in range(nchannels) :
                    out[..., j] = func(array[..., j])
                return out
            return parallel_array(function, *ins, outs=outs, stacks=True)
    




if __name__ == '__main__' :
    from arrlp import debug_array
    func = img_wiener

    # Arguments
    kwargs = dict(
        pixel=1,
        sigma=3,
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)

