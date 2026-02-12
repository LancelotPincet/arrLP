#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import xp, scipyx, axes



# %% Function
def vol_fft(array, *,
        stacks=False, channels=False, parallel=False, cuda=False, print=None,
        **kwargs) :
    '''
    Fast Fourier Transform on volumes.
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
        func = lambda array, *args, axes=None, **kwargs : _scipyx.fft.fftshift(_scipyx.fft.fftn(array, *args, axes=axes, workers=-1, **kwargs), axes=axes)
    else :
        func = lambda array, *args, axes=None, **kwargs : _scipyx.fft.fftshift(_scipyx.fft.fftn(array, *args, axes=axes, **kwargs), axes=axes)
    ndims = 3

    # Looping on axes
    _axes = axes(ndims, stacks)
    return func(array, axes=_axes, **kwargs)
        




if __name__ == '__main__' :



    # Imports
    import numpy as np
    try :
        import cupy as cp
    except ImportError :
        cp = None
    from time import perf_counter



    # Parameter ~2**24 ; 2**8=256
    func = vol_fft
    nstacks = int(2**8)
    shape = (int(2**4), int(2**5), int(2**5))
    nchannels = int(2**2)

    # Arguments
    kwargs = dict(
    )

    # Modes
    modes = { # list of dicts with kwargs
        {},
    }

    # Timeit function
    def timeit(array, stacks, channels, parallel, cuda, **kw) :

        # Init
        _xp = xp(cuda)
        array = _xp.asarray(array)

        # Calculate
        print(f'\n** Testing stacks={stacks}, channels={channels}, parallel={parallel}, cuda={cuda} **')
        print('Compile run...')
        func(array, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, **kwargs, **kw)
        print('Run...')
        tic = perf_counter()
        out = func(array, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, **kwargs, **kw)
        toc = perf_counter()
        print(f'...took {(toc-tic)*1000:.3f}ms\n')
        if cuda : out = _xp.asnumpy(out)
        return out



    # Loops
    optimization = {'None': dict(cuda=False, parallel=False), 'Parallel': dict(cuda=False, parallel=True), 'Cuda': dict(cuda=True, parallel=False)}
    opti = ["None", "Parallel"] if cp is None else ["None", "Parallel", "Cuda"]
    for channels in [False, True] :
        for stacks in [False, True] :

            # Array
            match stacks, channels :
                case (True, True) :
                    _shape = (nstacks, *shape, nchannels)
                case(True, False) :
                    _shape = (nstacks, *shape)
                case(False, True) :
                    _shape = (*shape, nchannels)
                case(False, False) :
                    _shape = shape
            array = np.random.random(_shape)

            # Calculate
            results = {}
            for opt in opti:
                results[opt] = timeit(array, stacks, channels, **optimization[opt])
                if not channels and not stacks and opt == "None" :
                    break
            
            # Compare
            ref = results["None"]
            for opt, out in results.items():
                if opt == "None" : continue
                print(f"Checking correctness vs reference: {opt}")
                np.testing.assert_allclose(
                    ref, out,
                    rtol=1e-4,
                    atol=1e-5,
                    err_msg="Outputs differ between optimizations"
                )

                
