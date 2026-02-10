#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import xp, axes, parallel_array



# %% Function
def img_ifft(array, *,
        stacks=False, channels=False, parallel=False, cuda=False, print=None,
        **kwargs) :
    '''
    Fast Inverse Fourier Transform on images.
    '''

    # Checks
    if parallel and not stacks and not channels :
        raise ValueError('Normal array (no stack of channel) cannot be calculated in parallel')
    if parallel :
        import warnings
        warnings.warn('Parallel optimization is not effective in this function')
    # if cuda :
    #    import warnings
    #    warnings.warn('Cuda optimization is not effective in this function')
        

    # Init
    _xp = xp(cuda)
    array = _xp.asarray(array)

    # Function info [update here]
    func = lambda array, *args, axes=None, **kwargs : _xp.fft.ifft2(_xp.fft.ifftshift(array, axes=axes), *args, axes=axes, **kwargs)
    ndims = 2
    ins = (array,)
    outs = (None,)

    # Looping on axes
    if cuda or not parallel :
        _axes = axes(ndims, stacks)
        return func(array, axes=_axes, **kwargs)
    
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



    # Imports
    import numpy as np
    try :
        import cupy as cp
    except ImportError :
        cp = None
    from time import perf_counter



    # Parameter ~2**24 ; 2**8=256
    func = img_ifft
    nstacks = int(2**8)
    shape = (int(2**7), int(2**7))
    nchannels = int(2**2)



    # Timeit function
    def timeit(array, stacks, channels, parallel, cuda) :

        # Init
        _xp = xp(cuda)
        array = _xp.asarray(array)

        # Calculate
        print(f'\n** Testing stacks={stacks}, channels={channels}, parallel={parallel}, cuda={cuda} **')
        print('Compile run...')
        func(array, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda)
        print('Run...')
        tic = perf_counter()
        out = func(array, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda)
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
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg="Outputs differ between optimizations"
                )

                
