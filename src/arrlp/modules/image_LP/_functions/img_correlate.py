#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import xp, ndimage, parallel_array
from arrlp import kernel as _kernel



# %% Function
def img_correlate(array, *,
        stacks=False, channels=False, parallel=False, cuda=False, print=None,
        kernel=None, out=None, **kernel_kwargs) :
    '''
    Makes correlation on images.
    '''

    # Checks
    if parallel and not stacks and not channels :
        raise ValueError('Normal array (no stack of channel) cannot be calculated in parallel')
    # if parallel :
    #     import warnings
    #     warnings.warn('Parallel optimization is not effective in this function')
    # if cuda :
    #     import warnings
    #     warnings.warn('Cuda optimization is not effective in this function')
        

    # Init
    _xp = xp(cuda)
    _ndimage = ndimage(cuda)
    array = _xp.asarray(array)
    if out is None: out = _xp.empty_like(array)

    # Function info [update here]
    func = _ndimage.correlate
    ndims = 2
    ins = (array,)
    outs = (out,)

    # Init
    if kernel is None : kernel = _kernel(ndims, cuda=cuda, **kernel_kwargs)
    nstacks, nchannels = array.shape[0], array.shape[-1]
    iterator = print.clock if print is not None else range

    # Looping on axes
    if cuda or not parallel :

        match (stacks, channels) :
            case (False, False) :
                func(array, weights=kernel, output=out, mode='constant')
            case (True, False) :
                for i in iterator(nstacks) :
                    func(array[i], weights=kernel, output=out[i], mode='constant')
            case (False, True) :
                for j in iterator(nchannels) :
                    func(array[..., j], weights=kernel, output=out[..., j], mode='constant')
            case (True, True) :
                for i in iterator(nstacks) :
                    a, o = array[i], out[i]
                    for j in range(nchannels) :
                        func(a[..., j], weights=kernel, output=o[..., j], mode='constant')
        return out
    
    # Parallel
    if print is not None : print('Looping in parallel... [ETA not available]')

    match (stacks, channels) :

        case (True, False) :
            return parallel_array(func, *ins, outs=outs, stacks=True, weights=kernel, mode='constant')

        case (False, True) :
            return parallel_array(func, *ins, outs=outs, stacks=False, weights=kernel, mode='constant')

        case (True, True) :
            def function(array) :
                out = np.empty_like(array)
                for j in range(nchannels) :
                    func(array[..., j], weights=kernel, output=out[..., j], mode='constant')
                return out
            return parallel_array(function, *ins, outs=outs, stacks=True)
    




if __name__ == '__main__' :



    # Imports
    import numpy as np
    try :
        import cupy as cp
    except ImportError :
        cp = None
    from time import perf_counter



    # Parameter ~2**24 ; 2**8=256
    func = img_correlate
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
        func(array, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, pixel=1, sigma=3)
        print('Run...')
        tic = perf_counter()
        out = func(array, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, pixel=1, sigma=3)
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

                
