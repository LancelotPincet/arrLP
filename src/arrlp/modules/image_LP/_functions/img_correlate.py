#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import xp, ndimage, parallel_array, axes
from arrlp import kernel as _kernel
import numpy as np



# %% Function
def img_correlate(array, *,
        stacks=False, channels=False, parallel=False, cuda=False, print=None,
        kernel=None, out=None, separate_tol=1e-10, **kernel_kwargs) :
    '''
    Makes correlation on images.
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

    # Check separability
    if separate_tol is not None :
        U, S, Vt = np.linalg.svd(kernel, full_matrices=False)
        if np.sum(S > separate_tol) == 1:
            ky = Vt[0, :] * np.sqrt(S[0])
            kx = U[:, 0] * np.sqrt(S[0])
            kernel = ky, kx
            func = lambda array, weights, mode, axes=None, output=None : _ndimage.correlate1d(_ndimage.correlate1d(array, weights=weights[0], output=output, axis=axes[0], mode=mode), weights=weights[1], output=output, axis=axes[1], mode=mode)

    # Looping on axes
    if not cuda and not parallel :
        _axes = axes(ndims, stacks)
        return func(array, output=out, weights=kernel, axes=_axes, mode='constant')

    # Init
    nstacks, nchannels = array.shape[0], array.shape[-1]
    iterator = print.clock if print is not None else range

    if cuda :
        match (stacks, channels) :
            case (False, False) :
                func(array, output=out, weights=kernel, mode='constant')
            case (True, False) :
                for i in iterator(nstacks) :
                    func(array[i], output=out[i], weights=kernel, mode='constant')
            case (False, True) :
                for j in iterator(nchannels) :
                    func(array[..., j], output=out[..., j], weights=kernel, mode='constant')
            case (True, True) :
                for i in iterator(nstacks) :
                    a, o = array[i], out[i]
                    for j in range(nchannels) :
                        func(a[..., j], output=o[..., j], weights=kernel, mode='constant')
        return out


    # Parallel
    if print is not None : print('Looping in parallel... [ETA not available]')

    match (stacks, channels) :

        case (True, False) :
            return parallel_array(func, *ins, outs=outs, stacks=True, weights=kernel, mode='constant', axes=(0, 1))

        case (False, True) :
            return parallel_array(func, *ins, outs=outs, stacks=False, weights=kernel, mode='constant', axes=(0, 1))

        case (True, True) :
            def function(array) :
                out = np.empty_like(array)
                for j in range(nchannels) :
                    func(array[..., j], weights=kernel, output=out[..., j], mode='constant', axes=(0, 1))
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

    # Arguments
    kwargs = dict(
        pixel=1, 
        sigma=6,
        separate_tol=1e-10,
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

                
