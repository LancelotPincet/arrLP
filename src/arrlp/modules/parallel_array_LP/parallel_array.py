#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-09
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : parallel_array

"""
Applies a function in parallel on stacks and channels dimensions.
"""



# %% Libraries
from joblib import Parallel, delayed
from numba import njit, prange
import numpy as np



# %% Function
def parallel_array(function, *args, outs, stacks, **kwargs) :
    '''
    Applies a function in parallel on stacks and channels dimensions.
    
    Parameters
    ----------
    function : function
        Function to apply in parallel.
    outs : tuple
        Tuple of out arrays on which to put results.
    *args : tuple
        Arrays on which to loop.
    stacks : bool
        True if looping on stacks.
    **kwargs : dict
        Fixed parameters to pass to function.

    Examples
    --------
    >>> from arrlp import parallel_array
    ...
    >>> parallel_array(xp.fft, arrays, outs=outs, stacks=True, kernel=kernel)
    '''

    if not isinstance(outs, tuple) :
        raise TypeError(f'outs should be tuple not {type(outs)}')
    outs = list(outs)

    # Infer dimensions
    shape = args[0].shape
    nstacks = shape[0]
    nchannels = shape[-1]

    # Loop on stacks
    if stacks :
        copy = Parallel(n_jobs=-1, backend="loky")(
            delayed(function)(
                *(arg[i] for arg in args), **kwargs
            )
            for i in range(nstacks)
        )

        results = (copy,) if len(outs) == 1 else zip(*copy)
        for i, (res, dst) in enumerate(zip(results, outs)):
            if dst is None :
                dst = np.empty_like(res[0], shape=(len(res), *res[0].shape))
                outs[i] = dst
            copystacks(list(res), dst)

    # Loop on channels
    else :
        copy = Parallel(n_jobs=-1, backend="loky")(
            delayed(function)(
                *(arg[..., i] for arg in args), **kwargs
            )
            for i in range(nchannels)
        )

        results = (copy,) if len(outs) == 1 else zip(*copy)
        for i, (res, dst) in enumerate(zip(results, outs)):
            if dst is None :
                dst = np.empty_like(res[0], shape=(*res[0].shape, len(res)))
                outs[i] = dst
            copychannels(list(res), dst)
    
    # End
    return outs[0] if len(outs) == 1 else tuple(outs)



@njit(parallel=True)
def copystacks(copyfrom, copyto):
    n = len(copyfrom)
    for i in prange(n):
        copyto[i] = copyfrom[i]

@njit(parallel=True)
def copychannels(copyfrom, copyto):
    n = len(copyfrom)
    for i in prange(n):
        copyto[..., i] = copyfrom[i]



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)