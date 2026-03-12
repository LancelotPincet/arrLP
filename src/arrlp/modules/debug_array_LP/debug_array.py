#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-12
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : debug_array

"""
This function allows to debug array functions while comparing various modes.
"""



# %% Libraries
import numpy as np
try :
    import cupy as cp
except ImportError :
    cp = None
from time import perf_counter
import warnings



# %% Function
def debug_array(func, modes, stacks_list=None, channels_list=None, narrays=1, **kwargs) :
    f'''
    This function allows to debug array functions while comparing various modes.
    
    Parameters
    ----------
    func : function
        function to test.
    modes : dict
        dict of dicts corresponding to various modes that should give same results.
    stacks_list : list
        List of stacks bools to test, must same size as channels_list.
    channels_list : list
        List of channels bools to test, must same size as stacks_list.
    narrays : int
        Number of input arrays.
    **kwargs : dict
        constant keyword arguments.

    Examples
    --------
    >>> from arrlp import debug_array
    ...
    >>> debug_array(func, ndimes, dict("Normal": dict()))
    '''



    # Data size ~2**24
    nstacks = int(2**8)
    nchannels = int(2**2)
    match func.ndims :
        case 1:
            shape = (int(2**14),)
        case 2:
            shape = (int(2**7), int(2**7))
        case 3:
            shape = (int(2**4), int(2**5), int(2**5))
        case _: raise ValueError(f'number of dimensions cannot be {func.ndims}')

    # Manage modes
    modes = {key: value for key, value in modes.items() if cp is not None or not value["cuda"]}



    # Timeit function
    def timeit(key, arrays, stacks, channels, parallel, cuda, **kw) :

        # Init
        xp = cp if cuda else np
        arrays = [xp.asarray(array) for array in arrays]

        # Calculate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print('Compile run...', end='')
            func(*arrays, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, test=True, **kwargs, **kw)
            print('\rRun...        ', end='')
            tic = perf_counter()
            out = func(*arrays, stacks=stacks, channels=channels, parallel=parallel, cuda=cuda, test=True, **kwargs, **kw)
            toc = perf_counter()
            print(f'\r{key} took {(toc-tic)*1000:.3f}ms      ')
            if cuda : out = xp.asnumpy(out)
        return out



    # Loops
    if channels_list is None : channels_list = [False, False, True, True]
    if stacks_list is None : stacks_list = [False, True, False, True]
    for channels, stacks in zip(channels_list, stacks_list) :

        print(f'\nTesting for stacks={stacks} and channels={channels}')

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
        arrays = [np.random.random(_shape) for _ in range(narrays)]

        # Calculate
        results = {}
        ref = None
        for key, value in modes.items():
            parallel = value['parallel']
            if (not channels and not stacks) and parallel and hasattr(func, "use_joblib") and func.use_joblib : continue
            results[key] = timeit(key, arrays, stacks, channels, **value)
            if ref is None :
                ref = key
        
        # Compare
        for key, value in results.items():
            if key == ref : continue
            print(f"\rChecking correctness vs reference: {key}" + " "*(60-34-len(key)), end='')
            np.testing.assert_allclose(
                results[ref], results[key],
                rtol=1e-4,
                atol=1e-5,
                err_msg="Outputs differ between optimizations"
            )
        print('\r'+' '*60+'\n')



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)