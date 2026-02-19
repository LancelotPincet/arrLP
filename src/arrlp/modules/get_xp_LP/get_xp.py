#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : get_xp

"""
This function returns numpy/cupy depending on the input.
"""



# %% Libraries
import numpy as np
try :
    import cupy as cp
except ImportError :
    cp = None



# %% Function
def get_xp(input) :
    '''
    This function returns numpy/cupy depending on the input.
    
    Parameters
    ----------
    input : bool or array
        If bool, says if using cuda, if array, will detect on cpu or gpu.

    Returns
    -------
    xp : module
        numpy or cupy.

    Examples
    --------
    >>> from arrlp import xp
    ...
    >>> get_xp(use_cuda) # with bool
    >>> get_xp(array) # with array
    '''

    if cp is None :
        return np
    if isinstance(input, bool) :
        return cp if input else np
    return cp.get_array_module(input)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)