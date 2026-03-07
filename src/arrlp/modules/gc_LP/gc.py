#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-06
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : gc

"""
Makes garbage collection to free (V)RAM.
"""



# %% Libraries
import gc as garbage_collect
try :
    import cupy as cp
except ImportError :
    cp = None



# %% Function
def gc() :
    '''
    Makes garbage collection to free (V)RAM.
    
    Examples
    --------
    >>> from arrlp import gc
    ...
    >>> gc()
    '''

    garbage_collect.collect()
    if cp is not None :
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)