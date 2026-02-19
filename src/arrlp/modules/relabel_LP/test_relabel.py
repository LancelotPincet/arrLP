#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : relabel

"""
This file allows to test relabel

relabel : This function redefines a label array to fill potential holes inside the considered labels.
"""



# %% Libraries
from corelp import debug
from arrlp import relabel
from time import perf_counter
import numpy as np
try :
    import cupy as cp
except ImportError :
    cp = None
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test relabel function
    '''
    n = 100000

    labels = np.arange(0, n*10 + 10, 10)
    relabel(labels)
    labels = np.arange(0, n*10 + 10, 10)/2
    tic = perf_counter()
    newlabels = relabel(labels)
    toc = perf_counter()
    print(f'CPU took {(toc-tic)*1000:.3f}ms')
    assert (newlabels == np.arange(n+1)).all()

    if cp is not None :
        labels = cp.arange(0, n*10 + 10, 10)
        relabel(labels)
        labels = cp.arange(0, n*10 + 10, 10)/2
        tic = perf_counter()
        newlabels = relabel(labels)
        toc = perf_counter()
        print(f'GPU took {(toc-tic)*1000:.3f}ms')
        assert (newlabels == cp.arange(n+1)).all()

    



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)