#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-04
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : nb_threads

"""
This context manager defines the number of logical cores on numba parallel threads.
"""



# %% Libraries
from contextlib import contextmanager
from numba import set_num_threads, get_num_threads
import os



# %% Function
@contextmanager
def nb_threads(parallel=False) :
    '''
    This context manager defines the number of logical cores on numba parallel threads.
    
    Parameters
    ----------
    threads : int or bool
        If bool set -1 by default for True and 1 for False
        If int sets the number of threads, -1 means maximum

    Examples
    --------
    >>> from arrlp import nb_threads
    ...
    >>> with nb_threads(3) : # 3 threads
    ...     parallel_funct()
    '''

    if parallel is False :
        threads = 1
    elif parallel is True :
        threads = os.cpu_count() or 1
    elif parallel == -1 :
        threads = os.cpu_count() or 1
    else :
        threads = parallel
    old = get_num_threads()
    set_num_threads(threads)
    try:
        yield
    finally:
        set_num_threads(old)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)