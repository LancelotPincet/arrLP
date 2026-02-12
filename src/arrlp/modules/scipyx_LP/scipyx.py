#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-10
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : ndimage

"""
Gets scipy.ndimage of cupyx.scipy.ndimage depending on the value of cuda argument.
"""



# %% Libraries
import scipy as cpu_scipyx
try:
    import cupyx.scipy as gpu_scipyx 
except ImportError:
    gpu_scipyx = None



# %% Function
def scipyx(cuda:bool) :
    '''
    Gets scipy of cupyx.scipy depending on the value of cuda argument.
    
    Parameters
    ----------
    cuda : bool
        Tells to take scipy or cupyx.scipy.

    Returns
    -------
    _scipyx : scipy or cupyx.scipy
        Library to use.

    Raises
    ------
    ImportError
        If cuda is asked but unavailable.

    Examples
    --------
    >>> from arrlp import scipyx
    ...
    >>> _scipyx = scipyx(cuda)
    '''

    if cuda and gpu_scipyx is None :
        raise ImportError('Cupy is not available for scipyx function')
    return gpu_scipyx if cuda else cpu_scipyx



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)