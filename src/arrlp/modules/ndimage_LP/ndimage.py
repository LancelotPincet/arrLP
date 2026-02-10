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
import scipy.ndimage as cpu_ndimage
try:
    import cupyx.scipy.ndimage as gpu_ndimage 
except ImportError:
    gpu_ndimage = None



# %% Function
def ndimage(cuda:bool) :
    '''
    Gets scipy.ndimage of cupyx.scipy.ndimage depending on the value of cuda argument.
    
    Parameters
    ----------
    cuda : bool
        Tells to take scipy or cupyx.scipy.

    Returns
    -------
    _ndimage : scipy.ndimage or cupyx.scipy.ndimage
        Library to use.

    Raises
    ------
    ImportError
        If cuda is asked but unavailable.

    Examples
    --------
    >>> from arrlp import ndimage
    ...
    >>> _ndimage = ndimage(cuda)
    '''

    if cuda and gpu_ndimage is None :
        raise ImportError('Cupy is not available for xp function')
    return gpu_ndimage if cuda else cpu_ndimage



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)