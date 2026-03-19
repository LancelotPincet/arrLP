#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-18
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : sortloop

"""
This generator allows to loop on unique values of an array rapidly by first sorting the array.
"""



# %% Libraries
from arrlp import get_xp



# %% Function
def sortloop(array, *arrays, axis=None, cuda=False, **kwargs) :
    '''
    This generator allows to loop on unique values of an array rapidly by first sorting the array.
    
    Parameters
    ----------
    array : ndarray
        The array to determine unique values and slices.
    *arrays : ndarrays
        Other arrays to slice in parallel with `array`.
    cuda : bool
        True to use on GPU in cupy.
    axis : int or None
        Axis along which to find unique values. If None, flattens the array.

    Returns
    -------
    positions : array
        Positions corresponding to the current masked elements.
    array : array
        first array which was used for defining the masked region
    *arrays :
        other array elements in the masked region

    Examples
    --------
    >>> from arrlp import sortloop
    ...
    >>> for pos, r, y, x in sortloop(R, Y, X) :
    ...     result[pos] = scalar_function(y, x) # masked for each r
    '''

    # arrays
    xp = get_xp(cuda)
    array = xp.asarray(array)
    arrays = [xp.asarray(a) for a in arrays]
    
    # axis
    if axis is None:
        flat_array = array.ravel()
        flat_arrays = [a.ravel() for a in arrays]
        ax = 0
    else:
        flat_array = array
        flat_arrays = list(arrays)
        ax = axis

    # sort
    argsort = xp.argsort(flat_array, axis=ax)
    sorted_array = flat_array[argsort]
    sorted_arrays = [a[argsort] for a in flat_arrays]

    # unique values and counts
    unique, counts = xp.unique(sorted_array, return_counts=True, axis=ax)
    cumsum = xp.hstack((0, xp.cumsum(counts)))

    # generate chunks
    for i, val in enumerate(unique):
        slc = slice(cumsum[i], cumsum[i+1])
        positions = argsort[slc]
        chunk_arrays = [a[slc] for a in sorted_arrays]
        yield positions, val, *chunk_arrays



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)