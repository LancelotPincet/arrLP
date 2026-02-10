#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-09
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : axes

"""
Get axes on which to apply the xp function.
"""



# %% Libraries



# %% Function
def axes(ndim:int, stacks:bool) :
    '''
    Get axes on which to apply the xp function.
    
    Parameters
    ----------
    ndim : int
        Number of dimensions of array.

    Returns
    -------
    _axes : tuple
        axes on which to do calculation.

    Examples
    --------
    >>> from arrlp import axes
    ...
    >>> axes = axes(2, True)
    '''

    start = int(stacks)
    return tuple(range(start, start + ndim))



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)