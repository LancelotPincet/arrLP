#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : relabel

"""
This function redefines a label array to fill potential holes inside the considered labels.
"""



# %% Libraries
from arrlp import get_xp



# %% Function
def relabel(labels) :
    '''
    This function redefines a label array to fill potential holes inside the considered labels.
    
    Parameters
    ----------
    labels : np.array
        Array of labels.

    Returns
    -------
    newlabels : np.array
        New array of labels.

    Examples
    --------
    >>> from arrlp import relabel
    ...
    >>> relabel(array)
    '''

    xp = get_xp(labels)
    shape = labels.shape
    labels = labels.ravel().astype(xp.uint32)  # Flatten array
    unique_values = xp.unique(labels)  # Get sorted unique values
    mapping = xp.zeros(unique_values[-1].item() + 1, dtype=xp.uint32)  # Create mapping array
    indices = xp.arange(len(unique_values), dtype=xp.uint32)
    mapping[unique_values] = indices
    labels[:] = mapping[labels]
    return labels.reshape(shape)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)