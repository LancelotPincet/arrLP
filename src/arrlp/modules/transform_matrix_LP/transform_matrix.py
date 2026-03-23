#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-22
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : transform_matrix

"""
Defines the 3x3 transformation matrix for affine transformation in 2D.
"""



# %% Libraries
import numpy as np



# %% Function
def transform_matrix(shape=None, *, shiftx=0., shifty=0., shearx=0., sheary=0., angle=0., scalex=1., scaley=1.) :
    '''
    Defines the 3x3 transformation matrix for affine transformation in 2D.
    
    Parameters
    ----------
    shape : int or tuple or np.ndarray
        Describes the shape of the coordinates.
        If int, all dimensions will have this value.
        If tuple, corresponds to the shape.
    shiftx : float
        Shift value in x
    shifty : float
        Shift value in y
    shearx : float
        Shear value in x
    sheary : float
        Shear value in y
    angle : float
        Rotation angle [°]
    scalex : float
        Scale value in x
    scaley : float
        Scale value in y

    Examples
    --------
    >>> from arrlp import transform_matrix
    ...
    >>> matrix = transform_matrix(shape, shiftx=2, shifty=3.4)
    '''

    # Manage shape argument
    if shape is None :
        shape = (0, 0)
    elif isinstance(shape, int) :
        shape = (shape,) * 2
    elif isinstance(shape, tuple) :
        pass
    else :
        shape = shape.shape

    h, w = shape
    h, w = h/2, w/2
    matrix = np.eye(3, dtype=np.float32)
    if shearx != 0 or sheary != 0 :
        matrix @= np.asarray([[1., 0., h],
                              [0., 1., w],
                              [0., 0., 1.]])
        matrix @= np.asarray([[1., sheary, 0.],
                              [shearx, 1., 0.],
                              [0., 0., 1.]])
        matrix @= np.asarray([[1., 0., -h],
                              [0., 1., -w],
                              [0., 0., 1.]])
    if angle != 0 :
        angle = -np.radians(angle)
        matrix @= np.asarray([[1., 0., h],
                              [0., 1., w],
                              [0., 0., 1.]])
        matrix @= np.asarray([[np.cos(angle), -np.sin(angle), 0.],
                              [np.sin(angle), np.cos(angle), 0.],
                              [0., 0., 1.]])
        matrix @= np.asarray([[1., 0., -h],
                              [0., 1., -w],
                              [0., 0., 1.]])
    if scalex != 1 or scaley != 1 :
        matrix @= np.asarray([[1., 0., h],
                              [0., 1., w],
                              [0., 0., 1.]])
        matrix @= np.asarray([[scaley, 0., 0.],
                              [0., scalex, 0.],
                              [0., 0., 1.]])
        matrix @= np.asarray([[1., 0., -h],
                              [0., 1., -w],
                              [0., 0., 1.]])
    if shiftx != 0 or shifty != 0 :
        matrix @= np.asarray([[1., 0., shifty],
                              [0., 1., shiftx],
                              [0., 0., 1.]])
    return matrix



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)