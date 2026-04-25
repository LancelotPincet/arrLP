#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : transform_parameters

"""
Defines the affine transformations in 2D for a 3x3 transformation matrix.
"""



# %% Libraries
import numpy as np



def transform_parameters(matrix, shape=None, *, shears=None, angle=None, scales=None, stacks=False):
    """
    Recover transform_matrix parameters from a 3x3 affine matrix.

    One of shears, angle, or scales may be provided to make the inverse unique.
    If none is provided, shears=(0, 0) is assumed.

    Parameters
    ----------
    matrix : np.ndarray
        3x3 affine transformation matrix.
    shape : int, tuple, np.ndarray, optional
        Same convention as transform_matrix.
    shears : tuple, optional
        Known shears as (sheary, shearx).
    angle : float, optional
        Known rotation angle in degrees.
    scales : tuple, optional
        Known scales as (scaley, scalex).

    Returns
    -------
    shiftx, shifty, shearx, sheary, angle, scalex, scaley
    """

    matrix = np.asarray(matrix, dtype=float)

    if matrix.shape != (3, 3):
        raise ValueError("matrix must be a 3x3 array.")

    # Manage shape argument
    if shape is None:
        shape = (0, 0)
    elif isinstance(shape, (int, float)):
        shape = (shape,) * 2
    elif isinstance(shape, tuple):
        pass
    else:
        shape = shape.shape[:2] if not stacks else shape.shape[1:3]

    h, w = shape
    center = np.asarray([h / 2, w / 2], dtype=float)

    A = matrix[:2, :2]
    t = matrix[:2, 2]

    defined = sum(x is not None for x in (shears, angle, scales))

    if defined == 0:
        shears = (0.0, 0.0)
    elif defined > 1:
        raise ValueError("Only one of shears, angle, or scales may be defined.")

    if shears is not None:
        sheary, shearx = shears

        H = np.asarray([
            [1.0, sheary],
            [shearx, 1.0],
        ])

        B = np.linalg.solve(H, A)

        scaley = np.linalg.norm(B[:, 0])
        scalex = np.linalg.norm(B[:, 1])

        if scaley == 0 or scalex == 0:
            raise ValueError("Cannot recover angle with zero scale.")

        R = B @ np.diag([1 / scaley, 1 / scalex])

        theta = np.arctan2(R[1, 0], R[0, 0])
        angle = -np.degrees(theta)

    elif angle is not None:
        theta = -np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)

        r0 = A[1, 0] / A[0, 0]
        r1 = A[0, 1] / A[1, 1]

        M = np.asarray([
            [c, -r0 * s],
            [r1 * s, c],
        ])

        b = np.asarray([
            r0 * c - s,
            r1 * c + s,
        ])

        shearx, sheary = np.linalg.solve(M, b)

        scaley = A[0, 0] / (c + sheary * s)
        scalex = A[1, 1] / (-shearx * s + c)

    elif scales is not None:
        scaley, scalex = scales

        if scaley == 0 or scalex == 0:
            raise ValueError("Cannot recover angle/shear with zero scale.")

        B = A @ np.diag([1 / scaley, 1 / scalex])

        M = np.asarray([
            [B[0, 0], -B[0, 1]],
            [B[1, 1],  B[1, 0]],
        ])

        c, s = np.linalg.solve(M, np.ones(2))

        norm = np.hypot(c, s)
        c, s = c / norm, s / norm

        theta = np.arctan2(s, c)
        angle = -np.degrees(theta)

        Rinv = np.asarray([
            [c, s],
            [-s, c],
        ])

        H = B @ Rinv

        sheary = H[0, 1]
        shearx = H[1, 0]

    L = A

    shift = np.linalg.solve(
        L,
        t - center + L @ center,
    )

    shifty = shift[0]
    shiftx = shift[1]

    return shiftx, shifty, shearx, sheary, angle, scalex, scaley


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)