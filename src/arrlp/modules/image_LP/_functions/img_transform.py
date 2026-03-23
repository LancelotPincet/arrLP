#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_img_transform(self, array, *, matrix, **kwargs) :
    return dict(
        matrix=self.xp.asarray(matrix),
    )

def out_img_transform(self, array, **kwargs) :
    return self.xp.empty_like(array, dtype=self.xp.float32)

def _img_transform(self, out, array, *, matrix, **kwargs) :
    return self.ndimage.affine_transform(array, matrix[:2, :2], offset=matrix[:2, 2], output=out, **kwargs)

def par_img_transform(self, array, *, matrix, **kwargs) :
    return self.ndimage.affine_transform(array, matrix[:2, :2], offset=matrix[:2, 2], **kwargs)



# Main function
img_transform = FunctionArray(
    
    # Mandatory
    ndims = 2,
    cpu_function = _img_transform,
    par_function = par_img_transform,
    gpu_function = _img_transform,
    out_function = out_img_transform,
    ini_function = ini_img_transform,

    # Loops
    cpu_loop = True,
    par_loop = True,
    gpu_loop = True, # True usually make bad GPU performance, check first if there is no ndimension support, and also try investigating reshaping
    use_joblib = True, # If True, arguments of parallel function should not have "out".

    # Performances
    remove_parallel = False,
    remove_cuda = False,

)



if __name__ == '__main__' :
    from arrlp import debug_array
    from arrlp import transform_matrix
    func = img_transform

    # Arguments
    kwargs = dict(
        matrix=transform_matrix(shiftx=2.3),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
