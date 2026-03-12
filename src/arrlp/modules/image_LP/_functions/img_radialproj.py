#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray, coordinates



# %% Function

# Initializations
def ini_img_radialproj(self, array, pixel=1., **kwargs) :
    try :
        if len(pixel) != 2 :
            raise ValueError('pixel cannot have other than 2 values in tuple')
    except TypeError :
        pixel = pixel, pixel
    ratio = pixel[0] / min(pixel), pixel[1]/min(pixel)
    y, x = coordinates(self.shape(array), pixel=ratio, grid=True, cuda=self.cuda)
    R = (self.xp.sqrt(y**2 + x**2)).astype(int).ravel()
    rsum = self.xp.bincount(R.ravel())
    return dict(R=R, rsum=rsum) # return values in this dict will be passed as kwargs in cpu/par/gpu functions 

def _img_radialproj(self, out, array, R, rsum, **kwargs) :
    psum = self.xp.bincount(R, weights=array.ravel())
    return psum / rsum

def par_img_radialproj(self, array, R, rsum, **kwargs) :
    psum = self.xp.bincount(R, weights=array.ravel())
    return psum / rsum



# Main function
img_radialproj = FunctionArray(
    
    # Mandatory
    ndims = 2,
    cpu_function = _img_radialproj,
    par_function = par_img_radialproj,
    gpu_function = _img_radialproj,
    out_function = None,
    ini_function = ini_img_radialproj,

    # Loops
    cpu_loop = True,
    par_loop = True,
    gpu_loop = True, # True usually make bad GPU performance, check first if there is no ndimension support, and also try investigating reshaping
    use_joblib = True, # If True, arguments of parallel function should not have "out".

    # Performances
    remove_parallel = True,
    remove_cuda = True,

)



if __name__ == '__main__' :
    from arrlp import debug_array
    func = img_radialproj

    # Arguments
    kwargs = dict(
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
