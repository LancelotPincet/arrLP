#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_img_wiener(self, array, kernel, **kwargs) :
    kernel = self.xp.asarray(kernel)
    return dict(
        kernel=kernel,
    )

def _img_wiener(self, out, array, kernel, balance=10**2, **kwargs) :
    return self.skimagex.restoration.wiener(array, kernel, balance, **kwargs)

def par_img_wiener(self, array, kernel, balance=10**2, **kwargs) :
    return self.skimagex.restoration.wiener(array, kernel, balance, **kwargs)



# Main function
img_wiener = FunctionArray(
    
    # Mandatory
    ndims = 2,
    cpu_function = _img_wiener,
    par_function = par_img_wiener,
    gpu_function = _img_wiener,
    out_function = None,
    ini_function = ini_img_wiener,

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
    from arrlp import kernel
    func = img_wiener

    # Arguments
    kwargs = dict(
        kernel = kernel(ndims=2, sigma=3),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
