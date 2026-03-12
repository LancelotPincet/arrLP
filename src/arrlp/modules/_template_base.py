#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_arr_function(self, array, **kwargs) :
    return dict() # return values in this dict will be passed as kwargs in cpu/par/gpu functions 

def out_arr_function(self, array, **kwargs) : # To delete if wrapped functions does not allow inplace output
    return self.xp.empty_like(array, dtype=self.xp.float32)

def cpu_arr_function(self, out, array, **kwargs) :
    return out # TODO

def par_arr_function(self, out, array, **kwargs) :
    return out # TODO

def gpu_arr_function(self, out, array, **kwargs) :
    return out # TODO



# Main function
arr_function = FunctionArray(
    
    # Mandatory
    ndims = 2,
    cpu_function = cpu_arr_function,
    par_function = par_arr_function,
    gpu_function = gpu_arr_function,
    out_function = out_arr_function,
    ini_function = ini_arr_function,

    # Loops
    cpu_loop = False,
    par_loop = False,
    gpu_loop = False, # True usually make bad GPU performance, check first if there is no ndimension support, and also try investigating reshaping
    use_joblib = True, # If True, arguments of parallel function should not have "out".

    # Performances
    remove_parallel = False,
    remove_cuda = False,

)



if __name__ == '__main__' :
    from arrlp import debug_array
    func = arr_function

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
