#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_img_correlate1d(self, array, kernel, **kwargs) :
    return dict(
        kernel=(self.xp.asarray(kernel[0]), self.xp.asarray(kernel[1])), # y, x
    )

def out_img_correlate1d(self, array, **kwargs) :
    return self.xp.empty_like(array, dtype=self.xp.float32)

def cpu_img_correlate1d(self, out, array, kernel, mode='nearest', **kwargs) :
    return self.ndimage.correlate1d(self.ndimage.correlate1d(array, weights=kernel[0], output=out, axis=self.axes[0], mode=mode), weights=kernel[1], output=out, axis=self.axes[1], mode=mode, **kwargs)

def par_img_correlate1d(self, array, kernel, mode='nearest', **kwargs) :
    return self.ndimage.correlate1d(self.ndimage.correlate1d(array, weights=kernel[0], axis=0, mode=mode), weights=kernel[1], axis=1, mode=mode, **kwargs)

def gpu_img_correlate1d(self, out, array, kernel, mode='nearest', **kwargs) :
    return self.ndimage.correlate1d(self.ndimage.correlate1d(array, weights=kernel[0], axis=self.axes[0], mode=mode), weights=kernel[1], axis=self.axes[1], mode=mode, output=out, **kwargs) # First call must be allocated into a temporary buffer (output=None)



# Main function
img_correlate1d = FunctionArray(
    
    # Mandatory
    ndims = 2,
    cpu_function = cpu_img_correlate1d,
    par_function = par_img_correlate1d,
    gpu_function = gpu_img_correlate1d,
    out_function = out_img_correlate1d,
    ini_function = ini_img_correlate1d,

    # Loops
    par_loop = True,
    use_joblib = True, # If True, arguments of parallel function should not have "out".

    # Performances
    remove_parallel = False,
    remove_cuda = False,

)



if __name__ == '__main__' :
    from arrlp import debug_array
    from arrlp import kernel
    func = img_correlate1d

    # Arguments
    kwargs = dict(
        kernel=(kernel(ndims=1, pixel=1, sigma=2), kernel(ndims=1, pixel=1, sigma=3)),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
