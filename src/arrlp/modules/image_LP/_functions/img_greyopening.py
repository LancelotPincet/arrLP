#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_img_greyopening(self, array, *, structure=None, footprint=None, **kwargs) :
    return dict(
        structure = None if structure is None else self.xp.asarray(structure),
        footprint = None if footprint is None else self.xp.asarray(footprint),
    )

def out_img_greyopening(self, array, **kwargs) :
    return self.xp.empty_like(array, dtype=self.xp.float32)

def _img_greyopening(self, out, array, *, structure, footprint, mode='nearest', **kwargs) :
    return self.ndimage.grey_opening(array, structure=structure, footprint=footprint, axes=self.axes, output=out, mode=mode, **kwargs)

def par_img_greyopening(self, array, *, structure, footprint, mode='nearest', **kwargs) :
    return self.ndimage.grey_opening(array, structure=structure, footprint=footprint, mode=mode, **kwargs)



# Main function
img_greyopening = FunctionArray(
    
    # Mandatory
    ndims = 2,
    cpu_function = _img_greyopening,
    par_function = par_img_greyopening,
    gpu_function = _img_greyopening,
    out_function = out_img_greyopening,
    ini_function = ini_img_greyopening,

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
    func = img_greyopening

    # Arguments
    kwargs = dict(
        footprint=kernel(ndims=2, pixel=1, window=12),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
