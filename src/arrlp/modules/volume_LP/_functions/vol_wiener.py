#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_vol_wiener(self, array, kernel, power=2, balance=10**2, **kwargs) :
    kernel = self.xp.asarray(kernel)
    axes = self.axes
    Z, Y, X = array.shape[axes[0]], array.shape[axes[1]], array.shape[axes[2]]
    pad_z_total = Z - kernel.shape[0]
    if pad_z_total < 0 :
        start = -pad_z_total // 2
        kernel = kernel[start:start + Z, :, :]
        pad_z_total = 0
    pad_y_total = Y - kernel.shape[1]
    if pad_y_total < 0 :
        start = -pad_y_total // 2
        kernel = kernel[:, start:start + Y, :]
        pad_y_total = 0
    pad_x_total = X - kernel.shape[2]
    if pad_x_total < 0 :
        start = -pad_x_total // 2
        kernel = kernel[:, :, start:start + X]
        pad_x_total = 0
    pad_z = (pad_z_total // 2, pad_z_total - pad_z_total // 2)
    pad_y = (pad_y_total // 2, pad_y_total - pad_y_total // 2)
    pad_x = (pad_x_total // 2, pad_x_total - pad_x_total // 2)
    padded = self.xp.pad(kernel, (pad_z, pad_y, pad_x), mode='constant')
    fft_kernel = self.xp.abs(self.scipyx.fft.fftshift(self.scipyx.fft.fftn(padded)))
    W = fft_kernel**(power-1) / (fft_kernel**power + fft_kernel.max() / balance)
    if self.stacks and self.channels :
        W = W.reshape((1, *W.shape, 1))
    elif self.stacks :
        W = W.reshape((1, *W.shape))
    elif self.channels :
        W = W.reshape((*W.shape, 1))
    return dict(
        W=W,
    )

def _vol_wiener(self, out, array, W, kernel, power=2, balance=10**2, **kwargs) :
    dtype = array.dtype
    fft = self.scipyx.fft.fftshift(self.scipyx.fft.fftn(array, axes=self.axes, **kwargs), axes=self.axes)
    return self.xp.real(self.scipyx.fft.ifftn(self.scipyx.fft.ifftshift(fft * W, axes=self.axes), axes=self.axes, **kwargs)).astype(dtype)

def par_vol_wiener(self, out, array, W, kernel, power=2, balance=10**2, **kwargs) :
    dtype = array.dtype
    fft = self.scipyx.fft.fftshift(self.scipyx.fft.fftn(array, axes=self.axes, workers=-1, **kwargs), axes=self.axes)
    return self.xp.real(self.scipyx.fft.ifftn(self.scipyx.fft.ifftshift(fft * W, axes=self.axes), axes=self.axes, workers=-1, **kwargs)).astype(dtype)



# Main function
vol_wiener = FunctionArray(
    
    # Mandatory
    ndims = 3,
    cpu_function = _vol_wiener,
    par_function = par_vol_wiener,
    gpu_function = _vol_wiener,
    out_function = None,
    ini_function = ini_vol_wiener,

    # Loops
    use_joblib = False, # If True, arguments of parallel function should not have "out".

    # Performances
    remove_parallel = False,
    remove_cuda = False,

)



if __name__ == '__main__' :
    from arrlp import debug_array
    from arrlp import kernel
    func = vol_wiener

    # Arguments
    kwargs = dict(
        kernel = kernel(ndims=3, sigma=3),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
