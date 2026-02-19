#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from arrlp import FunctionArray



# %% Function

# Initializations
def ini_sig_wiener(self, array, kernel, power=2, balance=10**2, **kwargs) :
    kernel = self.xp.asarray(kernel)
    axes = self.axes
    X = array.shape[axes[0]]
    pad_x_total = X - kernel.shape[0]
    if pad_x_total < 0 :
        start = -pad_x_total // 2
        kernel = kernel[start:start + X]
        pad_x_total = 0
    pad_x = (pad_x_total // 2, pad_x_total - pad_x_total // 2)
    padded = self.xp.pad(kernel, pad_x, mode='constant')
    fft_kernel = self.xp.abs(self.scipyx.fft.fftshift(self.scipyx.fft.fft(padded)))
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

def _sig_wiener(self, out, array, W, kernel, power=2, balance=10**2, **kwargs) :
    dtype = array.dtype
    fft = self.scipyx.fft.fftshift(self.scipyx.fft.fft(array, axis=self.axes[0], **kwargs), axes=self.axes)
    return self.xp.real(self.scipyx.fft.ifft(self.scipyx.fft.ifftshift(fft * W, axes=self.axes), axis=self.axes[0], **kwargs)).astype(dtype)

def par_sig_wiener(self, out, array, W, kernel, power=2, balance=10**2, **kwargs) :
    dtype = array.dtype
    fft = self.scipyx.fft.fftshift(self.scipyx.fft.fft(array, axis=self.axes[0], workers=-1, **kwargs), axes=self.axes)
    return self.xp.real(self.scipyx.fft.ifft(self.scipyx.fft.ifftshift(fft * W, axes=self.axes), axis=self.axes[0], workers=-1, **kwargs)).astype(dtype)



# Main function
sig_wiener = FunctionArray(
    
    # Mandatory
    ndims = 1,
    cpu_function = _sig_wiener,
    par_function = par_sig_wiener,
    gpu_function = _sig_wiener,
    out_function = None,
    ini_function = ini_sig_wiener,

    # Loops
    use_joblib = False, # If True, arguments of parallel function should not have "out".

    # Performances
    remove_parallel = True,
    remove_cuda = False,

)



if __name__ == '__main__' :
    from arrlp import debug_array
    from arrlp import kernel
    func = sig_wiener

    # Arguments
    kwargs = dict(
        kernel = kernel(ndims=1, sigma=3),
    )

    # Modes
    modes = { # list of dicts with kwargs
        "Reference": dict(cuda=False, parallel=False),
        "Parallel": dict(cuda=False, parallel=True),
        "Cuda": dict(cuda=True, parallel=False),
    }

    debug_array(func, modes, **kwargs)
