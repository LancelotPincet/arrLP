#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-08-28
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP

"""
A library that provides tools for arrays on CPU & GPU [any shape].
"""



# %% Source import
sources = {
'FunctionArray': 'arrlp.modules.FunctionArray_LP.FunctionArray',
'compress': 'arrlp.modules.compress_LP.compress',
'coordinates': 'arrlp.modules.coordinates_LP.coordinates',
'debug_array': 'arrlp.modules.debug_array_LP.debug_array',
'gc': 'arrlp.modules.gc_LP.gc',
'get_xp': 'arrlp.modules.get_xp_LP.get_xp',
'kernel': 'arrlp.modules.kernel_LP.kernel',
'nb_threads': 'arrlp.modules.nb_threads_LP.nb_threads',
'relabel': 'arrlp.modules.relabel_LP.relabel',
'sortloop': 'arrlp.modules.sortloop_LP.sortloop',
'transform_matrix': 'arrlp.modules.transform_matrix_LP.transform_matrix',
'transform_parameters': 'arrlp.modules.transform_parameters_LP.transform_parameters',
'vol_autocorr': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_autocorr',
'vol_correlate': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_correlate',
'vol_crosscorr': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_crosscorr',
'vol_correlate1d': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_correlate1d',
'vol_convolve1d': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_convolve1d',
'vol_maxifilter': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_maxifilter',
'vol_ifft': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_ifft',
'vol_convolve': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_convolve',
'vol_wiener': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_wiener',
'vol_minifilter': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_minifilter',
'vol_greyopening': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_greyopening',
'vol_fft': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_fft',
'vol_gaussianfilter': 'arrlp.modules.FunctionArray_LP._functions.volume.vol_gaussianfilter',
'sig_crosscorr': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_crosscorr',
'sig_minifilter': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_minifilter',
'sig_wiener': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_wiener',
'sig_greyopening': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_greyopening',
'sig_fft': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_fft',
'sig_convolve': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_convolve',
'sig_correlate': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_correlate',
'sig_gaussianfilter': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_gaussianfilter',
'sig_ifft': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_ifft',
'sig_maxifilter': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_maxifilter',
'sig_autocorr': 'arrlp.modules.FunctionArray_LP._functions.signal.sig_autocorr',
'img_greyopening': 'arrlp.modules.FunctionArray_LP._functions.image.img_greyopening',
'img_radialproj': 'arrlp.modules.FunctionArray_LP._functions.image.img_radialproj',
'img_wiener': 'arrlp.modules.FunctionArray_LP._functions.image.img_wiener',
'img_transform': 'arrlp.modules.FunctionArray_LP._functions.image.img_transform',
'img_correlate': 'arrlp.modules.FunctionArray_LP._functions.image.img_correlate',
'img_fft': 'arrlp.modules.FunctionArray_LP._functions.image.img_fft',
'img_maxifilter': 'arrlp.modules.FunctionArray_LP._functions.image.img_maxifilter',
'img_convolve1d': 'arrlp.modules.FunctionArray_LP._functions.image.img_convolve1d',
'img_ifft': 'arrlp.modules.FunctionArray_LP._functions.image.img_ifft',
'img_autocorr': 'arrlp.modules.FunctionArray_LP._functions.image.img_autocorr',
'img_correlate1d': 'arrlp.modules.FunctionArray_LP._functions.image.img_correlate1d',
'img_minifilter': 'arrlp.modules.FunctionArray_LP._functions.image.img_minifilter',
'img_convolve': 'arrlp.modules.FunctionArray_LP._functions.image.img_convolve',
'img_gaussianfilter': 'arrlp.modules.FunctionArray_LP._functions.image.img_gaussianfilter',
'img_crosscorr': 'arrlp.modules.FunctionArray_LP._functions.image.img_crosscorr'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)