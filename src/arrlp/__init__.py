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
'image': 'arrlp.modules.image_LP.image',
'kernel': 'arrlp.modules.kernel_LP.kernel',
'nb_threads': 'arrlp.modules.nb_threads_LP.nb_threads',
'relabel': 'arrlp.modules.relabel_LP.relabel',
'signal': 'arrlp.modules.signal_LP.signal',
'sortloop': 'arrlp.modules.sortloop_LP.sortloop',
'volume': 'arrlp.modules.volume_LP.volume',
'img_ifft': 'arrlp.modules.image_LP._functions.img_ifft',
'img_autocorr': 'arrlp.modules.image_LP._functions.img_autocorr',
'img_gaussianfilter': 'arrlp.modules.image_LP._functions.img_gaussianfilter',
'img_convolve': 'arrlp.modules.image_LP._functions.img_convolve',
'img_crosscorr': 'arrlp.modules.image_LP._functions.img_crosscorr',
'img_': 'arrlp.modules.image_LP._functions.img_',
'img_wiener': 'arrlp.modules.image_LP._functions.img_wiener',
'img_fft': 'arrlp.modules.image_LP._functions.img_fft',
'img_radialproj': 'arrlp.modules.image_LP._functions.img_radialproj',
'img_convolve1d': 'arrlp.modules.image_LP._functions.img_convolve1d',
'img_correlate': 'arrlp.modules.image_LP._functions.img_correlate',
'img_greyopening': 'arrlp.modules.image_LP._functions.img_greyopening',
'img_correlate1d': 'arrlp.modules.image_LP._functions.img_correlate1d',
'sig_wiener': 'arrlp.modules.signal_LP._functions.sig_wiener',
'sig_correlate': 'arrlp.modules.signal_LP._functions.sig_correlate',
'sig_ifft': 'arrlp.modules.signal_LP._functions.sig_ifft',
'sig_gaussianfilter': 'arrlp.modules.signal_LP._functions.sig_gaussianfilter',
'sig_greyopening': 'arrlp.modules.signal_LP._functions.sig_greyopening',
'sig_convolve': 'arrlp.modules.signal_LP._functions.sig_convolve',
'sig_crosscorr': 'arrlp.modules.signal_LP._functions.sig_crosscorr',
'sig_autocorr': 'arrlp.modules.signal_LP._functions.sig_autocorr',
'sig_fft': 'arrlp.modules.signal_LP._functions.sig_fft',
'vol_crosscorr': 'arrlp.modules.volume_LP._functions.vol_crosscorr',
'vol_ifft': 'arrlp.modules.volume_LP._functions.vol_ifft',
'vol_gaussianfilter': 'arrlp.modules.volume_LP._functions.vol_gaussianfilter',
'vol_correlate': 'arrlp.modules.volume_LP._functions.vol_correlate',
'vol_fft': 'arrlp.modules.volume_LP._functions.vol_fft',
'vol_convolve1d': 'arrlp.modules.volume_LP._functions.vol_convolve1d',
'vol_correlate1d': 'arrlp.modules.volume_LP._functions.vol_correlate1d',
'vol_wiener': 'arrlp.modules.volume_LP._functions.vol_wiener',
'vol_autocorr': 'arrlp.modules.volume_LP._functions.vol_autocorr',
'vol_greyopening': 'arrlp.modules.volume_LP._functions.vol_greyopening',
'vol_convolve': 'arrlp.modules.volume_LP._functions.vol_convolve'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)