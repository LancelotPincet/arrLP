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
'axes': 'arrlp.modules.axes_LP.axes',
'check_optimizations': 'arrlp.modules.check_optimizations_LP.check_optimizations',
'compress': 'arrlp.modules.compress_LP.compress',
'coordinates': 'arrlp.modules.coordinates_LP.coordinates',
'debug_array': 'arrlp.modules.debug_array_LP.debug_array',
'image': 'arrlp.modules.image_LP.image',
'kernel': 'arrlp.modules.kernel_LP.kernel',
'ndimage': 'arrlp.modules.ndimage_LP.ndimage',
'parallel_array': 'arrlp.modules.parallel_array_LP.parallel_array',
'scipyx': 'arrlp.modules.scipyx_LP.scipyx',
'signal': 'arrlp.modules.signal_LP.signal',
'volume': 'arrlp.modules.volume_LP.volume',
'xp': 'arrlp.modules.xp_LP.xp',
'img_ifft': 'arrlp.modules.image_LP._functions.img_ifft',
'img_wiener': 'arrlp.modules.image_LP._functions.img_wiener',
'img_fft': 'arrlp.modules.image_LP._functions.img_fft',
'img_correlate': 'arrlp.modules.image_LP._functions.img_correlate',
'sig_wiener': 'arrlp.modules.signal_LP._functions.sig_wiener',
'sig_correlate': 'arrlp.modules.signal_LP._functions.sig_correlate',
'sig_ifft': 'arrlp.modules.signal_LP._functions.sig_ifft',
'sig_fft': 'arrlp.modules.signal_LP._functions.sig_fft',
'vol_ifft': 'arrlp.modules.volume_LP._functions.vol_ifft',
'vol_fft': 'arrlp.modules.volume_LP._functions.vol_fft'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)