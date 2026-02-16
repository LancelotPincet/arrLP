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
'image': 'arrlp.modules.image_LP.image',
'kernel': 'arrlp.modules.kernel_LP.kernel',
'signal': 'arrlp.modules.signal_LP.signal',
'volume': 'arrlp.modules.volume_LP.volume',
'img_ifft': 'arrlp.modules.image_LP._functions.img_ifft',
'img_gaussianfilter': 'arrlp.modules.image_LP._functions.img_gaussianfilter',
'img_fft': 'arrlp.modules.image_LP._functions.img_fft',
'img_correlate': 'arrlp.modules.image_LP._functions.img_correlate',
'img_correlate1d': 'arrlp.modules.image_LP._functions.img_correlate1d'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)