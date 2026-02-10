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
'compress': 'arrlp.modules.compress_LP.compress',
'coordinates': 'arrlp.modules.coordinates_LP.coordinates',
'image': 'arrlp.modules.image_LP.image',
'kernel': 'arrlp.modules.kernel_LP.kernel',
'ndimage': 'arrlp.modules.ndimage_LP.ndimage',
'parallel_array': 'arrlp.modules.parallel_array_LP.parallel_array',
'signal': 'arrlp.modules.signal_LP.signal',
'volume': 'arrlp.modules.volume_LP.volume',
'xp': 'arrlp.modules.xp_LP.xp',
'img_wiener': 'arrlp.modules.image_LP._functions.img_wiener',
'img_correlate': 'arrlp.modules.image_LP._functions.img_correlate',
'img_fft': 'arrlp.modules.image_LP._functions.img_fft',
'img_ifft': 'arrlp.modules.image_LP._functions.img_ifft'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)