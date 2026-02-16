#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-13
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : FunctionArray

"""
This class defines a function for various array configurations.
"""



# %% Libraries
import types
from dataclasses import dataclass, field
import numpy as np
import scipy

# Check cupy
try :
    import cupy as cp
    import cupyx.scipy as cupyx_scipy
    from cupyx.scipy import ndimage as cupyx_ndimage
except ImportError :
    cp = None



# %% Class
@dataclass(slots=True, kw_only=True)
class FunctionArray() :
    '''
    This class defines a function for various array configurations.
    
    Parameters
    ----------
    ndims : int
        Number of dimensions of the base array.
    function : types.FunctionType
        Default function used if no override.
    out_function : types.FunctionType
        Function defining output array.
    _cpu_function : types.FunctionType
        Cpu function override.
    _par_function : types.FunctionType
        Par function override.
    _gpu_function : types.FunctionType
        Gpu function override.
    cpu_out_name : str
        Name of argument of inplace output for cpu_function.
    par_out_name : str
        Name of argument of inplace output for par_function.
    gpu_out_name : str
        Name of argument of inplace output for gpu_function.
    cpu_loop : bool
        True if cpu_function applyies default loop.
    par_loop : bool
        True if par_function applyies default loop.
    gpu_loop : bool
        True if gpu_function applyies default loop.
    cpu_factory : types.FunctionType
        Function that creates numba compiled functions on cpu.
    par_factory : types.FunctionType
        Function that creates numba compiled functions on parallel.
    gpu_factory : types.FunctionType
        Function that creates numba compiled functions on gpu.
    cpu_axes_name : str
        Name of argument of axes definition for cpu_function.
    par_axes_name : str
        Name of argument of axes definition for par_function.
    gpu_axes_name : str
        Name of argument of axes definition for gpu_function.
    use_joblib : bool
        True to use joblib for parallel processes in parallel loop.
    remove_parallel : bool
        True if the parallel implementation is slower than normal python.
    remove_cuda : bool
        True if the cuda implementation is slower than normal python.

    Examples
    --------
    >>> from arrlp import FunctionArray
    ...
    >>> instance = FunctionArray(TODO)
    '''

    # Mandatory
    ndims : int
    function : types.FunctionType = field(repr=False)
    out_function : types.FunctionType = field(repr=False)

    # Functions overrides
    _cpu_function : types.FunctionType = field(default=None, repr=False)
    _par_function : types.FunctionType = field(default=None, repr=False)
    _gpu_function : types.FunctionType = field(default=None, repr=False)

    # Output
    cpu_out_name : str = field(default='out', repr=False)
    par_out_name : str = field(default='out', repr=False)
    gpu_out_name : str = field(default='out', repr=False)

    # Loops
    cpu_loop : types.FunctionType = field(default=False, repr=False)
    par_loop : types.FunctionType = field(default=False, repr=False)
    gpu_loop : types.FunctionType = field(default=False, repr=False)

    # Numba
    cpu_factory : types.FunctionType = field(default=None, repr=False)
    par_factory : types.FunctionType = field(default=None, repr=False)
    gpu_factory : types.FunctionType = field(default=None, repr=False)

    # Axes
    cpu_axes_name : bool = field(default=None, repr=False)
    par_axes_name : bool = field(default=None, repr=False)
    gpu_axes_name : bool = field(default=None, repr=False)

    # Parallel loops
    use_joblib : bool = field(default=True, repr=False)

    # Performances
    remove_parallel : bool = field(default=False, repr=False)
    remove_cuda : bool = field(default=False, repr=False)



    # Init
    def __post_init__(self) :
        pass



    # Methods
    def __call__(self, *args, out=None, # Arrays
        stacks=False, channels=False, parallel=False, cuda=False, test=False, iterator=range, # Modes
        **kwargs) : # Arguments of the function
        '''
        This is the main function call
    
        Parameters
        ----------
        a : int or float
            TODO.

        Returns
        -------
        b : int or float
            TODO.

        Raises
        ------
        TypeError
            TODO.

        Examples
        --------
        >>> self.method() # TODO
        '''

        # checks
        self.checks(stacks, channels, parallel, cuda, test)
        self.stacks, self.channels, self.parallel, self.cuda = stacks, channels, parallel, cuda

        # Cuda
        if cuda :
            if self.gpu_axes_name :
                if self.gpu_out_name is not None : kwargs[self.gpu_out_name] = self.out_function(*args, **kwargs) if out is None else out
                elif out is not None : raise SyntaxError('Output cannot be defined in this gpu function')
                kwargs[self.gpu_axes_name] = self.axes
                return self.gpu_function(*args, **kwargs)
            if self.gpu_factory is not None :
                if self.gpu_out_name is not None : kwargs[self.gpu_out_name] = self.out_function(*args, **kwargs) if out is None else out
                elif out is not None : raise SyntaxError('Output cannot be defined in this gpu function')
                return self.gpu_function(*args, **kwargs)
            if self.gpu_loop :
                out = self.out_function(*args, **kwargs) if out is None else out
                return self.loop(self.gpu_function, iterator, out, self.gpu_out_name, *args, **kwargs)
            raise SyntaxError('Gpu calculation was not found')

        # Parallel
        elif parallel :
            if self.par_axes_name :
                if self.par_out_name is not None : kwargs[self.par_out_name] = self.out_function(*args, **kwargs) if out is None else out
                elif out is not None : raise SyntaxError('Output cannot be defined in this parallel function')
                kwargs[self.par_axes_name] = self.axes
                return self.par_function(*args, **kwargs)
            if self.par_factory is not None :
                if self.par_out_name is not None : kwargs[self.par_out_name] = self.out_function(*args, **kwargs) if out is None else out
                elif out is not None : raise SyntaxError('Output cannot be defined in this parallel function')
                return self.gpu_function(*args, **kwargs)
            if self.par_loop :
                if self.use_joblib :
                    return self.parallel_loop(self.par_function, iterator, self.par_out_name, *args, **kwargs)
                else :
                    out = self.out_function(*args, **kwargs) if out is None else out
                    return self.loop(self.par_function, iterator, out, self.par_out_name, *args, **kwargs)
            raise SyntaxError('Parallel calculation was not found')

        # Python
        else :
            if self.cpu_axes_name :
                if self.gpu_out_name is not None : kwargs[self.gpu_out_name] = self.out_function(*args, **kwargs) if out is None else out
                elif out is not None : raise SyntaxError('Output cannot be defined in this gpu function')
                kwargs[self.cpu_axes_name] = self.axes
                return self.cpu_function(*args, **kwargs)
            if self.cpu_factory is not None :
                return self.cpu_function(*args, **kwargs)
            if self.cpu_loop :
                out = self.out_function(*args, **kwargs) if out is None else out
                return self.loop(self.cpu_function, iterator, out, self.cpu_out_name, *args, **kwargs)
            raise SyntaxError('Cpu calculation was not found')



    # Properties
    cpu_none_numba : str = field(default=None, init=False, repr=False)
    cpu_stack_numba : str = field(default=None, init=False, repr=False)
    cpu_channel_numba : str = field(default=None, init=False, repr=False)
    cpu_full_numba : str = field(default=None, init=False, repr=False)
    @property
    def cpu_function(self) :
        if self.cpu_factory is not None :
            match (self.stacks, self.channels) :
                case (True, True) :
                    if self.cpu_full_numba is None :
                        self.cpu_full_numba = self.cpu_factory(self.stacks, self.channels)
                    return self.cpu_full_numba
                case (False, False) :
                    if self.cpu_none_numba is None :
                        self.cpu_none_numba = self.cpu_factory(self.stacks, self.channels)
                    return self.cpu_none_numba
                case (True, False) :
                    if self.cpu_stack_numba is None :
                        self.cpu_stack_numba = self.cpu_factory(self.stacks, self.channels)
                    return self.cpu_stack_numba
                case (False, True) :
                    if self.cpu_channel_numba is None :
                        self.cpu_channel_numba = self.cpu_factory(self.stacks, self.channels)
                    return self.cpu_channel_numba
        if self.function is None and self._cpu_function is None : raise SyntaxError('Cpu function was not defined')
        return self.function if self._cpu_function is None else self._cpu_function
    par_none_numba : str = field(default=None, init=False, repr=False)
    par_stack_numba : str = field(default=None, init=False, repr=False)
    par_channel_numba : str = field(default=None, init=False, repr=False)
    par_full_numba : str = field(default=None, init=False, repr=False)
    @property
    def par_function(self) :
        if self.par_factory is not None :
            match (self.stacks, self.channels) :
                case (True, True) :
                    if self.par_full_numba is None :
                        self.par_full_numba = self.par_factory(self.stacks, self.channels)
                    return self.par_full_numba
                case (False, False) :
                    if self.par_none_numba is None :
                        self.par_none_numba = self.par_factory(self.stacks, self.channels)
                    return self.par_none_numba
                case (True, False) :
                    if self.par_stack_numba is None :
                        self.par_stack_numba = self.par_factory(self.stacks, self.channels)
                    return self.par_stack_numba
                case (False, True) :
                    if self.par_channel_numba is None :
                        self.par_channel_numba = self.par_factory(self.stacks, self.channels)
                    return self.par_channel_numba
        if self.function is None and self._par_function is None : raise SyntaxError('Parallel function was not defined')
        return self.function if self._par_function is None else self._par_function
    gpu_none_numba : str = field(default=None, init=False, repr=False)
    gpu_stack_numba : str = field(default=None, init=False, repr=False)
    gpu_channel_numba : str = field(default=None, init=False, repr=False)
    gpu_full_numba : str = field(default=None, init=False, repr=False)
    @property
    def gpu_function(self) :
        if self.gpu_factory is not None :
            match (self.stacks, self.channels) :
                case (True, True) :
                    if self.gpu_full_numba is None :
                        self.gpu_full_numba = self.gpu_factory(self.stacks, self.channels)
                    return self.gpu_full_numba
                case (False, False) :
                    if self.gpu_none_numba is None :
                        self.gpu_none_numba = self.gpu_factory(self.stacks, self.channels)
                    return self.gpu_none_numba
                case (True, False) :
                    if self.gpu_stack_numba is None :
                        self.gpu_stack_numba = self.gpu_factory(self.stacks, self.channels)
                    return self.gpu_stack_numba
                case (False, True) :
                    if self.gpu_channel_numba is None :
                        self.gpu_channel_numba = self.gpu_factory(self.stacks, self.channels)
                    return self.gpu_channel_numba
        if self.function is None and self._gpu_function is None : raise SyntaxError('Gpu function was not defined')
        return self.function if self._gpu_function is None else self._gpu_function
    @property
    def axes(self) :
        start = int(self.stacks)
        return tuple(range(start, start + self.ndims))
    @property
    def xp(self) :
        return cp if self.cuda else np
    @property
    def scipyx(self) :
        return cupyx_scipy if self.cuda else scipy
    @property
    def ndimage(self) :
        return cupyx_ndimage if self.cuda else scipy.ndimage



    def checks(self, stacks, channels, parallel, cuda, test) :
        '''
        Make checks on asked mode
        '''

        # One optimization
        if parallel and cuda :
            raise ValueError('Cuda and Parallel cannot be True at the same time')
        
        # Cuda not available
        if cuda and cp is None :
            raise ValueError('Cuda was asked but is not available in this environment')

        # No parallel
        if parallel and self.remove_parallel and not test :
            raise ValueError('Parallel optimization is not effective in this function')

        # No cuda
        if cuda and self.remove_cuda and not test :
            raise ValueError('Cuda optimization is not effective in this function')

        # Joblib when no additional channels 
        if parallel and not stacks and not channels and self.par_loop:
            raise ValueError('Normal array (no stack of channel) cannot be calculated in parallel')



    def loop(self, func, iterator, array, *args, out=None, out_name=None, **kwargs) :
    
        nstacks, nchannels = array.shape[0], array.shape[-1]
        if self.gpu_out_name is not None : kwargs[self.gpu_out_name] = self.out_function(*args, **kwargs) if out is None else out
        match (self.stacks, self.channels) :
            case (False, False) :
                out[:] = func(array)
            case (True, False) :
                for i in iterator(nstacks) :
                    out[i] = func(array[i])
            case (False, True) :
                for j in iterator(nchannels) :
                    out[..., j] = func(array[..., j])
            case (True, True) :
                for i in iterator(nstacks) :
                    a, o = array[i], out[i]
                    for j in range(nchannels) :
                        o[..., j] = func(a[..., j])
        return out



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)