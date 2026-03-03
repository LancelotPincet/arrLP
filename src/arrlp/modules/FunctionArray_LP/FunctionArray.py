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
from joblib import Parallel, delayed
from numba import njit, prange
import skimage

# Check cupy
try :
    import cupy as cp
    import cupyx.scipy as cupyx_scipy
    from cupyx.scipy import ndimage as cupyx_ndimage
    import cucim.skimage as cucim_skimage
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
    cpu_function : types.FunctionType
        Cpu function.
    par_function : types.FunctionType
        Par function.
    gpu_function : types.FunctionType
        Gpu function.
    out_function : types.FunctionType
        Function defining output array.
    ini_function : types.FunctionType
        Function defining initialization kwargs.
    cpu_loop : bool
        True if cpu_function applyies default loop.
    par_loop : bool
        True if par_function applyies default loop.
    gpu_loop : bool
        True if gpu_function applyies default loop.
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
    cpu_function : types.FunctionType = field(repr=False)
    par_function : types.FunctionType = field(repr=False)
    gpu_function : types.FunctionType = field(repr=False)
    out_function : types.FunctionType = field(repr=False)
    ini_function : types.FunctionType = field(repr=False)

    # Loops
    cpu_loop : types.FunctionType = field(default=False, repr=False)
    par_loop : types.FunctionType = field(default=False, repr=False)
    gpu_loop : types.FunctionType = field(default=False, repr=False)
    use_joblib : bool = field(default=True, repr=False)

    # Performances
    remove_parallel : bool = field(default=False, repr=False)
    remove_cuda : bool = field(default=False, repr=False)



    # Methods
    stacks : bool = field(init=False, repr=False)
    channels : bool = field(init=False, repr=False)
    parallel : bool = field(init=False, repr=False)
    cuda : bool = field(init=False, repr=False)
    def __call__(self, array, *args, out=None, # Arrays
        stacks=False, channels=False, parallel=False, cuda=False, test=False, iterator=range, # Modes
        **kwargs) : # Arguments of the function
        '''
        This is the main function call
    
        Parameters
        ----------
        array : np.array
            At least one array from which we derive stacks and channels.
        *args : tuple(np.array)
            Tuple of array corresponding to inputs that follow the same stacks and channels dimensions.
        out : np.array
            Output array where to put result, might not be always available
        stacks : bool
            True to consider stacks dimensions
        channels : bool
            True to consider channels dimensions
        parallel : bool
            True to optimize in parallel cpu cores
        cuda : bool
            True to optimize on gpu
        test : bool
            [dev only], True when testing the speed to avoid raising errors on function that were defined as not optimal
        iterator : iterator
            Iterator defining how to loop on dimensions, must take an int as only input, allow to put progress bars.
        **kwargs : dict
            Other constant inputs to apply on each array

        Returns
        -------
        out : np.array
            Output array.
        '''

        # checks
        if iterator is None : iterator = range
        if parallel is True : parallel = -1
        self.checks(out, stacks, channels, parallel, cuda, test)
        self.stacks, self.channels, self.parallel, self.cuda = stacks, channels, parallel, cuda
        array = self.xp.asarray(array)
        out = out if out is not None else self.out_function(self, array, *args, **kwargs) if self.out_function is not None else None
        if self.ini_function is not None :
            kwargs.update(self.ini_function(self, array, *args, **kwargs))

        # Cuda
        if cuda :
            if self.gpu_loop :
                return self.loop(self.gpu_function, iterator, out, array, *args, **kwargs)
            return self.gpu_function(self, out, array, *args, **kwargs)

        # Parallel
        elif parallel :
            if self.par_loop :
                if self.use_joblib :
                    return self.parallel_loop(self.par_function, out, array, *args, **kwargs)
                else :
                    return self.loop(self.par_function, iterator, out, array, *args, **kwargs)
            return self.par_function(self, out, array, *args, **kwargs)

        # Python
        else :
            if self.cpu_loop :
                return self.loop(self.cpu_function, iterator, out, array, *args, **kwargs)
            return self.cpu_function(self, out, array, *args, **kwargs)



    # Properties
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
    @property
    def skimagex(self) :
        return cucim_skimage if self.cuda else skimage



    def checks(self, out, stacks, channels, parallel, cuda, test) :
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
        if parallel and not stacks and not channels and self.use_joblib :
            raise ValueError('Normal array (no stack of channel) cannot be calculated in parallel')

        # Inplace out not possible
        if out is not None and self.out_function is None and not stacks and not channels :
            raise ValueError('Output cannot be defined in this function for normal single array')



    def loop(self, func, iterator, out, array, *args, **kwargs) :
    
        nstacks, nchannels = array.shape[0], array.shape[-1]
        match (self.stacks, self.channels) :

            case (False, False) :
                return func(self, out, array, *args, **kwargs)
            case (True, False) :
                for i in iterator(nstacks) :
                    stack_out = None if out is None else out[i]
                    _stack_out = func(self, stack_out, array[i], *(arg[i] for arg in args), **kwargs)
                    if out is None :
                        if i == 0 : _out = self.xp.empty_like(_stack_out, shape=(nstacks, *_stack_out.shape))
                        _out[i] = _stack_out
                return _out if out is None else out
            case (False, True) :
                for j in iterator(nchannels) :
                    channel_out = None if out is None else out[..., j]
                    _channel_out = func(self, channel_out, array[..., j], *(arg[..., j] for arg in args), **kwargs)
                    if out is None :
                        if j == 0 : _out = self.xp.empty_like(_channel_out, shape=(*_channel_out.shape, nchannels))
                        _out[..., j] = _channel_out
                return _out if out is None else out
            case (True, True) :
                for i in iterator(nstacks) :
                    stack_out, stack_array, stack_args = None if out is None else out[i], array[i], (arg[i] for arg in args)
                    if out is None and i > 0 : _stack_out = _out[i]
                    for j in range(nchannels) :
                        channel_out = None if stack_out is None else stack_out[..., j]
                        _channel_out = func(self, channel_out, stack_array[..., j], *(arg[..., j] for arg in stack_args), **kwargs)
                        if out is None :
                            if i == 0 and j == 0 : 
                                _out = self.xp.empty_like(_channel_out, shape=(nstacks, *_channel_out.shape, nchannels))
                                _stack_out = _out[i]
                            _stack_out[..., j] = _channel_out
                return _out if out is None else out

            case _ : raise SyntaxError(f'Cannot use (stacks, channels, out_name)={(self.stacks, self.channels, out_name)}')



    def parallel_loop(self, func, out, array, *args, **kwargs) :

        nstacks, nchannels = array.shape[0], array.shape[-1]
        match (self.stacks, self.channels) :

            case (False, False) :
                raise SyntaxError('This parallel scenario with no stack nor channel should not exist. [should be corrected in checks]')
            case (True, False) :
                copy = Parallel(n_jobs=self.parallel, backend="loky")(delayed(func)(self, array[i], *(arg[i] for arg in args), **kwargs) for i in range(nstacks))
                if out is None : out = np.empty_like(copy[0], shape=(len(copy), *copy[0].shape))
                return copystacks(list(copy), out)
            case (False, True) :
                copy = Parallel(n_jobs=self.parallel, backend="loky")(delayed(func)(self, array[..., i], *(arg[..., i] for arg in args), **kwargs) for i in range(nchannels))
                if out is None : out = np.empty_like(copy[0], shape=(*copy[0].shape, len(copy)))
                return copychannels(list(copy), out)
            case (True, True) :
                newfunc = lambda self, array, *args, **kwargs : [func(self, array[..., j], *(arg[..., j] for arg in args), **kwargs) for j in range(nchannels)]
                copy = Parallel(n_jobs=self.parallel, backend="loky")(delayed(newfunc)(self, array[i], *(arg[i] for arg in args), **kwargs) for i in range(nstacks))
                if out is None : out = np.empty_like(copy[0][0], shape=(len(copy), *copy[0][0].shape, len(copy[0])))
                return copystacksnchannels(list(copy), out)
            
            case _ : raise SyntaxError(f'Cannot use (stacks, channels)={(self.stacks, self.channels)}')



@njit(parallel=True)
def copystacks(copyfrom, copyto):
    for i in prange(len(copyfrom)):
        copyto[i] = copyfrom[i]
    return copyto

@njit(parallel=True)
def copychannels(copyfrom, copyto):
    for j in prange(len(copyfrom)):
        copyto[..., j] = copyfrom[j]
    return copyto

def copystacksnchannels(copyfrom, copyto):
    for i in prange(len(copyfrom)):
        stack_copyfrom, stack_copyto = copyfrom[i], copyto[i]
        for j in range(len(stack_copyfrom)) :
            stack_copyto[..., j] = stack_copyfrom[j]
    return copyto



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)