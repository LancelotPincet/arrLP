#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-12
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : check_optimizations

"""
This modules checks the logic behind optimization inputs.
"""



# %% Libraries
try :
    import cupy as cp
except ImportError :
    cp = None



# %% Function
def check_optimizations(stacks=False, channels=False, parallel=False, cuda=False, *, remove_parallel=False, remove_cuda=False) :
    '''
    This modules checks the logic behind optimization inputs.
    
    Parameters
    ----------
    stacks : bool
        True if stacks is asked.
    channels : bool
        True if channels is asked.
    parallel : bool
        True if parallel is asked.
    cuda : bool
        True if cuda is asked.
    remove_parallel : bool
        True if stacks is asked.
    remove_cuda : bool
        True if stacks is asked.

    Examples
    --------
    >>> from arrlp import check_optimizations
    ...
    >>> check_optimizations(stacks, channels, parallel, cuda, remove_parallel=False, remove_cuda=False)
    '''

    # No parallel 
    if parallel and not stacks and not channels :
        raise ValueError('Normal array (no stack of channel) cannot be calculated in parallel')
    if parallel and cuda :
        raise SyntaxError('Cuda and Parallel cannot be True at the same time')
    # if parallel :
    #     import warnings
    #     warnings.warn('Parallel optimization is not effective in this function')
    # if cuda :
    #     import warnings
    #     warnings.warn('Cuda optimization is not effective in this function')
        



# %% Libraries
from corelp import prop
from arrlp import *
from dataclasses import dataclass, field



# %% Class
@dataclass(slots=True, kw_only=True)
class check_optimizations() :
    '''
    This modules checks the logic behind optimization inputs.
    
    Parameters
    ----------
    a : int or float
        TODO.

    Attributes
    ----------
    _attr : int or float
        TODO.

    Examples
    --------
    >>> from arrlp import check_optimizations
    ...
    >>> instance = check_optimizations(TODO)
    '''

    # Attributes
    # myattr : str = ""
    # mylist : list[str] = field(default_factory=list)
    # _mycal : str = field(init=False, repr=False) 
    name : str = None



    # Init
    def __post_init__(self) :
        pass



    # Properties
    @prop(cache=True)
    def myprop(self) :
        return ""
    @myprop.setter()
    def mypropr(self, value) :
        self._myprop = value



    # Methods
    def method(self) :
        '''
        TODO
    
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

        return None



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)