#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-09
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : volume

"""
Volume array functions of base (z, y, x).
"""



# %% Libraries
from arrlp import *



# %% Function
def volume(**kwargs) :
    '''
    Volume array functions of base (z, y, x).
    
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
    >>> from arrlp import volume
    ...
    >>> volume() # TODO
    '''

    return None



# %% Libraries
from corelp import prop
from arrlp import *
from dataclasses import dataclass, field



# %% Class
@dataclass(slots=True, kw_only=True)
class volume() :
    '''
    Volume array functions of base (z, y, x).
    
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
    >>> from arrlp import volume
    ...
    >>> instance = volume(TODO)
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