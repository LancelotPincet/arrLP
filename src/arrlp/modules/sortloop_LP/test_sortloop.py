#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-18
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : sortloop

"""
This file allows to test sortloop

sortloop : This generator allows to loop on unique values of an array rapidly by first sorting the array.
"""



# %% Libraries
from corelp import debug
import pytest
from arrlp import sortloop
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test sortloop function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return sortloop()

def test_instance(instance) :
    '''
    Test on fixture
    '''
    pass


# %% Returns test
@pytest.mark.parametrize("args, kwargs, expected, message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_returns(args, kwargs, expected, message) :
    '''
    Test sortloop return values
    '''
    assert sortloop(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test sortloop error values
    '''
    with pytest.raises(error, match=error_message) :
        sortloop(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)