#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-06
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : gc

"""
This file allows to test gc

gc : Makes garbage collection to free (V)RAM.
"""



# %% Libraries
from corelp import debug
import pytest
from arrlp import gc
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test gc function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return gc()

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
    Test gc return values
    '''
    assert gc(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test gc error values
    '''
    with pytest.raises(error, match=error_message) :
        gc(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)