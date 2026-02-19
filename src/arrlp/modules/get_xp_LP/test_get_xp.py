#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : template_[library]
# Module        : get_xp

"""
This file allows to test get_xp

get_xp : This function returns numpy/cupy depending on the input.
"""



# %% Libraries
from corelp import print, debug
import pytest
from template_[lowerlib] import xp
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test get_xp function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return get_xp()

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
    Test get_xp return values
    '''
    assert get_xp(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test get_xp error values
    '''
    with pytest.raises(error, match=error_message) :
        get_xp(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)