#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-12
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : template_[library]
# Module        : debug_array

"""
This file allows to test debug_array

debug_array : This function allows to debug array functions while comparing various modes.
"""



# %% Libraries
from corelp import print, debug
import pytest
from template_[lowerlib] import debug_array
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test debug_array function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return debug_array()

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
    Test debug_array return values
    '''
    assert debug_array(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test debug_array error values
    '''
    with pytest.raises(error, match=error_message) :
        debug_array(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)