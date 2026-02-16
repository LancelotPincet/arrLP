#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-13
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : template_[library]
# Module        : FunctionArray

"""
This file allows to test FunctionArray

FunctionArray : This class defines a function for various array configurations.
"""



# %% Libraries
from corelp import print, debug
import pytest
from template_[lowerlib] import FunctionArray
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test FunctionArray function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return FunctionArray()

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
    Test FunctionArray return values
    '''
    assert FunctionArray(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test FunctionArray error values
    '''
    with pytest.raises(error, match=error_message) :
        FunctionArray(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)