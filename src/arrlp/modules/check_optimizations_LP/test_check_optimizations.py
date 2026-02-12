#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-12
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : template_[library]
# Module        : check_optimizations

"""
This file allows to test check_optimizations

check_optimizations : This modules checks the logic behind optimization inputs.
"""



# %% Libraries
from corelp import print, debug
import pytest
from template_[lowerlib] import check_optimizations
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test check_optimizations function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return check_optimizations()

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
    Test check_optimizations return values
    '''
    assert check_optimizations(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test check_optimizations error values
    '''
    with pytest.raises(error, match=error_message) :
        check_optimizations(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)