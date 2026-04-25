#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : transform_parameters

"""
This file allows to test transform_parameters

transform_parameters : Defines the affine transformations in 2D for a 3x3 transformation matrix.
"""



# %% Libraries
from corelp import debug
import pytest
from arrlp import transform_parameters
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test transform_parameters function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return transform_parameters()

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
    Test transform_parameters return values
    '''
    assert transform_parameters(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test transform_parameters error values
    '''
    with pytest.raises(error, match=error_message) :
        transform_parameters(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)