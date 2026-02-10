#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-09
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : template_[library]
# Module        : image

"""
This file allows to test image

image : Image array functions of base shape (y, x).
"""



# %% Libraries
from corelp import print, debug
import pytest
from template_[lowerlib] import image
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test image function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return image()

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
    Test image return values
    '''
    assert image(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test image error values
    '''
    with pytest.raises(error, match=error_message) :
        image(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)