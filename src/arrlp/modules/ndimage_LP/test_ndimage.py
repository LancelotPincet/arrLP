#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-10
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : template_[library]
# Module        : ndimage

"""
This file allows to test ndimage

ndimage : Gets scipy.ndimage of cupyx.scipy.ndimage depending on the value of cuda argument.
"""



# %% Libraries
from corelp import print, debug
import pytest
from template_[lowerlib] import ndimage
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test ndimage function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return ndimage()

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
    Test ndimage return values
    '''
    assert ndimage(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test ndimage error values
    '''
    with pytest.raises(error, match=error_message) :
        ndimage(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)