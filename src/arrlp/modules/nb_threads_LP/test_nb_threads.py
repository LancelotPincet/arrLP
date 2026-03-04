#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-04
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : arrLP
# Module        : nb_threads

"""
This file allows to test nb_threads

nb_threads : This context manager defines the number of logical cores on numba parallel threads.
"""



# %% Libraries
from corelp import debug
import pytest
from arrlp import nb_threads
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test nb_threads function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return nb_threads()

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
    Test nb_threads return values
    '''
    assert nb_threads(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test nb_threads error values
    '''
    with pytest.raises(error, match=error_message) :
        nb_threads(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)