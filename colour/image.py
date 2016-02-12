#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
image: Colour image, part of the colour package

Copyright (C) 2013 Ivar Farup

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

#==============================================================================
# Functions for FDM on images
#==============================================================================


# Shifted images

def ip1(im):
    """
    Image shifted one pixel positive i
    """
    sh = np.shape(im)
    m = sh[0]
    return im[np.r_[np.arange(1, m), m - 1], ...]


def im1(im):
    """
    Image shifted one pixel negative i
    """
    sh = np.shape(im)
    m = sh[0]
    return im[np.r_[0, np.arange(0, m - 1)], ...]


def jp1(im):
    """
    Image shifted one pixel positive j
    """
    sh = np.shape(im)
    n = sh[1]
    return im[:, np.r_[np.arange(1, n), n - 1], ...]


def jm1(im):
    """
    Image shifted one pixel negative j
    """
    sh = np.shape(im)
    n = sh[1]
    return im[:, np.r_[0, np.arange(0, n - 1)], ...]


# Finite differences

def dip1(im):
    """
    Finite difference positive i
    """
    sh = np.shape(im)
    m = sh[0]
    return im[np.r_[np.arange(1, m), m - 1], ...] - im


def dim1(im):
    """
    Finite difference negative i
    """
    sh = np.shape(im)
    m = sh[0]
    return im - im[np.r_[0, np.arange(0, m-1)], ...]


def dic1(im):
    """
    Finite difference centered i
    """
    sh = np.shape(im)
    m = sh[0]
    return 0.5 * (im[np.r_[np.arange(1, m), m - 1], ...] -
                  im[np.r_[0, np.arange(0, m-1)], ...])


def djp1(im):
    """
    Finite difference positive j
    """
    sh = np.shape(im)
    n = sh[1]
    return im[:, np.r_[np.arange(1, n), n - 1], ...] - im


def djm1(im):
    """
    Finite difference negative j
    """
    sh = np.shape(im)
    n = sh[1]
    return im - im[:, np.r_[0, np.arange(0, n - 1)], ...]


def djc1(im):
    """
    Finite difference centered j
    """
    sh = np.shape(im)
    n = sh[1]
    return 0.5 * (im[:, np.r_[np.arange(1, n), n - 1], ...] -
                  im[:, np.r_[0, np.arange(0, n - 1)], ...])
