#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
misc: Various auxilliary functions, part of the colour package

Copyright (C) 2013-2016 Ivar Farup

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

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Auxiliary functions
# =============================================================================

def plot_ellipses(ellipses, axis=None, alpha=1,
                  facecolor=[.5, .5, .5], edgecolor=[0, 0, 0], fill=False):
    """
    Plot the list of ellipses on the given axis.

    Parameters
    ----------
    ellipses : list
        List of Ellipse objects.
    axis : AxesSubplot
        Axis on which to plot the ellipses.
    alpha : float
        Alpha for colour blending.
    facecolor : ndarray
        RGB array of the ellipse faces.
    edgecolor : ndarray
        RGB array of the ellipse edges.
    fill : bool
        Fill the ellipses or not.
    """
    if axis is None:
        axis = plt.gca()
    for e in ellipses:
        axis.add_artist(e)
        e.set_clip_box(axis.bbox)
        e.set_alpha(alpha)
        e.set_facecolor(facecolor)
        e.set_edgecolor(edgecolor)
        e.set_fill(fill)


def safe_div(a, b, fill=1.):
    """
    Divide the two arrays, filling with fill value where denominator is zero.

    Parameters
    ----------
    a : ndarray
        The nominator
    b : ndarray
        The denominator
    fill : float
        The number to fill where the denominator is zeros

    Returns
    -------
    res : ndarray
        The quotient a / b filled with fill value where b == 0-
    """
    res = np.zeros(np.shape(a))
    res[b != 0] = a[b != 0] / b[b != 0]
    res[b == 0] = fill
    return res


def inner(data1, data2, tensor):
    """
    Compute the inner products of two datasets with a given metric tensor.

    The data sets and the tensor data set must have corresponding dimensions.

    Parameters
    ----------
    data1: ndarray
        The first dataset of the inner product
    data2: ndarray
        The second data set of the inner product
    tensor: ndarray
        The metric tensor for the inner product

    Returns
    -------
    inner_product: ndarray
        Array with numerical values for the inner product
    """
    dot1 = np.zeros(np.shape(data1))
    for i in range(3):
        dot1[:, i] = np.sum(data1 * tensor[..., i], -1)
    return np.sum(dot1 * data2, -1)


def norm_sq(data, tensor):
    """
    Compute the squared norm of a colour data set with a given metric tensor.

    The data set and the tensor data set must have corresponding dimensions.

    Parameters
    ----------
    data: ndarray
        The data set of which to compute the squared norm
    tensor: ndarray
        The metric tensor for the norm.

    Returns
    -------
    norms: ndarray
        Array with numerical (scalar) values of the squared norm.
    """
    return inner(data, data, tensor)


def norm(data, tensor):
    """
    Compute the norm of a colour data set with a given metric tensor.

    The data set and the tensor data set must have corresponding dimensions.

    Parameters
    ----------
    data: ndarray
        The data set of which to compute the norm
    tensor: ndarray
        The metric tensor for the norm.

    Returns
    -------
    norms: ndarray
        Array with numerical (scalar) values of the norm.
    """
    return np.sqrt(norm_sq(data, tensor))
