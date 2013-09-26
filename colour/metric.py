#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metric: Colour metric functions. Part of the colour package.

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
import data
import space
import tensor

#==============================================================================
# Colour metric functions
#==============================================================================

def linear(sp, dat1, dat2, metric_tensor_function):
    """
    Compute the linearised colour difference between the two data sets.
    
    The function metric_tensor_function is used to compute the metric tensor
    at the midpoint between the two data sets in the given colour space. Then
    the colour metric is computed as dC^T * g * dC.
    
    Parameters
    ----------
    sp : Space
        The colour space in which to compute the linearised metric.
    dat1 : Data
        The colour data of the first data set.
    dat2 : Data
        The colour data of the second data set.
    metric_tensor_function : function
        Function giving the metric tensors at given colour data points.
    
    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    d1 = dat1.get_linear(sp)
    d2 = dat2.get_linear(sp)
    midp = (d1 + d2) * .5
    diff = d1 - d2
    g = metric_tensor_function(data.Data(sp, midp))
    g = g.get(sp)
    m = np.zeros(np.shape(diff)[0])
    for i in range(np.shape(m)[0]):
        m[i] = np.sqrt(np.dot(diff[i].T, np.dot(g[i], diff[i])))
    return m

def dE_ab(dat1, dat2):
    """
    Compute the DEab metric.
    
    Since the metric is Euclidean, this can be done using the linearised function.

    Parameters
    ----------
    dat1 : Data
        The colour data of the first data set.
    dat2 : Data
        The colour data of the second data set.
    
    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return linear(space.cielab, dat1, dat2, tensor.dE_ab)

def dE_uv(dat1, dat2):
    """
    Compute the DEuv metric.
    
    Since the metric is Euclidean, this can be done using the linearised function.

    Parameters
    ----------
    dat1 : Data
        The colour data of the first data set.
    dat2 : Data
        The colour data of the second data set.
    
    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return linear(space.cieluv, dat1, dat2, tensor.dE_uv)

def dE_00(dat1, dat2, k_L=1, k_C=1, k_h=1):
    """
    Compute the CIEDE00 metric.
    
    Parameters
    ----------
    dat1 : Data
        The colour data of the first data set.
    dat2 : Data
        The colour data of the second data set.
    k_L : float
        Parameter of the CIEDE00 metric
    k_C : float
        Parameter of the CIEDE00 metric
    k_h : float
        Parameter of the CIEDE00 metric

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    lch1 = dat1.get_linear(space.ciede00lch)
    lch2 = dat2.get_linear(space.ciede00lch)
    avg_lch = .5 * (lch1 + lch2)
    d_lch = lch1 - lch2
    
    h_deg = np.rad2deg(avg_lch[:,2])
    h_deg[h_deg < 0] = h_deg + 360
    S_L = 1 + (0.015 * (avg_lch[:,0] - 50)**2) / np.sqrt(20 + (avg_lch[:,0] - 50)**2)
    S_C = 1 + 0.045 * avg_lch[:,1]
    T = 1 - 0.17 * np.cos(np.deg2rad(h_deg - 30)) + \
        .24 * np.cos(2*avg_lch[:,2]) + \
        .32 * np.cos(np.deg2rad(3 * h_deg + 6)) - \
        .2 * np.cos(np.deg2rad(4 * h_deg - 63))
    S_h = 1 + 0.015 * avg_lch[:,1] * T
    R_C = 2 * np.sqrt(avg_lch[:,1]**7 / (avg_lch[:,1]**7 + 25**7))
    d_theta = 30 * np.exp(-((h_deg - 275) / 25)**2)
    R_T = - R_C * np.sin(np.deg2rad(2 * d_theta))
    dH = 2 * np.sqrt(lch1[:,1] * lch2[:,1]) * np.sin(d_lch[:,2] / 2)
    return np.sqrt((d_lch[:,0] / (k_L * S_L))**2 + 
                   (d_lch[:,1] / (k_C * S_C))**2 +
                   (dH / (k_h * S_h))**2 +
                   R_T * d_lch[:,1] * dH / (k_C * S_C * k_h * S_h))

#==============================================================================
# Test module
#==============================================================================

def test():
    """
    Test module, print results.
    """
    d1 = data.build_d_regular(space.cielab,
                             np.linspace(0, 100, 10),
                             np.linspace(-100, 100, 21),
                             np.linspace(-100, 100, 21))
    d2 = data.Data(space.cielab,
                   d1.get(space.cielab) + 1)
    print "Metric errors (all should be < 1e-11):"
    print np.max(dE_ab(d1, d2) - np.sqrt(3))
    d1 = data.build_d_regular(space.cieluv,
                             np.linspace(0, 100, 10),
                             np.linspace(-100, 100, 21),
                             np.linspace(-100, 100, 21))
    d2 = data.Data(space.cieluv,
                   d1.get(space.cieluv) + 1)
    print np.max(dE_uv(d1, d2) - np.sqrt(3))
