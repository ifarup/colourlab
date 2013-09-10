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
from data import Data
from space import spaceCIELAB, spaceCIELUV
from tensor import tensor_DEab, tensor_DEuv

#==============================================================================
# Colour metric functions
#==============================================================================

def metric_linear(space, data1, data2, metric_tensor_function):
    """
    Compute the linearised colour difference between the two data sets.
    
    The function metric_tensor_function is used to compute the metric tensor
    at the midpoint between the two data sets in the given colour space. Then
    the colour metric is computed as dC^T * g * dC.
    
    Parameters
    ----------
    space : Space
        The colour space in which to compute the linearised metric.
    data1 : Data
        The colour data of the first data set.
    data2 : Data
        The colour data of the second data set.
    metric_tensor_function : function
        Function giving the metric tensors at given colour data points.
    
    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    d1 = data1.get_linear(space)
    d2 = data2.get_linear(space)
    midp = (d1 + d2) * .5
    diff = d1 - d2
    g = metric_tensor_function(Data(space, midp))
    g = g.get(space)
    m = np.zeros(np.shape(diff)[0])
    for i in range(np.shape(m)[0]):
        m[i] = np.sqrt(np.dot(diff[i].T, np.dot(g[i], diff[i])))
    return m

def metric_DEab(data1, data2):
    """
    Compute the DEab metric.
    
    Since the metric is Euclidean, this can be done using the linearised function.

    Parameters
    ----------
    data1 : Data
        The colour data of the first data set.
    data2 : Data
        The colour data of the second data set.
    
    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return metric_linear(spaceCIELAB, data1, data2, tensor_DEab)

def metric_DEuv(data1, data2):
    """
    Compute the DEuv metric.
    
    Since the metric is Euclidean, this can be done using the linearised function.

    Parameters
    ----------
    data1 : Data
        The colour data of the first data set.
    data2 : Data
        The colour data of the second data set.
    
    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return metric_linear(spaceCIELUV, data1, data2, tensor_DEuv)
