#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tensor: Compute colour metric tensors.

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
from data import TensorData
from space import spaceCIELAB, spaceCIELUV

#==============================================================================
# Colour metric tensors
#==============================================================================

def tensor_Euclidean(space, data):
    """
    Compute the general Euclidean metric in the given colour space as TensorData.
    
    Parameters
    ----------
    space : Space
        The colour space in which the metric tensor is Euclidean.
    data : Data
        The colour points for which to compute the metric.
        
    Returns
    -------
    Euclidean : TensorData
        The metric tensors.
    """
    g = space.empty_matrix(data.linear_XYZ)
    for i in range(np.shape(g)[0]):
        g[i] = np.eye(3)
    return TensorData(space, data, g)

def tensor_DEab(data):
    """
    Compute the DEab metric as TensorData for the given data points.

    Parameters
    ----------
    data : Data
        The colour points for which to compute the metric.
        
    Returns
    -------
    DEab : TensorData
        The metric tensors.
    """
    return tensor_Euclidean(spaceCIELAB, data)

def tensor_DEuv(data):
    """
    Compute the DEuv metric as TensorData for the given data points.

    Parameters
    ----------
    data : Data
        The colour points for which to compute the metric.
        
    Returns
    -------
    DEuv : TensorData
        The metric tensors.
    """
    return tensor_Euclidean(spaceCIELUV, data)
    
def tensor_Poincare_disk(space, data):
    """
    Compute the general Poincare Disk metric in the given colour space as TensorData

    Parameters
    ----------
    data : Data
        The colour points for which to compute the metric.
        
    Returns
    -------
    Poincare : TensorData
        The metric tensors.
    """
    
    d = data.get_linear(space)
    g = space.empty_matrix(d)
    for i in range(np.shape(g)[0]):
        g[i, 0, 0] = 1
        g[i, 1, 1] = 4. / (1 - d[i, 1]**2 - d[i, 2]**2)**2
        g[i, 2, 2] = 4. / (1 - d[i, 1]**2 - d[i, 2]**2)**2
    return TensorData(space, data, g)

# TODO:
#
# Functions (returning TensorData):
#     tensor_DE00(data)
#     tensor_Stiles(data)
#     tensor_Helmholz(data)
#     tensor_Schrodinger(data)
#     tensor_Vos(data)
#     tensor_SVF(data)
#     tensor_CIECAM02
#     tensor_DIN99
#     +++

