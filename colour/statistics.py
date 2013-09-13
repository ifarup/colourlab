#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
statistics: Colour metric statistics, part of the colour package

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
import scipy.integrate
import scipy.optimize

#==============================================================================
# Statistics of metrics
#==============================================================================

def stress(diff1, diff2):
    """
    Compute the STRESS for the two sets of differences.
    
    The STRESS (standardised residual sum of squares) is returned as a
    percentage.
    
    Parameters
    ----------
    diff1 : ndarray
        1D array of colour differences.
    diff2 : ndarray
        1D array of colour differences.
    
    Returns
    -------
    stress : float
        Standard residual sum of squares.
    """
    F = (diff1**2).sum() / (diff1 * diff2).sum()
    stress = (100 *np.sqrt(((diff1 - F * diff2)**2).sum() / 
              ((F * diff2)**2).sum()))
    return stress

def _ellipse_union(th, ell1, ell2):
    """
    For the Pant R values. Integrand the union of two ellipses.
    """
    r1 = ell1[0] * ell1[1] / (np.sqrt(ell1[1]**2 * np.cos(th-ell1[2])**2 +
                              ell1[0]**2 * np.sin(th-ell1[2])**2))
    r2 = ell2[0] * ell2[1] / (np.sqrt(ell2[1]**2 * np.cos(th-ell2[2])**2 +
                              ell2[0]**2 * np.sin(th-ell2[2])**2))
    u = max(r1,r2)
    return .5 * u**2

def _ellipse_intersection(th, ell1, ell2):
    """
    For the Pant R values. Integrand for the intersection of two ellipses.
    """
    r1 = ell1[0] * ell1[1] / (np.sqrt(ell1[1]**2 * np.cos(th-ell1[2])**2 +
                              ell1[0]**2 * np.sin(th-ell1[2])**2))
    r2 = ell2[0] * ell2[1] / (np.sqrt(ell2[1]**2 * np.cos(th-ell2[2])**2 +
                              ell2[0]**2 * np.sin(th-ell2[2])**2))
    u = min(r1,r2)
    return .5 * u**2

def _pant_R_value(ell1, ell2):
    """
    Compute single R value for the two given ellipses.
    """    
    area_intersection = scipy.integrate.quad(_ellipse_intersection,
                                             0, 2 * np.pi, (ell1, ell2))    
    area_union = scipy.integrate.quad(_ellipse_union,
                                      0, 2 * np.pi, (ell1, ell2))
    return area_intersection[0] / area_union[0]

def _pant_R_values(ells1, ells2, scale=1):
    """
    Compute set of R values for the two given sets of ellipses.
    """
    ells1 = ells1.copy()
    ells1[:,0:2] = ells1[:,0:2] * scale
    N = np.shape(ells1)[0]
    r_values = np.zeros(N)
    for i in range(N):
        r_values[i] = _pant_R_value(ells1[i], ells2[i])
    return r_values

def _cost_function(scale, ells1, ells2):
    """
    Cost function for the optimisation of the scale for the R values.
    """
    r_values = _pant_R_values(ells1, ells2, scale)
    return 1 - r_values.mean()
    
def pant_R_values(space, tdata1, tdata2, optimise=True, plane=None):
    """
    Compute the list of R values for the given metric tensors in tdataN.
    
    The R values are computed in the given colour space. If optimise=True,
    the maximum overall R values are found by scaling one of the data sets.
    The ellipses are computed in the given plane. If plane=None, all three
    principal planes are used, and the resulting array of R values will
    be three times the length ot tdataN.
    
    Parameters
    ----------
    space : Space
        The colour space in which to compute the R values.
    tdata1 : TensorData
        The first set of colour metric tensors.
    tdata2 : TensorData
        The second set of colour metric tensors.
    optimise : bool
        Whether or not to optimise the scaling of the ellipse set.
    plane : slice
        The principal plan for the ellipsoid cross sections.
    """    
    if plane == None:
        ell1a = tdata1.get_ellipse_parameters(space, tdata1.plane_01)
        ell1b = tdata1.get_ellipse_parameters(space, tdata1.plane_12)
        ell1c = tdata1.get_ellipse_parameters(space, tdata1.plane_20)
        ell1 = np.concatenate((ell1a, ell1b, ell1c))
        ell2a = tdata2.get_ellipse_parameters(space, tdata1.plane_01)
        ell2b = tdata2.get_ellipse_parameters(space, tdata1.plane_12)
        ell2c = tdata2.get_ellipse_parameters(space, tdata1.plane_20)
        ell2 = np.concatenate((ell2a, ell2b, ell2c))
    else:
        ell1 = tdata1.get_ellipse_parameters(space, plane)
        ell2 = tdata2.get_ellipse_parameters(space, plane)
    if optimise:
        res = scipy.optimize.fmin(_cost_function, 1, (ell1, ell2))
        return _pant_R_values(ell1, ell2, res[0]), res[0]
    else:
        return _pant_R_values(ell1, ell2)        

#==============================================================================
# Test module
#==============================================================================

def test():
    """
    Test entire module, and print report.
    """
    import data
    import space
    import metric
    import tensor
    d1 = data.build_d_regular(space.cielab,
                             np.linspace(0, 100, 10),
                             np.linspace(-100, 100, 21),
                             np.linspace(-100, 100, 21))
    d2 = data.Data(space.cielab,
                   d1.get(space.cielab) + 1)
    diff = metric.dE_ab(d1, d2)
    print "Various tests (should be True):"
    print stress(diff, diff) == 0
    print stress(diff, diff + 1) < 1e-11
    d1 = data.build_d_regular(space.cielab,
                             np.linspace(0, 100, 3),
                             np.linspace(-100, 100, 3),
                             np.linspace(-100, 100, 3))
    d2 = data.Data(space.cielab,
                   d1.get(space.cielab) + 1)
    t1 = tensor.dE_ab(d1)
    t2 = data.TensorData(space.cielab,
                         t1.points,
                         t1.get(space.cielab) * 2)
    print "\nOptimising Pant R values (takes some time)..."
    R, scale = pant_R_values(space.cielab, t1, t2)
    print np.max(np.abs(1 - R)) < 1e-4
