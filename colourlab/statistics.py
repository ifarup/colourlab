#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statistics: Colour metric statistics, part of the colourlab package

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

import numpy as np
import scipy.integrate
import scipy.optimize


# =============================================================================
# Statistics of metrics
# =============================================================================


def stress(diff1, diff2, weights=None, confidence=.95):
    """
    Compute the STRESS for the two sets of differences.

    The STRESS (standardised residual sum of squares) is returned as a
    percentage. If weights are given, WSTRESS is calculated.

    Parameters
    ----------
    diff1 : ndarray
        1D array of colour differences.
    diff2 : ndarray
        1D array of colour differences.
    weights : ndarray
        1D array of individual weights for the colour differences. If None,
        the standard STRESS is calculated, if given, WSTRESS is calculated.
    confidence : float
        The size of the confidence interval (e.g., .95 for a 95%
        confidence interval)

    Returns
    -------
    stress : float
        Standard residual sum of squares.
    interval : tuple
        The confidence interval for STRESS_a / STRESS_b
    """
    from scipy.stats import f
    if weights is None:
        weights = np.ones(np.shape(diff1))
    F = (diff1**2).sum() / (diff1 * diff2).sum()
    stress = np.sqrt((weights * (diff1 - F * diff2)**2).sum() /
                     (weights * (F * diff2)**2).sum())
    N = np.shape(weights)[0]
    return stress, f.interval(confidence, N - 1, N - 1)


def _ellipse_union(th, ell1, ell2):
    """
    For the Pant R values. Integrand the union of two ellipses.
    """
    r1 = ell1[0] * ell1[1] / (np.sqrt(ell1[1]**2 * np.cos(th-ell1[2])**2 +
                              ell1[0]**2 * np.sin(th-ell1[2])**2))
    r2 = ell2[0] * ell2[1] / (np.sqrt(ell2[1]**2 * np.cos(th-ell2[2])**2 +
                              ell2[0]**2 * np.sin(th-ell2[2])**2))
    u = max(r1, r2)
    return .5 * u**2


def _ellipse_intersection(th, ell1, ell2):
    """
    For the Pant R values. Integrand for the intersection of two ellipses.
    """
    r1 = ell1[0] * ell1[1] / (np.sqrt(ell1[1]**2 * np.cos(th-ell1[2])**2 +
                              ell1[0]**2 * np.sin(th-ell1[2])**2))
    r2 = ell2[0] * ell2[1] / (np.sqrt(ell2[1]**2 * np.cos(th-ell2[2])**2 +
                              ell2[0]**2 * np.sin(th-ell2[2])**2))
    u = min(r1, r2)
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
    ells1[:, 0:2] = ells1[:, 0:2] * scale
    N = np.shape(ells1)[0]
    r_values = np.zeros(N)
    for i in range(N):
        r_values[i] = _pant_R_value(ells1[i], ells2[i])
    return r_values


def _cost_function_pant(scale, ells1, ells2):
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
    tdata1 : Tensors
        The first set of colour metric tensors.
    tdata2 : Tensors
        The second set of colour metric tensors.
    optimise : bool
        Whether or not to optimise the scaling of the ellipse set.
    plane : slice
        The principal plan for the ellipsoid cross sections.

    Returns
    -------
    r_values : ndarray
        Pant R values
    """
    if plane is None:
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
        res = scipy.optimize.fmin(_cost_function_pant, 1, (ell1, ell2))
        return _pant_R_values(ell1, ell2, res[0]), res[0]
    else:
        return _pant_R_values(ell1, ell2), 0


def dataset_distance(data1, data2):
    """
    Euclidean distances between data in the two data sets.

    Parameters
    ----------
    data1 : ndarray
        The first dataset, Nx3.
    data2 : ndarray
        The second data set, Nx3.

    Returns
    -------
    diff : ndarray
        Array of Euclidean distances, N.
    """
    return np.sqrt(((data1 - data2)**2).sum(axis=1))


def _scale_rot_dataset(params, dataset):
    """
    Scale and rotate dataset for optimisation.

    Parameters
    ----------
    params : ndarray
        Optimising parameters, scale and angle.
    dataset : ndarray
        The data set to optimise.

    Returns
    -------
    new_set : ndarray
        The scaled and rotated dataset.
    """
    scale_mat = params[1] * np.eye(3)
    scale_mat[0, 0] = params[0]
    th = params[2]
    rot_mat = np.array([[1, 0, 0],
                        [0, np.cos(th), -np.sin(th)],
                        [0, np.sin(th),  np.cos(th)]])
    sys_mat = np.dot(scale_mat, rot_mat)
    return np.dot(sys_mat, dataset.T).T


def _cost_function_dataset(params, dataset, ground_truth):
    """
    Cost function for the optimisation of the scale for the R values.

    Parameters
    ----------
    params : ndarray
        Optimising parameters, L-scale, C-scale and angle.
    dataset : ndarray
        The data set to optimise.
    ground_truth : ndarray
        The ground truth dataset.

    Returns
    -------
    cost : float
        The cost for the current values of scale and angle.
    """
    return dataset_distance(
        _scale_rot_dataset(params, dataset), ground_truth).sum()


def minimal_dataset_distance(dataset, ground_truth):
    """
    Return the minimal dataset distance between a dataset and a ground truth.

    The dataset is assumed to be on the Lab form (as an Nx3 ndarray) and is
    changed by scaling and rotation about the L axis.

    Parameters
    ----------
    dataset : ndarray
        Nx3 ndarray with the colour data.
    ground_truth : ndarray
        Nx3 ndarray with the ground truth colour data.

    Returns
    -------
    diff : ndarray
        Array of minimal Euclidean distances.
    opt_data : ndarray
        The optimised data set by scaling and rotation.
    L-scale : float
        The optimal scale.
    C-scale : float
        The optimal scale.
    angle : float
        The optimal angle.
    """
    params = scipy.optimize.fmin(_cost_function_dataset, np.array([1, 1, 0]),
                                 (dataset, ground_truth))
    opt_data = _scale_rot_dataset(params, dataset)
    return dataset_distance(opt_data, ground_truth), \
        opt_data, params[0], params[1], params[2]
