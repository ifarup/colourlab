#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metric: Colour metric functions. Part of the colourlab package.

Copyright (C) 2013-2017 Ivar Farup

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
from . import data, space


# =============================================================================
# Auxiliary functions
# =============================================================================


def reshape_diff(diff, sh):
    """
    Reshape the computed metric differences to fit with original data.

    The purpose is for, e.g., the difference of two MxNx3 images to be
    a MxN scalar image etc.

    Parameters
    ----------
    diff : ndarray
        The computed differences
    sh : tuple
        The shape of the original data (not the diff)
    """
    l = len(sh)
    if l == 1:        # one-dimensional colour data (one colour point)
        return diff[0]
    elif l == 2:       # two-dimensional colour data (list of colours)
        return diff
    else:                       # three or more dimensions (images++)
        return np.reshape(diff, tuple(np.array(sh)[:-1]))


# =============================================================================
# Colour metric functions
# =============================================================================


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
    dat1 : Points
        The colour data of the first data set.
    dat2 : Points
        The colour data of the second data set.
    metric_tensor_function : function
        Function giving the metric tensors at given colour data points.

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    d1 = dat1.get_flattened(sp)
    d2 = dat2.get_flattened(sp)
    midp = (d1 + d2) * .5
    diff = d1 - d2
    g = metric_tensor_function(data.Points(sp, midp))
    g = g.get(sp)
    m = np.zeros(np.shape(diff)[0])
    for i in range(np.shape(m)[0]):
        m[i] = np.sqrt(np.dot(diff[i].T, np.dot(g[i], diff[i])))
    return reshape_diff(m, dat1.sh)


def euclidean(sp, dat1, dat2):
    """
    Compute the Euclidean metric between the two data sets in the given space.

    Parameters
    ----------
    sp : Space
        Colour space
    dat1 : Points
        Colour data set 1
    dat2 : Points
        Colour data set 2

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    d1 = dat1.get_flattened(sp)
    d2 = dat2.get_flattened(sp)
    diff = d1 - d2
    m = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2 + diff[:, 2]**2)
    return reshape_diff(m, dat1.sh)


def poincare_disk(sp, dat1, dat2):
    """
    Compute the Poincare Disk metric betwen the two data sets.

    Compted in the given space. Assumes that the space is some form of
    a Poincare Disk space, such that the radius of curvature is given
    by sp.R. The first coordinate is treated as Euclidean.

    Parameters
    ----------
    sp : Space
        Colour space (should be of Poincare Disk type)
    dat1: Points
        Colour data set 1
    dat2: Points
        Colour data set 2

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    d1 = dat1.get_flattened(sp)
    d2 = dat2.get_flattened(sp)
    diff = d1 - d2
    delta = 2 * ((diff[:, 1]**2 + diff[:, 2]**2) /
                 ((1 - d1[:, 1]**2 - d1[:, 2]**2) *
                  (1 - d2[:, 1]**2 - d2[:, 2]**2)))
    duv = sp.R * np.arccosh(1 + delta)
    d = np.sqrt(diff[:, 0]**2 + duv**2)
    return reshape_diff(d, dat1.sh)


def dE_ab(dat1, dat2):
    """
    Compute the DEab metric.

    Parameters
    ----------
    dat1 : Points
        The colour data of the first data set.
    dat2 : Points
        The colour data of the second data set.

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return euclidean(space.cielab, dat1, dat2)


def dE_uv(dat1, dat2):
    """
    Compute the DEuv metric.

    Parameters
    ----------
    dat1 : Points
        The colour data of the first data set.
    dat2 : Points
        The colour data of the second data set.

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return euclidean(space.cieluv, dat1, dat2)


def dE_E(dat1, dat2):
    """
    Compute the DEE metric.

    Parameters
    ----------
    dat1 : Points
        The colour data of the first data set.
    dat2 : Points
        The colour data of the second data set.

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return euclidean(space.lgj_e, dat1, dat2)


def dE_DIN99(dat1, dat2):
    """
    Compute the DIN99 metric.

    Parameters
    ----------
    dat1 : Points
        The colour data of the first data set.
    dat2 : Points
        The colour data of the second data set.

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return euclidean(space.din99, dat1, dat2)


def dE_DIN99b(dat1, dat2):
    """
    Compute the DIN99b metric.

    Parameters
    ----------
    dat1 : Points
        The colour data of the first data set.
    dat2 : Points
        The colour data of the second data set.

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return euclidean(space.din99b, dat1, dat2)


def dE_DIN99c(dat1, dat2):
    """
    Compute the DIN99c metric.

    Parameters
    ----------
    dat1 : Points
        The colour data of the first data set.
    dat2 : Points
        The colour data of the second data set.

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return euclidean(space.din99c, dat1, dat2)


def dE_DIN99d(dat1, dat2):
    """
    Compute the DIN99d metric.

    Parameters
    ----------
    dat1 : Points
        The colour data of the first data set.
    dat2 : Points
        The colour data of the second data set.

    Returns
    -------
    distance : ndarray
        Array of the difference or distances between the two data sets.
    """
    return euclidean(space.din99d, dat1, dat2)


def dE_00(dat1, dat2, k_L=1, k_C=1, k_h=1):
    """
    Compute the CIEDE00 metric.

    Parameters
    ----------
    dat1 : Points
        The colour data of the first data set.
    dat2 : Points
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
    lch1 = dat1.get_flattened(space.ciede00lch)
    lch2 = dat2.get_flattened(space.ciede00lch)
    avg_lch = .5 * (lch1 + lch2)
    d_lch = lch1 - lch2

    h_deg = np.rad2deg(avg_lch[:, 2])
    h_deg[h_deg < 0] = h_deg[h_deg < 0] + 360
    S_L = 1 + ((0.015 * (avg_lch[:, 0] - 50)**2) /
               np.sqrt(20 + (avg_lch[:, 0] - 50)**2))
    S_C = 1 + 0.045 * avg_lch[:, 1]
    T = 1 - 0.17 * np.cos(np.deg2rad(h_deg - 30)) + \
        .24 * np.cos(2*avg_lch[:, 2]) + \
        .32 * np.cos(np.deg2rad(3 * h_deg + 6)) - \
        .2 * np.cos(np.deg2rad(4 * h_deg - 63))
    S_h = 1 + 0.015 * avg_lch[:, 1] * T
    R_C = 2 * np.sqrt(avg_lch[:, 1]**7 / (avg_lch[:, 1]**7 + 25**7))
    d_theta = 30 * np.exp(-((h_deg - 275) / 25)**2)
    R_T = - R_C * np.sin(np.deg2rad(2 * d_theta))
    dH = 2 * np.sqrt(lch1[:, 1] * lch2[:, 1]) * np.sin(d_lch[:, 2] / 2)
    d = np.sqrt((d_lch[:, 0] / (k_L * S_L))**2 +
                (d_lch[:, 1] / (k_C * S_C))**2 +
                (dH / (k_h * S_h))**2 +
                R_T * d_lch[:, 1] * dH / (k_C * S_C * k_h * S_h))
    return reshape_diff(d, dat1.sh)
