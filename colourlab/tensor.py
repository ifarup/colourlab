#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensor: Compute colour metric tensors. Part of the colourlab package.

Copyright (C) 2013-2017 Ivar Farup

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from . import data, space


# =============================================================================
# Colour metric tensors
# =============================================================================

def construct_tensor(sp, tensor_ndata, dat):
    """
    Construct the Tensors object with correct dimensions

    The tensors_ndata has shape N x 3 x 3. Construct Tensors object
    with shape correspoinding to the data points in dat.

    Parameters
    ----------
    sp: space.Space
        The colour space of the tensor_ndata
    tensor_ndata: ndarray
        N x 3 x 3 ndarray of tensor data.
    dat: data.Points
        Colour data points
    """
    sh = np.hstack((np.array(dat.sh), 3))
    return data.Tensors(sp, np.reshape(tensor_ndata, sh), dat)

def euclidean(sp, dat):
    """
    Compute the general Euclidean metric in the given colour space.

    Returns Tensors.

    Parameters
    ----------
    sp : space.Space
        The colour space in which the metric tensor is Euclidean.
    dat : data.Points
        The colour points for which to compute the metric.

    Returns
    -------
    Euclidean : Tensors
        The metric tensors.
    """
    g = sp.empty_matrix(dat.flattened_XYZ)
    for i in range(np.shape(g)[0]):
        g[i] = np.eye(3)
    return construct_tensor(sp, g, dat)


def dE_ab(dat):
    """
    Compute the DEab metric as Tensors for the given data points.

    Parameters
    ----------
    dat : data.Points
        The colour points for which to compute the metric.

    Returns
    -------
    DEab : Tensors
        The metric tensors.
    """
    return euclidean(space.cielab, dat)


def dE_uv(dat):
    """
    Compute the DEuv metric as Tensors for the given data points.

    Parameters
    ----------
    dat : data.Points
        The colour points for which to compute the metric.

    Returns
    -------
    DEuv : Tensors
        The metric tensors.
    """
    return euclidean(space.cieluv, dat)


def dE_E(dat):
    """
    Compute the DEE metric as Tensors for the given data points.

    Parameters
    ----------
    dat : data.Points
        The colour points for which to compute the metric.

    Returns
    -------
    DEE : Tensors
        The metric tensors.
    """
    return euclidean(space.lgj_e, dat)


def dE_DIN99(dat):
    """
    Compute the DIN99 metric as Tensors for the given data points.

    Parameters
    ----------
    dat : data.Points
        The colour points for which to compute the metric.

    Returns
    -------
    DIN99 : Tensors
        The metric tensors.
    """
    return euclidean(space.din99, dat)


def dE_DIN99b(dat):
    """
    Compute the DIN99b metric as Tensors for the given data points.

    Parameters
    ----------
    dat : data.Points
        The colour points for which to compute the metric.

    Returns
    -------
    DIN99b : Tensors
        The metric tensors.
    """
    return euclidean(space.din99b, dat)


def dE_DIN99c(dat):
    """
    Compute the DIN99c metric as Tensors for the given data points.

    Parameters
    ----------
    dat : data.Points
        The colour points for which to compute the metric.

    Returns
    -------
    DIN99c : Tensors
        The metric tensors.
    """
    return euclidean(space.din99c, dat)


def dE_DIN99d(dat):
    """
    Compute the DIN99d metric as Tensors for the given data points.

    Parameters
    ----------
    dat : data.Points
        The colour points for which to compute the metric.

    Returns
    -------
    DIN99d : Tensors
        The metric tensors.
    """
    return euclidean(space.din99d, dat)


def dE_00(dat, k_L=1, k_C=1, k_h=1):
    """
    Compute the Riemannised CIEDE00 metric for the given data points.

    Returns Tensors. Be aware that the tensor is singluar at C = 0.

    Parameters
    ----------
    dat : data.Points
        The colour points for which to compute the metric.
    k_L : float
        Parameter of the CIEDE00 metric
    k_C : float
        Parameter of the CIEDE00 metric
    k_h : float
        Parameter of the CIEDE00 metric

    Returns
    -------
    DE00 : Tensors
        The metric tensors.
    """
    lch = dat.get_flattened(space.ciede00lch)
    L = lch[:, 0]
    C = lch[:, 1]
    h = lch[:, 2]
    h_deg = np.rad2deg(h)
    h_deg[h_deg < 0] = h_deg[h_deg < 0] + 360
    S_L = 1 + (0.015 * (L - 50)**2) / np.sqrt(20 + (L - 50)**2)
    S_C = 1 + 0.045 * C
    T = 1 - 0.17 * np.cos(np.deg2rad(h_deg - 30)) + \
        .24 * np.cos(2*h) + \
        .32 * np.cos(np.deg2rad(3 * h_deg + 6)) - \
        .2 * np.cos(np.deg2rad(4 * h_deg - 63))
    S_h = 1 + 0.015 * C * T
    R_C = 2 * np.sqrt(C**7 / (C**7 + 25**7))
    d_theta = 30 * np.exp(-((h_deg - 275) / 25)**2)
    R_T = - R_C * np.sin(np.deg2rad(2 * d_theta))
    g = space.ciede00lch.empty_matrix(lch)
    g[:, 0, 0] = (k_L * S_L)**(-2)
    g[:, 1, 1] = (k_C * S_C)**(-2)
    g[:, 2, 2] = C**2 * (k_h * S_h)**(-2)
    g[:, 1, 2] = .5 * C * R_T / (k_C * S_C * k_h * S_h)
    g[:, 2, 1] = .5 * C * R_T / (k_C * S_C * k_h * S_h)
    return construct_tensor(space.ciede00lch, g, dat)


def poincare_disk(sp, dat):
    """
    Compute the general Poincare Disk metric in the given colour space.

    Returns Tensors. Assumes that sp is a Poincare Disk of some
    kind, and thus has a radius of curvature as sp.R.

    Parameters
    ----------
    dat : data.Points
        The colour points for which to compute the metric.

    Returns
    -------
    Poincare : Tensors
        The metric tensors.
    """
    d = dat.get_flattened(sp)
    g = sp.empty_matrix(d)
    for i in range(np.shape(g)[0]):
        g[i, 0, 0] = 1
        g[i, 1, 1] = sp.R**2 * 4. / (1 - d[i, 1]**2 - d[i, 2]**2)**2
        g[i, 2, 2] = sp.R**2 * 4. / (1 - d[i, 1]**2 - d[i, 2]**2)**2
    return construct_tensor(sp, g, dat)

# TODO:
#
# Functions (returning Tensors):
#     stiles
#     helmholz
#     schrodinger
#     vos
#     SVF
#     CIECAM02
#     +++
