#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_core: Colour image core operations, part of the colourlab package

For image processing operations that are applied directly to ndarrays. To be
used by the colour-space-specific methods in colour.image.Image.

Copyright (C) 2017-2021 Ivar Farup

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
import sys
from scipy.signal import correlate2d

try:                            # Hack to use numba only when installed
    from numba import jit       # (mainly to avoid trouble with Travis)
except ImportError:
    def jit(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

if ('sphinx' in sys.modules) or ('coverage' in sys.modules): # Hack to make and coverage avoid using @jit
    def jit(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

ANGLE_PRIME = 95273        # for LUTs, to be true to the original
RADIUS_PRIME = 29537       # for LUTs, to be true to the original

diff_forward = (
    np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]),
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
)

diff_backward = (
    np.array([[0, -1, 0], [ 0, 1, 0], [0, 0, 0]]),
    np.array([[0,  0, 0], [-1, 1, 0], [0, 0, 0]])
)

diff_centered = (
    .5 * np.array([[0, -1, 0], [ 0, 0, 0], [0, 1, 0]]),
    .5 * np.array([[0,  0, 0], [-1, 0, 1], [0, 0, 0]])
)

diff_sobel = (
    .125 * np.array([[-1, -2, -1], [ 0, 0, 0], [ 1, 2, 1]]),
    .125 * np.array([[-1,  0,  1], [-2, 0, 2], [-1, 0, 1]])
)

diff_feldman = (
    1 / 32 * np.array([[-3, -10, -3], [  0, 0,  0], [ 3, 10, 3]]),
    1 / 32 * np.array([[-3,   0,  3], [-10, 0, 10], [-3,  0, 3]])
)


def dpdl_perona_invsq(lambd, kappa=1e-2):
    """
    Perona and Malik's inverse square diffusion supression

    Name suggest dpsi/dlambda = 1 / (1 + lambd / kappa^2)

    Parameters
    ----------

    lambd : ndimage
        Eigenvalue of the structure tensor (single channelimage)
    kappa : float
        Paramter
    
    Returns = 1 / (1 + lambd / kappa**2)
    -------
    ndimage : dpsi / dlambda
    """
    return 1 / (1 + lambd / kappa**2)


def dpdl_perona_exp(lambd, kappa=1e-2):
    """
    Perona and Malik's exponential diffusion supression

    Name suggest dpsi/dlambda

    Parameters
    ----------

    lambd : ndimage
        Eigenvalue of the structure tensor (single channelimage)
    kappa : float
        Paramter
    
    Returns
    -------
    ndimage : dpsi / dlambda
    """
    return np.exp(-lambd / kappa**2)


def dpdl_tv(lambd, epsilon=1e-4):
    """
    Total variation diffusion supression

    Name suggest dpsi/dlambda = 1 / sqrt(lambd + epsilon)

    Parameters
    ----------

    lambd : ndimage
        Eigenvalue of the structure tensor (single channelimage)
    epsilon : float
        Regularisation paramter
    
    Returns
    -------
    ndimage : dpsi / dlambda
    """
    return 1 / np.sqrt(lambd + epsilon)

def diffusion_tensor_from_structure(s_tuple, dpsi_dlambda1=None, dpsi_dlambda2=None):
    """
    Compute the diffusion tensor coefficients from the structure
    tensor parameters

    Parameters
    ----------
    s_tuple : tuple
        The resulting tuple from a call to image.structure_tensor
    dpsi_dlambda1 : func
        The diffusion supression function for the first eigenvalue of the
        structure tensor. If None, use Perona and Malik's inverse square with
        default value from image_core.
    dpsi_dlambda2 : func
        Same for the second eigenvalue. If None use dpsi_dlambda1

    Returns
    -------
    d11 : ndarray
        The d11 component of the structure tensor of the image data.
    d12 : ndarray
        The d12 component of the structure tensor of the image data.
    d22 : ndarray
        The d22 component of the structure tensor of the image data.

    """
    _, _, _, lambda1, lambda2, e1x, e1y, e2x, e2y = s_tuple

    # Diffusion tensor

    if dpsi_dlambda1 == None:
        dpsi_dlambda1 = lambda x : dpdl_perona_invsq(x)
    
    if dpsi_dlambda2 == None:
        dpsi_dlambda2 = dpsi_dlambda1

    D1 = dpsi_dlambda1(lambda1)
    D2 = dpsi_dlambda2(lambda2)

    d11 = D1 * e1x ** 2 + D2 * e2x ** 2
    d12 = D1 * e1x * e1y + D2 * e2x * e2y
    d22 = D1 * e1y ** 2 + D2 * e2y ** 2

    return d11, d12, d22


def correlate(im, filter):
    """
    Correlation filter for multi-channel image with symmetric boundary conditions.

    Paramteters
    -----------
    im : ndarray
        M x N x C image
    filter : ndarray
        
    """
    im_c = np.zeros(im.shape)

    if len(im.shape) > 2:
        for c in range(im.shape[2]):
            im_c[..., c] = correlate2d(im[..., c], filter, 'same', 'symm')
    else:
        im_C = correlate2d(im, filter, 'same', 'symm')

    return im_c


def gradient(im, diff=diff_centered):
    """
    Compute the gradient of the image with the given filters.

    Parameters
    ----------
    im : ndarray
        M x N x C image
    diff : tuple
        Tuple with the two gradient filters
    
    Returns
    -------
    ndarray : d im / di
    ndarray : d im / dj
    """
    return correlate(im, diff[0]), correlate(im, diff[1])

def divergence(imi, imj, diff=diff_centered):
    """
    Compute the divergence of the image components with the given filters.

    Parameters
    ----------
    imi : ndarray
        M x N x C image
    imj : ndarray
        M x N x C image
    diff : tuple
        Tuple with the two gradient filters
    
    Returns
    -------
    ndarray : div(imi, imj)
    """
    return correlate(imi, diff[0]) + correlate(imj, diff[1])

@jit
def stress_channel(im, ns=3, nit=5, R=0):
    """
    Compute the stress image and range for a single channel image.

    Parameters
    ----------
    im : ndarray
        Greyscale image
    ns : int
        Number of sample points
    nit : int
        Number of iterations
    R : int
        Maximum radius. If R=0, the diagonal of the image is used.

    Returns
    -------
    stress_im : ndarray
        The result of stress
    range_im : ndarray
        The range image (see paper)
    """
    theta = np.random.rand(ANGLE_PRIME) * 2 * np.pi  # create LUTs
    lut_cos = np.cos(theta)
    lut_sin = np.sin(theta)
    radiuses = np.random.rand(RADIUS_PRIME)

    if R == 0:
        R = np.sqrt(im.shape[0]**2 + im.shape[1]**2)

    angle_no = 0                # indexes to LUTs
    radius_no = 0

    res_v = np.zeros(im.shape)
    res_r = np.zeros(im.shape)

    for i in range(im.shape[0]):  # iterate over image
        for j in range(im.shape[1]):

            for it in range(nit):  # iterations
                best_min = im[i, j]
                best_max = best_min

                for s in range(ns):  # samples
                    while True:      # "repeat"
                        angle_no = (angle_no + 1) % ANGLE_PRIME
                        radius_no = (radius_no + 1) % RADIUS_PRIME

                        u = i + int(R * radiuses[radius_no] *
                                    lut_cos[angle_no])
                        v = j + int(R * radiuses[radius_no] *
                                    lut_sin[angle_no])

                        if ((u < im.shape[0]) and
                            (u >= 0) and
                            (v < im.shape[1]) and
                            (v >= 0)):
                            break # "until"

                    if best_min > im[u, v]:
                        best_min = im[u, v]
                    if best_max < im[u, v]:
                        best_max = im[u, v] # end samples

                ran = best_max - best_min
                if ran == 0:
                    s = 0.5
                else:
                    s = (im[i, j] - best_min) / ran

                res_v[i, j] += s
                res_r[i, j] += ran # end iterations

    return res_v / nit, res_r / nit

def stress(im, ns=3, nit=5, R=0):
    """
    Compute the stress image and range for a multi-channel image.

    Parameters
    ----------
    im : ndarray
        Multi-channel image
    ns : int
        Number of sample points
    nit : int
        Number of iterations
    R : int
        Maximum radius. If R=0, the diagonal of the image is used.

    Returns
    -------
    stress_im : ndarray
        The result of stress
    range_im : ndarray
        The range image (see paper)
    """
    stress_im = im.copy()
    range_im = im.copy()
    for c in range(im.shape[2]):
        stress_im[..., c], range_im[..., c] = stress_channel(im[..., c], ns, nit, R)
    return stress_im, range_im