#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_core: Colour image core operations, part of the colourlab package

Copyright (C) 2017 Ivar Farup

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

try:                            # Hack to use numba only when installed
    from numba import jit       # (mainly to avoid trouble with Travis)
except ImportError:
    def jit(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

if 'sphinx' in sys.modules:     # Hack to make sphinx avoid using @jit
    def jit(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

ANGLE_PRIME = 95273        # for LUTs, to be true to the original
RADIUS_PRIME = 29537       # for LUTs, to be true to the original


@jit
def stress(im, ns=3, nit=5, R=0):
    """
    Compute the stress image and range.

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
