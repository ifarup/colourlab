#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image: Colour image, part of the colour package

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
from . import data, misc, tensor

class Image(data.Data):
    """
    Subclass of colour.data.Data specifically for image shaped data.
    """

    def __init__(self, sp, ndata):
        """
        Construct new image instance and set colour space and data.

        Parameters
        ----------
        sp : Space
            The colour space for the given instanisiation data.
        ndata : ndarray
            The colour data in the given space.
        """
        data.Data.__init__(self, sp, ndata)

    def diff(self, sp, dat):
        return data.VectorData(sp, self, self.get(sp) - dat.get(sp))

    def dip(self, sp):
        return data.VectorData(sp, self, misc.dip(self.get(sp)))

    def dim(self, sp):
        return data.VectorData(sp, self, misc.dim(self.get(sp)))

    def dic(self, sp):
        return data.VectorData(sp, self, misc.dic(self.get(sp)))

    def djp(self, sp):
        return data.VectorData(sp, self, misc.djp(self.get(sp)))

    def djm(self, sp):
        return data.VectorData(sp, self, misc.djm(self.get(sp)))

    def djc(self, sp):
        return data.VectorData(sp, self, misc.djc(self.get(sp)))

    def structure_tensor(self, sp, g=None, dir='p'):
        """
        Return the structure tensor of the underlying data image point set

        Assumes (for now) that the underlying data constitutes an image, i.e.,
        is on the shape M x N x 3.

        Parameters
        ----------
        sp : Space
            The space in which to perform the computations
        g : TensorData
            The metric tensor to use. If not given, uses Euclidean in the current space
        dir : str
            The direction for the finite differences, p (plus), m (minus), c (centered)

        Returns
        -------
        s11 : ndarray
            The s11 component of the structure tensor of the image data.
        s12 : ndarray
            The s12 component of the structure tensor of the image data.
        s22 : ndarray
            The s22 component of the structure tensor of the image data.
        """
        if dir == 'p':
            di = self.dip(sp)
            dj = self.djp(sp)
        elif dir == 'm':
            di = self.dim(sp)
            dj = self.djm(sp)
        elif dir == 'c':
            di = self.dic(sp)
            dj = self.djc(sp)

        if g == None:
            g = tensor.euclidean(sp, self)

        s11 = g.inner(sp, di, di) # components of the structure tensor
        s12 = g.inner(sp, di, dj)
        s22 = g.inner(sp, dj, dj)

        return s11, s12, s22

    def diffusion_tensor(self, sp, param=1e-4, g=None, type='invsq', dir='p'):
        """
        Compute the diffusion tensor coefficients for the underying image point set

        Assumes (for now) that the underlying data constitutes an image, i.e.,
        is on the shape M x N x 3.

        Parameters
        ----------
        sp : Space
            The space in which to perform the computations
        param : float
            The parameter for the nonlinear diffusion function
        g: TensorData
            The colour metric tensor. If not given, use Euclidean
        type : str
            The type of diffusion function, invsq (inverse square) or
            exp (exponential), see Perona and Malik (1990)
        dir : str
            The direction for the finite differences, p (plus), m (minus), c (centered)

        Returns
        -------
        d11 : ndarray
            The d11 component of the structure tensor of the image data.
        d12 : ndarray
            The d12 component of the structure tensor of the image data.
        d22 : ndarray
            The d22 component of the structure tensor of the image data.
        """

        s11, s12, s22 = self.structure_tensor(sp, g, dir)

        # Eigenvalues

        lambda1 = .5 * (s11 + s22 + np.sqrt((s11 - s22)**2 + 4 * s12**2))
        lambda2 = .5 * (s11 + s22 - np.sqrt((s11 - s22)**2 + 4 * s12**2))

#        return lambda1, lambda2

        theta1 = .5 * np.arctan2(2 * s12, s11 - s22)
        theta2 = theta1 + np.pi / 2

        # Eigenvectors

        v1x = np.cos(theta1)
        v1y = np.sin(theta1)
        v2x = np.cos(theta2)
        v2y = np.sin(theta2)

        # Diffusion tensor

        if type == 'invsq':
            def D(lambdax):
                return 1 / (1 + param * lambdax**2)
        elif type == 'exp':
            def D(lambdax):
                return np.exp(-lambdax / param)

        D1 = D(lambda1)
        D2 = D(lambda2)

        d11 = D1 * v1x**2 + D2 * v2x**2
        d12 = D1 * v1x * v1y + D2 * v2x * v2y
        d22 = D1 * v1y**2 + D2 * v2y**2
        return d11, d12, d22
