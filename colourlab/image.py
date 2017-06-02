#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image: Colour image, part of the colourlab package

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
from . import data, space, tensor, misc, image_core

class Image(data.Points):
    """
    Subclass of data.Points specifically for image shaped data.
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
        data.Points.__init__(self, sp, ndata)

    def diff(self, sp, dat):
        return data.Vectors(sp, self, self.get(sp) - dat.get(sp))

    def dip(self, sp):
        return data.Vectors(sp, self, misc.dip(self.get(sp)))

    def dim(self, sp):
        return data.Vectors(sp, self, misc.dim(self.get(sp)))

    def dic(self, sp):
        return data.Vectors(sp, self, misc.dic(self.get(sp)))

    def djp(self, sp):
        return data.Vectors(sp, self, misc.djp(self.get(sp)))

    def djm(self, sp):
        return data.Vectors(sp, self, misc.djm(self.get(sp)))

    def djc(self, sp):
        return data.Vectors(sp, self, misc.djc(self.get(sp)))

    def structure_tensor(self, sp, g=None, dir='p'):
        """
        Return the structure tensor of the underlying data image point set

        Assumes (for now) that the underlying data constitutes an image, i.e.,
        is on the shape M x N x 3.

        Parameters
        ----------
        sp : Space
            The space in which to perform the computations
        g : Tensors
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

        # Eigenvalues

        lambda1 = .5 * (s11 + s22 + np.sqrt((s11 - s22)**2 + 4 * s12**2))
        lambda2 = .5 * (s11 + s22 - np.sqrt((s11 - s22)**2 + 4 * s12**2))

        theta1 = .5 * np.arctan2(2 * s12, s11 - s22)
        theta2 = theta1 + np.pi / 2

        # Eigenvectors

        e1x = np.cos(theta1)
        e1y = np.sin(theta1)
        e2x = np.cos(theta2)
        e2y = np.sin(theta2)

        return s11, s12, s22, lambda1, lambda2, e1x, e1y, e2x, e2y

    def diffusion_tensor_from_structure(self, s_tuple, param=1e-4, type='invsq'):
        """
        Compute the diffusion tensor coefficients from the structure tensor parameters

        Parameters
        ----------
        s_tuple : tuple
            The resulting tuple from a call to self.structure_tensor
        param : float
            The parameter for the nonlinear diffusion function
        type : str
            The type of diffusion function, invsq (inverse square) or
            exp (exponential), see Perona and Malik (1990)

        Returns
        -------
        d11 : ndarray
            The d11 component of the structure tensor of the image data.
        d12 : ndarray
            The d12 component of the structure tensor of the image data.
        d22 : ndarray
            The d22 component of the structure tensor of the image data.

        """
        s11, s12, s22, lambda1, lambda2, e1x, e1y, e2x, e2y = s_tuple

        # Diffusion tensor

        if type == 'invsq':
            def D(lambdax):
                return 1 / (1 + param * lambdax ** 2)
        elif type == 'exp':
            def D(lambdax):
                return np.exp(-lambdax / param)

        D1 = D(lambda1)
        D2 = D(lambda2)

        d11 = D1 * e1x ** 2 + D2 * e2x ** 2
        d12 = D1 * e1x * e1y + D2 * e2x * e2y
        d22 = D1 * e1y ** 2 + D2 * e2y ** 2
        return d11, d12, d22

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
        g: Tensors
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
        return self.diffusion_tensor_from_structure(self.structure_tensor(sp, g, dir), param, type)

    def c2g_anisotropic(self, sp, nit, g=None, param=1e-4, type='invsq', scale=1, dt = .24):
        """
        Convert colour image to greyscale using anisotropic diffusion

        Parameters
        ----------
        sp : Space
            Colour space in which to perform the numerical computations
        g : Tensors
            The colour metric tensor. If not given, use Euclidean
        nit : int
            Number of iterations to compute
        param : float
            The parameter for the nonlinear diffusion function
        type : str
            The type of diffusion function, invsq (inverse square) or
            exp (exponential), see Perona and Malik (1990)
        scale : float
            The distance from black to white according to the applied metric

        Returns
        -------
        grey_image : ndarray
            Greyscale image (range 0–1)
        """
        s_tuple = self.structure_tensor(sp, g)
        s11, s12, s22, lambda1, lambda2, e1x, e1y, e2x, e2y = s_tuple
        d11, d12, d22 = self.diffusion_tensor_from_structure(s_tuple, param, type)

        vi = e1x * np.sqrt(lambda1 - lambda2) / scale
        vj = e1y * np.sqrt(lambda1 - lambda2) / scale

        grey_image = self.get(space.cielab)[..., 0] / 100

        for i in range(nit):
            gi = misc.dip(grey_image) - vi
            gj = misc.djp(grey_image) - vj

#            ti = d11 * gi + d12 * gj
#            tj = d12 * gi + d22 * gj

            tv = misc.dim(gi) + misc.djm(gj)

            grey_image += dt * tv
            grey_image[grey_image < 0] = 0
            grey_image[grey_image > 1] = 1

        return grey_image

    def stress(self, sp_in, sp_out=None, ns=3, nit=5, R=0):
        """
        Compute STRESS in the given colour space.

        Parameters
        ----------
        sp_in : space.Space
            The input colour space.
        sp_out : space.Space
            The colour space for the interpretation of the result. If
            None, it is taken to be the same as sp_in.
        ns : int
            Number of sample points.
        nit : int
            Number of iterations.
        R : int
            Radius in pixels. If R=0, the diagonal of the image is used.

        Returns
        -------
        image.Image
            The resulting STRESS image.
        """
        if sp_out is None:
            sp_out = sp_in
        im_in = self.get(sp_in)
        im_out = im_in.copy()
        for c in range(3):
            im_out[..., c], _ = image_core.stress(im_in[..., c], ns, nit, R)
        return Image(sp_out, im_out)
