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

        # Dimensions
        self.M = self.sh[0]
        self.N = self.sh[1]
        self.rip = np.r_[np.arange(1, self.M), self.M - 1]
        self.rim = np.r_[0, np.arange(0, self.M - 1)]
        self.rjp = np.r_[np.arange(1, self.N), self.N - 1]
        self.rjm = np.r_[0, np.arange(0, self.N - 1)]

    def diff(self, sp, dat):
        return data.Vectors(sp, self.get(sp) - dat.get(sp), self)

    def dip(self, sp):
        im = self.get(sp)
        return data.Vectors(sp, im[self.rip, ...] - im, self)

    def dim(self, sp):
        im = self.get(sp)
        return data.Vectors(sp, im - im[self.rim, ...], self)

    def dic(self, sp):
        im = self.get(sp)
        return data.Vectors(sp, .5 * (im[self.rip, ...] -
                                      im[self.rim, ...]), self)

    def djp(self, sp):
        im = self.get(sp)
        return data.Vectors(sp, im[:, self.rjp, :] - im, self)

    def djm(self, sp):
        im = self.get(sp)
        return data.Vectors(sp, im - im[:, self.rjm, :], self)

    def djc(self, sp):
        im = self.get(sp)
        return data.Vectors(sp, .5 * (im[:, self.rjp, :] -
                                      im[:, self.rjm, :]), self)

    def structure_tensor(self, sp, g=None, dir='p', grey=None):
        """
        Return the structure tensor of the underlying data image point set

        Assumes (for now) that the underlying data constitutes an
        image, i.e., is on the shape M x N x 3. Note that the returned
        eigenvectors are not oriented in a particular way (+- pi).

        Parameters
        ----------
        sp : Space
            The space in which to perform the computations
        g : Tensors
            The metric tensor to use. If not given, uses Euclidean in
            the current space
        dir : str
            The direction for the finite differences, p (plus), m
            (minus), c (centered)
        grey : ndarray Grey scale image for orientation of lightness
            gradient. If not present, use CIELAB L* channel

        Returns
        -------
        s11 : ndarray
            The s11 component of the structure tensor of the image data.
        s12 : ndarray
            The s12 component of the structure tensor of the image data.
        s22 : ndarray
            The s22 component of the structure tensor of the image data.
        lambda1 : ndarray
            The first eigenvalue of the structure tensor
        lambda2 : ndarray
            The second eigenvalue of the structure tensor
        e1i : ndarray
            The first component of the first eigenvector
        e1j : ndarray
            The second component of the first eigenvector
        e2i : ndarray
            The first component of the second eigenvector
        e2j : ndarray
            The second component of the second eigenvector
        """

        # Greyscale image for orientation of the final eigenvectors

        if grey is None:
            grey = self.get(space.cielab)[..., 0]

        # Gradient components

        if dir == 'p':
            di = self.dip(sp)
            dj = self.djp(sp)
            gi = grey[self.rip, :] - grey
            gj = grey[:, self.rjp] - grey
        elif dir == 'm':
            di = self.dim(sp)
            dj = self.djm(sp)
            gi = grey - grey[self.rim, :]
            gj = grey - grey[:, self.rjm]
        elif dir == 'c':
            di = self.dic(sp)
            dj = self.djc(sp)
            gi = .5 * (grey[self.rip, :] - grey[self.rim, :])
            gj = .5 * (grey[:, self.rjp] - grey[:, self.rjm])

        if g is None:
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

        e1i = np.cos(theta1)
        e1j = np.sin(theta1)
        e2i = np.cos(theta2)
        e2j = np.sin(theta2)

        # Rotate eigenvectors according to gradient of luminance image

        index = ((e1i * gi + e1j * gj) < 0)
        e1i[index] = -e1i[index]
        e1j[index] = -e1j[index]
        e2i[index] = -e2i[index]
        e2j[index] = -e2j[index]

        return s11, s12, s22, lambda1, lambda2, e1i, e1j, e2i, e2j

    def diffusion_tensor_from_structure(self, s_tuple, param=1e-4,
                                        type='invsq'):
        """
        Compute the diffusion tensor coefficients from the structure
        tensor parameters

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

    def diffusion_tensor(self, sp, param=1e-4, g=None, type='invsq',
                         dir='p', grey=None):
        """
        Compute the diffusion tensor coefficients for the image point set

        Assumes (for now) that the underlying data constitutes an
        image, i.e., is on the shape M x N x 3.

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
            The direction for the finite differences, p (plus), m
            (minus), c (centered)

        Returns
        -------
        d11 : ndarray
            The d11 component of the structure tensor of the image data.
        d12 : ndarray
            The d12 component of the structure tensor of the image data.
        d22 : ndarray
            The d22 component of the structure tensor of the image data.
        """
        return self.diffusion_tensor_from_structure(
            self.structure_tensor(sp, g, dir, grey), param, type)

    def c2g_diffusion(self, sp, nit, g=None, l_minus=True, scale=1,
                      dt=.25, aniso=True, param=1e-4, type='invsq'):
        """
        Convert colour image to greyscale using anisotropic diffusion

        Parameters
        ----------
        sp : Space
            Colour space in which to perform the numerical computations
        nit : int
            Number of iterations to compute
        g : Tensors
            The colour metric tensor. If not given, use Euclidean
        l_minus : bool
            Use lambda_minus in the computation of the gradient
        scale : float
            Distance from black to white according to metric
        dt : float
            Time step
        param : float
            The parameter for the nonlinear diffusion function
        type : str
            The type of diffusion function, invsq (inverse square) or
            exp (exponential), see Perona and Malik (1990)

        Returns
        -------
        grey_image : ndarray
            Greyscale image (range 0–1)
        """
        s_tuple = self.structure_tensor(sp, g)
        s11, s12, s22, lambda1, lambda2, e1x, e1y, e2x, e2y = s_tuple

        if l_minus:
            vi = e1x * np.sqrt(lambda1 - lambda2)
            vj = e1y * np.sqrt(lambda1 - lambda2)
        else:
            vi = e1x * np.sqrt(lambda1)
            vj = e1y * np.sqrt(lambda1)

        vi /= scale
        vj /= scale

        grey_image = self.get(space.cielab)[..., 0] / 100

        if aniso:               # anisotropic diffusion

            d11, d12, d22 = self.diffusion_tensor_from_structure(s_tuple, param, type)

            for i in range(nit):
                gi = grey_image[self.rip, :] - grey_image - vi
                gj = grey_image[:, self.rjp] - grey_image - vj
                
                ti = d11 * gi + d12 * gj
                tj = d12 * gi + d22 * gj

                tv = ti - ti[self.rim, :] + tj - tj[:, self.rjm]

                grey_image += dt * tv

                grey_image[grey_image < 0] = 0
                grey_image[grey_image > 1] = 1

        else:                   # isotropic diffusion

            for i in range(nit):
                gi = grey_image[self.rip, :] - grey_image - vi
                gj = grey_image[:, self.rjp] - grey_image - vj
                
                tv = gi - gi[self.rim, :] + gj - gj[:, self.rjm]

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
