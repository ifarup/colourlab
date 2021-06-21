#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image: Colour image, part of the colourlab package

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

    def gradient(self, sp, diff=image_core.diff_centered):
        """
        Compute the gradient components in the given colour space

        Parameters
        ----------
        sp : colour.space.Space
            The colour space
        diff : tuple
            Tuple with the two filters for the derivatives. Defaults to
            centered differences
        
        Returns
        -------
        image.data.Vector : The i component of the gradient as a colour vector
        image.data.Vector : The j component of the gradient as a colour vector
        """
        im = self.get(sp)
        gi, gj = image_core.gradient(im, diff)
        return data.Vectors(sp, gi, self), data.Vectors(sp, gj, self)

    def structure_tensor(self, sp, g_func=None, diff=image_core.diff_centered,
                         grey=None):
        """
        Return the structure tensor of the underlying data image point set

        Assumes (for now) that the underlying data constitutes an
        image, i.e., is on the shape M x N x 3. Note that the returned
        eigenvectors are not oriented in a particular way (+- pi).

        Parameters
        ----------
        sp : Space
            The space in which to perform the computations
        g_func : func
            Function computing the metric tensor to use. If not given, uses
            Euclidean in the given space
        diff : tuple
            Tuple with the two filters for the derivatives. Defaults to
            centered differences
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

        di, dj = self.gradient(sp, diff)
        gi, gj = image_core.gradient(grey, diff)

        # Metric tensor

        if g_func is None:
            g = tensor.euclidean(sp, self)
        else:
            g = g_func(self)
        
        # The structure tensor

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

    def diffusion_tensor(self, sp, dpsi_dlambda1=None, dpsi_dlambda2=None,
                         g_func=None, diff=image_core.diff_centered,
                         grey=None):
        """
        Compute the diffusion tensor coefficients for the image point set

        Assumes (for now) that the underlying data constitutes an
        image, i.e., is on the shape M x N x 3.

        Parameters
        ----------100100
        sp : Space
            The space in which to perform the computations
        dpsi_dlambda1 : func
            The diffusion supression function for the first eigenvalue of the
            structure tensor. If None, use Perona and Malik's inverse square
            with kappa = 1e-2.
        dpsi_dlambda2 : func
            Same for the second eigenvalue. If None use dpsi_dlambda1
        g_func : func
            Function computing the metric tensor to use. If not given, uses
            Euclidean in the given space

        Returns
        -------
        d11 : ndarray
            The d11 component of the structure tensor of the image data.
        d12 : ndarray
            The d12 component of the structure tensor of the image data.
        d22 : ndarray
            The d22 component of the structure tensor of the image data.
        """
        return image_core.diffusion_tensor_from_structure(
            self.structure_tensor(sp, g_func, diff, grey), dpsi_dlambda1,
                                                           dpsi_dlambda2)

    def c2g_diffusion(self, sp, nit, g_func=None, l_minus=True, scale=1,
                      dt=.25, dpsi_dlambda1=None, dpsi_dlambda2=None):
        """
        Convert colour image to greyscale using linear anisotropic diffusion

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
        dpsi_dlambda1 : func
            The diffusion supression function for the first eigenvalue of the
            structure tensor. If None, use Perona and Malik's inverse square
            with kappa = 1e-2.
        dpsi_dlambda2 : func
            Same for the second eigenvalue. If None use dpsi_dlambda1

        Returns
        -------
        grey_image : ndarray
            Greyscale image (range 0–1)
        """
        s_tuple = self.structure_tensor(sp, g_func)
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

        d11, d12, d22 = image_core.diffusion_tensor_from_structure(s_tuple,
                                               dpsi_dlambda1, dpsi_dlambda2)

        for i in range(nit):
            gi, gj = image_core.gradient(grey_image)
            gi -= vi
            gj -= vj
            
            ti = d11 * gi + d12 * gj
            tj = d12 * gi + d22 * gj

            tv = image_core.divergence(ti, tj)

            grey_image += dt * tv

            grey_image[grey_image < 0] = 0
            grey_image[grey_image > 1] = 1

        return grey_image

    def anisotropic_diffusion(self, sp, nit, dpsi_dlambda1=None,
                              dpsi_dlambda2=None, dt=.25, linear=True,
                              g_func=None, constraint=None):
        """
        Compute the anisotropic diffusion of the image.

        Parameters
        ----------
        sp : space.Space
            The colour space in which to perform the diffusion
        nit : int
            Number of iterations
        dpsi_dlambda1 : func
            The diffusion supression function for the first eigenvalue of the
            structure tensor. If None, use Perona and Malik's inverse square
            with kappa = 1e-2.
        dpsi_dlambda2 : func
            Same for the second eigenvalue. If None use dpsi_dlambda1
        linear : bool
            Linear anisotropic diffusion if true
        g_func : func
            Function computing the metric tensor to use. If not given, uses
            Euclidean in the given space
        constraint : func
            Function returning the constrained image (e.g., gamut or internal
            boundaries, e.g. CFAs or fixed domains for inpainting) in sp
        
        Returns
        -------
        space.Image : the resulting anisotropically diffused image
        """
        if constraint==None:
            constraint = lambda x : x
        im = self.get(sp).copy()
        d11, d12, d22 = self.diffusion_tensor(sp, dpsi_dlambda1,
                                              dpsi_dlambda2, g_func)
        d11 = np.stack((d11, d11, d11), 2)
        d12 = np.stack((d12, d12, d12), 2)
        d22 = np.stack((d22, d22, d22), 2)

        for i in range(nit):
            gi, gj = image_core.gradient(im, image_core.diff_forward)
            ti = d11 * gi + d12 * gj
            tj = d12 * gi + d22 * gj
            tv = image_core.divergence(ti, tj, image_core.diff_backward)

            im += dt * tv

            im = constraint(im)

            if not linear:
                d11, d12, d22 = self.diffusion_tensor(sp, dpsi_dlambda1,
                                                      dpsi_dlambda2, g_func)
                d11 = np.stack((d11, d11, d11), 2)
                d12 = np.stack((d12, d12, d12), 2)
                d22 = np.stack((d22, d22, d22), 2)

        return Image(sp, im)

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
        im = self.get(sp_in)
        stress_im, _ = image_core.stress(im, ns, nit, R)
        return Image(sp_out, stress_im)
