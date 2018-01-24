#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
space: Colour spaces, part of the colourlab package

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
from . import misc


# =============================================================================
# Colour space classes
#
# Throughout the code, the name ndata is used for numerical data (numpy
# arrays), and data is used for objects of the type Points.
# =============================================================================


class Space(object):
    """
    Base class for the colour space classes.
    """
    # White points in XYZ
    white_A = np.array([1.0985, 1., 0.35585])
    white_B = np.array([.990720, 1., .852230])
    white_C = np.array([.980740, 1., .82320])
    white_D50 = np.array([.964220, 1., .825210])
    white_D55 = np.array([.956820, 1., .921490])
    white_D65 = np.array([.950470, 1., 1.088830])
    white_D75 = np.array([.949720, 1., 1.226380])
    white_E = np.array([1., 1., 1.])
    white_F2 = np.array([.991860, 1., .673930])
    white_F7 = np.array([.950410, 1., 1.087470])
    white_F11 = np.array([1.009620, 1., .643500])

    def empty_matrix(self, ndata):
        """
        Return list of emtpy (zero) matrixes suitable for jacobians etc.

        Parameters
        ----------
        ndata : ndarray
            List of colour data.

        Returns
        -------
        empty_matrix : ndarray
            List of empty matrices of dimensions corresponding to ndata.
        """
        return np.zeros((np.shape(ndata)[0], 3, 3))

    def jacobian_XYZ(self, data):
        """
        Return the Jacobian to XYZ, dx^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Points class) by inverting the inverse Jacobian.

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to XYZ.
        """
        return np.linalg.inv(self.inv_jacobian_XYZ(data))

    def inv_jacobian_XYZ(self, data):
        """
        Return the inverse Jacobian to XYZ, dXYZ^i/dx^j.

        The inverse Jacobian is calculated at the given data points
        (of the Points class) by inverting the Jacobian.

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians from XYZ.
        """
        return np.linalg.inv(self.jacobian_XYZ(data))

    def vectors_to_XYZ(self, points_data, vectors_ndata):
        """
        Convert metric data to the XYZ colour space.

        Parameters
        ----------
        points_data : Points
            The colour data points.
        vectors_ndata : ndarray
            Array of colour metric tensors in current colour space.

        Returns
        -------
        xyz_vectors : ndarray
            Array of colour vectors in XYZ.
        """
        jacobian = self.inv_jacobian_XYZ(points_data)
        return np.einsum('...ij,...j->...i', jacobian, vectors_ndata)

    def vectors_from_XYZ(self, points_data, vectors_ndata):
        """
        Convert metric data from the XYZ colour space.

        Parameters
        ----------
        points_data : Points
            The colour data points.
        vectors_ndata : ndarray
            Array of colour metric tensors in XYZ.

        Returns
        -------
        vectors : ndarray
            Array of colour vectors in the current colour space.
        """
        jacobian = self.jacobian_XYZ(points_data)
        return np.einsum('...ij,...j->...i', jacobian, vectors_ndata)

    def metrics_to_XYZ(self, points_data, metrics_ndata):
        """
        Convert metric data to the XYZ colour space.

        Parameters
        ----------
        points_data : Points
            The colour data points.
        metrics_ndata : ndarray
            Array of colour metric tensors in current colour space.

        Returns
        -------
        xyz_metrics : ndarray
            Array of colour metric tensors in XYZ.
        """
        jacobian = self.jacobian_XYZ(points_data)
        return np.einsum('...ij,...ik,...kl->...jl', jacobian,
                         metrics_ndata, jacobian)

    def metrics_from_XYZ(self, points_data, metrics_ndata):
        """
        Convert metric data from the XYZ colour space.

        Parameters
        ----------
        points_data : Points
            The colour data points.
        metrics_ndata : ndarray
            Array of colour metric tensors in XYZ.

        Returns
        -------
        metrics : ndarray
            Array of colour metric tensors in the current colour space.
        """
        jacobian = self.inv_jacobian_XYZ(points_data)
        return np.einsum('...ij,...ik,...kl->...jl', jacobian,
                         metrics_ndata, jacobian)


class XYZ(Space):
    """
    The XYZ colour space.

    Assumes that the CIE 1931 XYZ colour matching functions are
    used. The white point is D65. Serves a special role in the code in that
    it serves as a common reference point.
    """

    def to_XYZ(self, ndata):
        """
        Convert from current colour space to XYZ.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space.

        Returns
        -------
        xyz : ndarray
            Colour data in the XYZ colour space.
        """
        return ndata.copy()      # identity transform

    def from_XYZ(self, ndata):
        """
        Convert from XYZ to current colour space.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the XYZ colour space.

        Returns
        -------
        xyz : ndarray
            Colour data in the current colour space.
        """
        return ndata.copy()      # identity transform

    def jacobian_XYZ(self, data):
        """
        Return the Jacobian to XYZ, dx^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to XYZ.
        """
        jac = self.empty_matrix(data.flattened_XYZ)
        jac[:] = np.eye(3)
        return jac

    def inv_jacobian_XYZ(self, data):
        """
        Return the inverse Jacobian to XYZ, dXYZ^i/dx^j.

        The inverse Jacobian is calculated at the given data points
        (of the Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians from XYZ.
        """
        ijac = self.empty_matrix(data.flattened_XYZ)
        ijac[:] = np.eye(3)
        return ijac


class Transform(Space):
    """
    Base class for colour space transforms.

    Real transforms (children) must implement to_base, from_base and either
    jacobian_base or inv_jacobian_base.
    """

    def __init__(self, base):
        """
        Construct instance and set base space for transformation.

        Parameters
        ----------
        base : Space
            The base for the colour space transform.
        """
        self.base = base

    def to_XYZ(self, ndata):
        """
        Transform data to XYZ by using the transformation to the base.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        xyz : ndarray
            Colour data in the XYZ colour space
        """
        return self.base.to_XYZ(self.to_base(ndata))

    def from_XYZ(self, ndata):
        """
        Transform data from XYZ using the transformation to the base.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the XYZ colour space.

        Returns
        -------
        xyz : ndarray
            Colour data in the current colour space.
        """
        return self.from_base(self.base.from_XYZ(ndata))

    def jacobian_base(self, data):
        """
        Return the Jacobian to base, dx^i/dbase^j.

        The Jacobian is calculated at the given data points (of the
        Points class) by inverting the inverse Jacobian.

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        return np.linalg.inv(self.inv_jacobian_base(data))

    def inv_jacobian_base(self, data):
        """
        Return the inverse Jacobian to base, dbase^i/dx^j.

        The inverse Jacobian is calculated at the given data points
        (of the Points class) by inverting the Jacobian.

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians from the base colour space.
        """
        return np.linalg.inv(self.jacobian_base(data))

    def jacobian_XYZ(self, data):
        """
        Return the Jacobian to XYZ, dx^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Points class) using the jacobian to the base and the Jacobian
        of the base space.

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to XYZ.

        """
        dxdbase = self.jacobian_base(data)
        dbasedXYZ = self.base.jacobian_XYZ(data)
        return np.einsum('...ij,...jk->...ik', dxdbase, dbasedXYZ)

    def inv_jacobian_XYZ(self, data):
        """
        Return the inverse Jacobian to XYZ, dXYZ^i/dx^j.

        The Jacobian is calculated at the given data points (of the
        Points class) using the inverse jacobian to the base and the
        inverse Jacobian of the base space.

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians from XYZ.
        """
        dXYZdbase = self.base.inv_jacobian_XYZ(data)
        dbasedx = self.inv_jacobian_base(data)
        return np.einsum('...ij,...jk->...ik', dXYZdbase, dbasedx)


class TransformxyY(Transform):
    """
    The XYZ to xyY projective transform.
    """

    def __init__(self, base):
        """
        Construct instance.

        Parameters
        ----------
        base : Space
            Base colour space.
        """
        super(TransformxyY, self).__init__(base)

    def to_base(self, ndata):
        """
        Convert from xyY to XYZ.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        xyz = np.zeros(np.shape(ndata))
        xyz[:, 0] = ndata[:, 0]*ndata[:, 2] / ndata[:, 1]
        xyz[:, 1] = ndata[:, 2]
        xyz[:, 2] = (1 - ndata[:, 0] - ndata[:, 1]) * ndata[:, 2] / ndata[:, 1]
        return xyz

    def from_base(self, ndata):
        """
        Convert from XYZ to xyY.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space.

        Returns
        -------
        col : ndarray
            Colour data in the current colour space.
        """
        xyz = ndata
        xyY = np.zeros(np.shape(xyz))
        xyz_sum = np.sum(xyz, axis=1)
        xyY[:, 0] = xyz[:, 0] / xyz_sum  # x
        xyY[:, 1] = xyz[:, 1] / xyz_sum  # y
        xyY[:, 2] = xyz[:, 1]            # Y
        return xyY

    def jacobian_base(self, data):
        """
        Return the Jacobian to XYZ, dxyY^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        xyzdata = data.get_flattened(self.base)
        jac = self.empty_matrix(xyzdata)
        for i in range(np.shape(jac)[0]):
            jac[i, 0, 0] = (xyzdata[i, 1] + xyzdata[i, 2]) / \
                (xyzdata[i, 0] + xyzdata[i, 1] + xyzdata[i, 2]) ** 2
            jac[i, 0, 1] = -xyzdata[i, 0] / \
                (xyzdata[i, 0] + xyzdata[i, 1] + xyzdata[i, 2]) ** 2
            jac[i, 0, 2] = -xyzdata[i, 0] / \
                (xyzdata[i, 0] + xyzdata[i, 1] + xyzdata[i, 2]) ** 2
            jac[i, 1, 0] = -xyzdata[i, 1] / \
                (xyzdata[i, 0] + xyzdata[i, 1] + xyzdata[i, 2]) ** 2
            jac[i, 1, 1] = (xyzdata[i, 0] + xyzdata[i, 2]) / \
                (xyzdata[i, 0] + xyzdata[i, 1] + xyzdata[i, 2]) ** 2
            jac[i, 1, 2] = -xyzdata[i, 1] / \
                (xyzdata[i, 0] + xyzdata[i, 1] + xyzdata[i, 2]) ** 2
            jac[i, 2, 1] = 1
        return jac

    def inv_jacobian_base(self, data):
        """
        Return the Jacobian from XYZ, dXYZ^i/dxyY^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        xyYdata = data.get_flattened(self)
        jac = self.empty_matrix(xyYdata)
        for i in range(np.shape(jac)[0]):
            jac[i, 0, 0] = xyYdata[i, 2] / xyYdata[i, 1]
            jac[i, 0, 1] = - xyYdata[i, 0] * xyYdata[i, 2] / xyYdata[i, 1] ** 2
            jac[i, 0, 2] = xyYdata[i, 0] / xyYdata[i, 1]
            jac[i, 1, 2] = 1
            jac[i, 2, 0] = - xyYdata[i, 2] / xyYdata[i, 1]
            jac[i, 2, 1] = xyYdata[i, 2] * (xyYdata[i, 0] - 1) / \
                xyYdata[i, 1] ** 2
            jac[i, 2, 2] = (1 - xyYdata[i, 0] - xyYdata[i, 1]) / xyYdata[i, 1]
        return jac


class TransformCIELAB(Transform):
    """
    The XYZ to CIELAB colour space transform.

    The white point is a parameter in the transform.
    """
    kappa = 24389. / 27.        # standard: 903.3
    epsilon = 216. / 24389.     # standard: 0.008856

    def __init__(self, base, white_point=Space.white_D65):
        """
        Construct instance by setting base space and white point.

        Parameters
        ----------
        base : Space
            The base colour space.
        white_point : ndarray or Points
            The white point
        """
        super(TransformCIELAB, self).__init__(base)
        if not isinstance(white_point, np.ndarray):
            self.white_point = white_point.get(xyz)
        else:
            self.white_point = white_point

    def f(self, ndata):
        """
        Auxiliary function for the conversion.
        """
        fx = (self.kappa * ndata + 16.) / 116.
        fx[ndata > self.epsilon] = ndata[ndata > self.epsilon] ** (1. / 3)
        return fx

    def dfdx(self, ndata):
        """
        Auxiliary function for the Jacobian.

        Returns the derivative of the function f above. Works for arrays.
        """
        df = self.kappa / 116. * np.ones(np.shape(ndata))
        df[ndata > self.epsilon] = \
            (ndata[ndata > self.epsilon] ** (-2. / 3)) / 3
        return df

    def to_base(self, ndata):
        """
        Convert from CIELAB to XYZ (base).

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        ndata
        fy = (ndata[:, 0] + 16.) / 116.
        fx = ndata[:, 1] / 500. + fy
        fz = fy - ndata[:, 2] / 200.
        xr = fx ** 3
        xr[xr <= self.epsilon] = ((116 * fx[xr <= self.epsilon] - 16) /
                                  self.kappa)
        yr = fy ** 3
        yr[ndata[:, 0] <= self.kappa * self.epsilon] = \
            ndata[ndata[:, 0] <= self.kappa * self.epsilon, 0] / self.kappa
        zr = fz ** 3
        zr[zr <= self.epsilon] = ((116 * fz[zr <= self.epsilon] - 16) /
                                  self.kappa)
        xyz = np.zeros(np.shape(ndata))
        xyz[:, 0] = xr * self.white_point[0]
        xyz[:, 1] = yr * self.white_point[1]
        xyz[:, 2] = zr * self.white_point[2]
        return xyz

    def from_base(self, ndata):
        """
        Convert from XYZ (base) to CIELAB.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space.

        Returns
        -------
        col : ndarray
            Colour data in the current colour space.
        """
        lab = np.zeros(np.shape(ndata))
        fx = self.f(ndata[:, 0] / self.white_point[0])
        fy = self.f(ndata[:, 1] / self.white_point[1])
        fz = self.f(ndata[:, 2] / self.white_point[2])
        lab[:, 0] = 116. * fy - 16.
        lab[:, 1] = 500. * (fx - fy)
        lab[:, 2] = 200. * (fy - fz)
        return lab

    def jacobian_base(self, data):
        """
        Return the Jacobian to XYZ (base), dCIELAB^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        d = data.get_flattened(self.base)
        dr = d.copy()
        for i in range(3):
            dr[:, i] = dr[:, i] / self.white_point[i]
        df = self.dfdx(dr)
        jac = self.empty_matrix(d)
        jac[:, 0, 1] = 116 * df[:, 1] / self.white_point[1]   # dL/dY
        jac[:, 1, 0] = 500 * df[:, 0] / self.white_point[0]   # da/dX
        jac[:, 1, 1] = -500 * df[:, 1] / self.white_point[1]  # da/dY
        jac[:, 2, 1] = 200 * df[:, 1] / self.white_point[1]   # db/dY
        jac[:, 2, 2] = -200 * df[:, 2] / self.white_point[2]  # db/dZ
        return jac


class TransformCIELUV(Transform):
    """
    The XYZ to CIELUV colour space transform.

    The white point is a parameter in the transform.
    """
    kappa = 24389. / 27.        # standard: 903.3
    epsilon = 216. / 24389.     # standard: 0.008856

    def __init__(self, base, white_point=Space.white_D65):
        """
        Construct instance by setting base space and white point.

        Parameters
        ----------
        base : Space
            The base colour space.
        white_point : ndarray or Points
            The white point
        """
        super(TransformCIELUV, self).__init__(base)
        if not isinstance(white_point, np.ndarray):
            self.white_point = white_point.get(xyz)
        else:
            self.white_point = white_point

    def f(self, ndata):
        """
        Auxiliary function for the conversion.
        """
        fx = (self.kappa * ndata + 16.) / 116.
        fx[ndata > self.epsilon] = ndata[ndata > self.epsilon] ** (1. / 3)
        return fx

    def dfdx(self, ndata):
        """
        Auxiliary function for the Jacobian.

        Returns the derivative of the function f above. Works for arrays.
        """
        df = self.kappa / 116. * np.ones(np.shape(ndata))
        df[ndata > self.epsilon] = \
            (ndata[ndata > self.epsilon] ** (-2. / 3)) / 3
        return df

    def to_base(self, ndata):
        """
        Convert from CIELUV to XYZ (base).

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        luv = ndata
        fy = (luv[:, 0] + 16.) / 116.
        y = fy ** 3
        y[luv[:, 0] <= self.kappa * self.epsilon] = \
            luv[luv[:, 0] <= self.kappa * self.epsilon, 0] / self.kappa
        upr = 4 * self.white_point[0] / (self.white_point[0] +
                                         15*self.white_point[1] +
                                         3*self.white_point[2])
        vpr = 9 * self.white_point[1] / (self.white_point[0] +
                                         15*self.white_point[1] +
                                         3*self.white_point[2])
        a = (52*luv[:, 0] / (luv[:, 1] + 13*luv[:, 0]*upr) - 1) / 3
        b = -5 * y
        c = -1/3.
        d = y * (39*luv[:, 0] / (luv[:, 2] + 13*luv[:, 0]*vpr) - 5)
        x = (d - b) / (a - c)
        z = x * a + b
        # Combine into matrix
        xyz = np.zeros(np.shape(luv))
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = z
        return xyz

    def from_base(self, ndata):
        """
        Convert from XYZ (base) to CIELUV.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space.

        Returns
        -------
        col : ndarray
            Colour data in the current colour space.
        """
        d = ndata
        luv = np.zeros(np.shape(d))
        fy = self.f(d[:, 1] / self.white_point[1])
        up = 4 * d[:, 0] / (d[:, 0] + 15*d[:, 1] + 3*d[:, 2])
        upr = 4 * self.white_point[0] / (self.white_point[0] +
                                         15*self.white_point[1] +
                                         3*self.white_point[2])
        vp = 9 * d[:, 1] / (d[:, 0] + 15*d[:, 1] + 3*d[:, 2])
        vpr = 9 * self.white_point[1] / (self.white_point[0] +
                                         15*self.white_point[1] +
                                         3*self.white_point[2])
        luv[:, 0] = 116. * fy - 16.
        luv[:, 1] = 13 * luv[:, 0] * (up - upr)
        luv[:, 2] = 13 * luv[:, 0] * (vp - vpr)
        return luv

    def jacobian_base(self, data):
        """
        Return the Jacobian to XYZ (base), dCIELUV^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        xyz_ = data.get_flattened(xyz)
        luv = data.get_flattened(cieluv)
        df = self.dfdx(xyz_)
        jac = self.empty_matrix(xyz_)
        # dL/dY:
        jac[:, 0, 1] = 116 * df[:, 1] / self.white_point[1]
        # du/dX:
        jac[:, 1, 0] = 13 * luv[:, 0] * \
            (60 * xyz_[:, 1] + 12 * xyz_[:, 2]) / \
            (xyz_[:, 0] + 15 * xyz_[:, 1] + 3 * xyz_[:, 2]) ** 2
        # du/dY:
        jac[:, 1, 1] = 13 * luv[:, 0] * \
            -60 * xyz_[:, 0] / \
            (xyz_[:, 0] + 15 * xyz_[:, 1] + 3 * xyz_[:, 2]) ** 2 + \
            13 * jac[:, 0, 1] * (
                4 * xyz_[:, 0] / (xyz_[:, 0] + 15 * xyz_[:, 1] +
                                  3 * xyz_[:, 2]) -
                4 * self.white_point[0] /
                (self.white_point[0] + 15 * self.white_point[1] +
                 3 * self.white_point[2]))
        # du/dZ:
        jac[:, 1, 2] = 13 * luv[:, 0] * \
            -12 * xyz_[:, 0] / \
            (xyz_[:, 0] + 15 * xyz_[:, 1] + 3 * xyz_[:, 2]) ** 2
        # dv/dX:
        jac[:, 2, 0] = 13 * luv[:, 0] * \
            -9 * xyz_[:, 1] / \
            (xyz_[:, 0] + 15 * xyz_[:, 1] + 3 * xyz_[:, 2]) ** 2
        # dv/dY:
        jac[:, 2, 1] = 13 * luv[:, 0] * \
            (9 * xyz_[:, 0] + 27 * xyz_[:, 2]) / \
            (xyz_[:, 0] + 15 * xyz_[:, 1] + 3 * xyz_[:, 2]) ** 2 + \
            13 * jac[:, 0, 1] * (
                9 * xyz_[:, 1] / (xyz_[:, 0] + 15 * xyz_[:, 1] +
                                  3 * xyz_[:, 2]) - 9 * self.white_point[1] /
                (self.white_point[0] + 15 * self.white_point[1] +
                 3 * self.white_point[2]))
        # dv/dZ:
        jac[:, 2, 2] = 13 * luv[:, 0] * \
            -27 * xyz_[:, 1] / \
            (xyz_[:, 0] + 15 * xyz_[:, 1] + 3 * xyz_[:, 2]) ** 2
        return jac


class TransformCIEDE00(Transform):
    """
    The CIELAB to CIEDE00 L'a'b' colour space transform.
    """

    def __init__(self, base):
        """
        Construct instance by setting base space.

        Parameters
        ----------
        base : Space
            The base colour space.
        """
        super(TransformCIEDE00, self).__init__(base)

    def to_base(self, ndata):
        """
        Convert from CIEDE00 to CIELAB (base).

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        raise RuntimeError('No conversion of CIEDE00 Lab to CIELAB implemented (yet).')

    def from_base(self, ndata):
        """
        Convert from CIELAB (base) to CIEDE00.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space.

        Returns
        -------
        labp : ndarray
            Colour data in the CIEDE00 L'a'b' colour space.
        """
        lab = ndata
        labp = lab.copy()
        Cab = np.sqrt(lab[:, 1]**2 + lab[:, 2]**2)
        G = .5 * (1 - np.sqrt(Cab**7 / (Cab**7 + 25**7)))
        labp[:, 1] = lab[:, 1] * (1 + G)
        return labp

    def jacobian_base(self, data):
        """
        Return the Jacobian to CIELAB (base), dCIEDE00^i/dCIELAB^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        lab = data.get_flattened(cielab)
        lch = data.get_flattened(cielch)
        a = lab[:, 1]
        b = lab[:, 2]
        C = lch[:, 1]
        G = .5 * (1 - np.sqrt(C**7 / (C**7 + 25**7)))
        jac = self.empty_matrix(lab)
        jac[:, 0, 0] = 1        # dLp/dL
        jac[:, 2, 2] = 1        # dbp/db
        # jac[:, 1, 1] = 1 + G - misc.safe_div(a**2, C) * \
        #     (7 * 25**7 * C**(5/2.) /
        #      (4 * (C**7 + 25**7)**(3/2.)))  # dap/da
        jac[:, 1, 1] = 1 + G - misc.safe_div(a**2, C) * \
            (7 * 25**7 * C**(5/2.) /
             (8 * (C**7 + 25**7)**(3/2.)))  # dap/da
        jac[C == 0, 1, 1] = 1
        # jac[:, 1, 2] = - a * misc.safe_div(b, C) * \
        #     (7 * 25**7 * C**(5/2.) / (4 * (C**7 + 25**7)**(3/2.)))
        jac[:, 1, 2] = - a * misc.safe_div(b, C) * \
            (7 * 25**7 * C**(5/2.) / (8 * (C**7 + 25**7)**(3/2.)))
        jac[C == 0, 1, 2] = 0
        return jac


class TransformSRGB(Transform):
    """
    Transform linear RGB with sRGB primaries to sRGB.
    """

    def __init__(self, base):
        """
        Construct sRGB space instance, setting the base (linear RGB).

        Parameters
        ----------
        base : Space
            The base colour space.
        """
        super(TransformSRGB, self).__init__(base)

    def to_base(self, ndata):
        """
        Convert from sRGB to linear RGB. Performs gamut clipping if necessary.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the sRGB colour space

        Returns
        -------
        col : ndarray
            Colour data in the linear RGB colour space
        """
        nd = ndata.copy()
        nd[nd < 0] = 0
        nd[nd > 1] = 1
        rgb = ((nd + 0.055) / 1.055)**2.4
        rgb[nd <= 0.04045] = nd[nd <= 0.04045] / 12.92
        return rgb

    def jacobian_base(self, data):
        """
        Return the Jacobian to linear RGB (base), dsRGB^i/dRGB^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        rgb = data.get_flattened(self.base)
        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]
        jac = self.empty_matrix(rgb)
        jac[:, 0, 0] = 1.055 / 2.4 * r**(1 / 2.4 - 1)
        jac[r < 0.0031308, 0, 0] = 12.92
        jac[:, 1, 1] = 1.055 / 2.4 * g**(1 / 2.4 - 1)
        jac[g < 0.0031308, 1, 1] = 12.92
        jac[:, 2, 2] = 1.055 / 2.4 * b**(1 / 2.4 - 1)
        jac[b < 0.0031308, 2, 2] = 12.92
        return jac

    def from_base(self, ndata):
        """
        Convert from linear RGB to sRGB. Performs gamut clipping if necessary.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the linear colour space

        Returns
        -------
        col : ndarray
            Colour data in the sRGB colour space
        """
        nd = ndata.copy()
        nd[nd < 0] = 0
        nd[nd > 1] = 1
        srgb = 1.055 * nd**(1 / 2.4) - 0.055
        srgb[nd <= 0.0031308] = 12.92 * nd[nd <= 0.0031308]
        return srgb


class TransformLinear(Transform):
    """
    General linear transform, transformed = M * base
    """

    def __init__(self, base, M=np.eye(3)):
        """
        Construct instance, setting the matrix of the linear transfrom.

        Parameters
        ----------
        base : Space
            The base colour space.
        """
        super(TransformLinear, self).__init__(base)
        self.M = M.copy()
        self.M_inv = np.linalg.inv(M)

    def to_base(self, ndata):
        """
        Convert from linear to the base.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        xyz = np.zeros(np.shape(ndata))
        for i in range(np.shape(ndata)[0]):
            xyz[i] = np.dot(self.M_inv, ndata[i])
        return xyz

    def from_base(self, ndata):
        """
        Convert from the base to linear.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space.

        Returns
        -------
        col : ndarray
            Colour data in the current colour space.
        """
        xyz = ndata
        lins = np.zeros(np.shape(xyz))
        for i in range(np.shape(xyz)[0]):
            lins[i] = np.dot(self.M, xyz[i])
        return lins

    def jacobian_base(self, data):
        """
        Return the Jacobian to XYZ (base), dlinear^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        xyzdata = data.get_flattened(xyz)
        jac = self.empty_matrix(xyzdata)
        jac[:] = self.M
        return jac

    def inv_jacobian_base(self, data):
        """
        Return the Jacobian from XYZ (base), dXYZ^i/dlinear^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        xyzdata = data.get_flattened(xyz)
        jac = self.empty_matrix(xyzdata)
        jac[:] = self.M_inv
        return jac


class TransformGamma(Transform):
    """
    General gamma transform, transformed = base**gamma

    Uses absolute value and sign for negative base values:
    transformed = sign(base) * abs(base)**gamma
    """

    def __init__(self, base, gamma=1):
        """
        Construct instance, setting the gamma of the transfrom.

        Parameters
        ----------
        base : Space
            The base colour space.
        gamma : float
            The exponent for the gamma transformation from the base.
        """
        super(TransformGamma, self).__init__(base)
        self.gamma = float(gamma)
        self.gamma_inv = 1. / gamma

    def to_base(self, ndata):
        """
        Convert from gamma corrected to XYZ (base).

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        return np.sign(ndata) * np.abs(ndata)**self.gamma_inv

    def from_base(self, ndata):
        """
        Convert from XYZ to gamma corrected.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space.

        Returns
        -------
        col : ndarray
            Colour data in the current colour space.
        """
        return np.sign(ndata) * np.abs(ndata)**self.gamma

    def jacobian_base(self, data):
        """
        Return the Jacobian to XYZ (base), dgamma^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        basedata = data.get_flattened(self.base)
        jac = self.empty_matrix(basedata)
        for i in range(np.shape(basedata)[0]):
            jac[i, 0, 0] = self.gamma * \
                np.abs(basedata[i, 0])**(self.gamma - 1)
            jac[i, 1, 1] = self.gamma * \
                np.abs(basedata[i, 1])**(self.gamma - 1)
            jac[i, 2, 2] = self.gamma * \
                np.abs(basedata[i, 2])**(self.gamma - 1)
        return jac


class TransformPolar(Transform):
    """
    Transform form Cartesian to polar coordinates in the two last variables.

    For example CIELAB to CIELCH.
    """

    def __init__(self, base):
        """
        Construct instance, setting base space.

        Parameters
        ----------
        base : Space
            The base colour space.
        """
        super(TransformPolar, self).__init__(base)

    def to_base(self, ndata):
        """
        Convert from polar to Cartesian.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        Lab = np.zeros(np.shape(ndata))
        Lab[:, 0] = ndata[:, 0]
        C = ndata[:, 1]
        h = ndata[:, 2]
        Lab[:, 1] = C * np.cos(h)
        Lab[:, 2] = C * np.sin(h)
        return Lab

    def from_base(self, ndata):
        """
        Convert from Cartesian (base) to polar.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space.

        Returns
        -------
        col : ndarray
            Colour data in the current colour space.
        """
        LCh = np.zeros(np.shape(ndata))
        LCh[:, 0] = ndata[:, 0]
        x = ndata[:, 1]
        y = ndata[:, 2]
        LCh[:, 1] = np.sqrt(x**2 + y**2)
        LCh[:, 2] = np.arctan2(y, x)
        return LCh

    def inv_jacobian_base(self, data):
        """
        Return the Jacobian from CIELAB (base), dCIELAB^i/dCIELCH^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        LCh = data.get_flattened(self)
        C = LCh[:, 1]
        h = LCh[:, 2]
        jac = self.empty_matrix(LCh)
        for i in range(np.shape(jac)[0]):
            jac[i, 0, 0] = 1                     # dL/dL
            jac[i, 1, 1] = np.cos(h[i])          # da/dC
            jac[i, 1, 2] = -C[i] * np.sin(h[i])  # da/dh
            jac[i, 2, 1] = np.sin(h[i])          # db/dC
            jac[i, 2, 2] = C[i] * np.cos(h[i])   # db/dh
            if C[i] == 0:
                jac[i, 2, 2] = 1
                jac[i, 1, 1] = 1
        return jac


class TransformCartesian(Transform):
    """
    Transform form polar to Cartesian coordinates in the two last variables.

    For example CIELCH to CIELAB.
    """

    def __init__(self, base):
        """
        Construct instance, setting base space.

        Parameters
        ----------
        base : Space
            The base colour space.
        """
        super(TransformCartesian, self).__init__(base)

    def from_base(self, ndata):
        """
        Convert from polar to Cartesian.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space.

        Returns
        -------
        col : ndarray
            Colour data in the current colour space.
        """
        Lab = np.zeros(np.shape(ndata))
        Lab[:, 0] = ndata[:, 0]
        C = ndata[:, 1]
        h = ndata[:, 2]
        Lab[:, 1] = C * np.cos(h)
        Lab[:, 2] = C * np.sin(h)
        return Lab

    def to_base(self, ndata):
        """
        Convert from Cartesian (base) to polar.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        LCh = np.zeros(np.shape(ndata))
        LCh[:, 0] = ndata[:, 0]
        x = ndata[:, 1]
        y = ndata[:, 2]
        LCh[:, 1] = np.sqrt(x**2 + y**2)
        LCh[:, 2] = np.arctan2(y, x)
        return LCh

    def jacobian_base(self, data):
        """
        Return the Jacobian from CIELCh (base), dCIELAB^i/dCIELCH^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        LCh = data.get_flattened(self.base)
        C = LCh[:, 1]
        h = LCh[:, 2]
        jac = self.empty_matrix(LCh)
        for i in range(np.shape(jac)[0]):
            jac[i, 0, 0] = 1                     # dL/dL
            jac[i, 1, 1] = np.cos(h[i])          # da/dC
            jac[i, 1, 2] = -C[i] * np.sin(h[i])  # da/dh
            jac[i, 2, 1] = np.sin(h[i])          # db/dC
            jac[i, 2, 2] = C[i] * np.cos(h[i])   # db/dh
        return jac


class TransformLGJOSA(Transform):
    """
    Transform from XYZ type coordinates to L_osa G J.
    """
    def __init__(self, base):
        """
        Construct instance, setting base space.

        Parameters
        ----------
        base : Space
            The base colour space.
        """
        super(TransformLGJOSA, self).__init__(base)
        self.space_ABC = TransformLinear(self.base,
                                         np.array([[0.6597, 0.4492, -0.1089],
                                                   [-0.3053, 1.2126, 0.0927],
                                                   [-0.0374, 0.4795, 0.5579]]))
        self.space_xyY = TransformxyY(self.base)

    def err_func(self, xyz, lgj):
        clgj = self.from_base(np.reshape(xyz, (1, 3)))
        diff = clgj - np.reshape(lgj, (1, 3))
        n = np.linalg.norm(diff)
        return n

    def to_base(self, ndata):
        """
        Convert from LGJOSA to XYZ (base).

        Implemented as numerical inversion of the from_base method,
        since the functions unfortunately are not analytically
        invertible.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        import scipy.optimize
        xyz = .5 * np.ones(np.shape(ndata))
        for i in range(np.shape(xyz)[0]):
            xyz_guess = xyz[i].copy()
            lgj = ndata[i].copy()
            xyz[i] = scipy.optimize.fmin(self.err_func, xyz_guess, (lgj,))
        return xyz

    def from_base(self, ndata):
        """
        Transform from base to LGJ OSA.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space (XYZ).

        Returns
        -------
        col : ndarray
            Colour data in the LGJOSA colour space.
        """
        abc = self.space_ABC.from_base(ndata)
        A = abc[:, 0]
        B = abc[:, 1]
        C = abc[:, 2]
        xyY = self.space_xyY.from_base(ndata)
        x = xyY[:, 0]
        y = xyY[:, 1]
        Y = xyY[:, 2]
        Y_0 = 100 * Y * (4.4934 * x**2 + 4.3034 * y**2 - 4.2760 * x * y -
                         1.3744 * x - 2.5643 * y + 1.8103)
        L_osa = (5.9 * ((Y_0**(1/3.) - (2/3.)) +
                        0.0042 * np.sign(Y_0 - 30) *
                        np.abs(Y_0 - 30)**(1/3.)) - 14.4) / np.sqrt(2)
        G = -2 * (0.764 * L_osa + 9.2521) * (
            0.9482 * (np.log(A) - np.log(0.9366 * B)) -
            0.3175 * (np.log(B) - np.log(0.9807 * C)))
        J = 2 * (0.5735 * L_osa + 7.0892) * (
            0.1792 * (np.log(A) - np.log(0.9366 * B)) +
            0.9237 * (np.log(B) - np.log(0.9807 * C)))
        col = np.zeros(np.shape(ndata))
        col[:, 0] = L_osa
        col[:, 1] = G
        col[:, 2] = J
        return col

    def jacobian_base(self, data):
        """
        Return the Jacobian from XYZ (base), dLGJOSA^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Points class). Like the colour space, a terrible mess...

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        ABC = data.get_flattened(self.space_ABC)
        xyY = data.get_flattened(self.space_xyY)
        x = xyY[:, 0]
        y = xyY[:, 1]
        Y = xyY[:, 2]
        A = ABC[:, 0]
        B = ABC[:, 1]
        C = ABC[:, 2]
        dxyY_dXYZ = self.space_xyY.jacobian_base(data)
        dx_dX = dxyY_dXYZ[:, 0, 0]
        dx_dY = dxyY_dXYZ[:, 0, 1]
        dx_dZ = dxyY_dXYZ[:, 0, 2]
        dy_dX = dxyY_dXYZ[:, 1, 0]
        dy_dY = dxyY_dXYZ[:, 1, 1]
        dy_dZ = dxyY_dXYZ[:, 1, 2]
        dY_dX = dxyY_dXYZ[:, 2, 0]
        dY_dY = dxyY_dXYZ[:, 2, 1]
        dY_dZ = dxyY_dXYZ[:, 2, 2]
        dABC_dXYZ = self.space_ABC.jacobian_base(data)
        dA_dX = dABC_dXYZ[:, 0, 0]
        dA_dY = dABC_dXYZ[:, 0, 1]
        dA_dZ = dABC_dXYZ[:, 0, 2]
        dB_dX = dABC_dXYZ[:, 1, 0]
        dB_dY = dABC_dXYZ[:, 1, 1]
        dB_dZ = dABC_dXYZ[:, 1, 2]
        dC_dX = dABC_dXYZ[:, 2, 0]
        dC_dY = dABC_dXYZ[:, 2, 1]
        dC_dZ = dABC_dXYZ[:, 2, 2]
        Y_0 = 100 * Y * (4.4934 * x**2 + 4.3034 * y**2 - 4.2760 * x * y -
                         1.3744 * x - 2.5643 * y + 1.8103)
        L = (5.9 * ((Y_0**(1/3.) - (2/3.)) +
                    0.0042 * np.sign(Y_0 - 30) *
                    np.abs(Y_0 - 30)**(1/3.)) - 14.4) / np.sqrt(2)
        dL_dY0 = 5.9 * (Y_0**(-2./3) + 0.042 * np.sign(Y_0 - 30) *
                        np.abs(Y_0 - 30)**(-2./3) / 3) / np.sqrt(2)
        dY0_dx = 100 * Y * (4.4934 * 2 * x - 4.2760 * y - 1.3744)
        dY0_dy = 100 * Y * (4.3034 * 2 * y - 4.2760 * x - 2.5643)
        dY0_dY = 100 * (4.4934 * x**2 + 4.3034 * y**2 - 4.2760 * x * y -
                        1.3744 * x - 2.5643 * y + 1.8103)
        dL_dX = dL_dY0 * (dY0_dx * dx_dX + dY0_dy * dy_dX + dY0_dY * dY_dX)
        dL_dY = dL_dY0 * (dY0_dx * dx_dY + dY0_dy * dy_dY + dY0_dY * dY_dY)
        dL_dZ = dL_dY0 * (dY0_dx * dx_dZ + dY0_dy * dy_dZ + dY0_dY * dY_dZ)
        TG = 0.9482 * (np.log(A) - np.log(0.9366 * B)) - \
            0.3175 * (np.log(B) - np.log(0.9807 * C))
        TJ = 0.1792 * (np.log(A) - np.log(0.9366 * B)) + \
            0.9237 * (np.log(B) - np.log(0.9807 * C))
        SG = - 2 * (0.764 * L + 9.2521)
        SJ = 2 * (0.5735 * L + 7.0892)
        dG_dL = - 2 * 0.764 * TG
        dJ_dL = 2 * 0.57354 * TJ
        dG_dA = misc.safe_div(SG * 0.9482, A)
        dG_dB = misc.safe_div(SG * (-0.9482 - 0.3175), B)
        dG_dC = misc.safe_div(SG * 0.3175, C)
        dJ_dA = misc.safe_div(SJ * 0.1792, A)
        dJ_dB = misc.safe_div(SJ * (-0.1792 + 0.9837), B)
        dJ_dC = misc.safe_div(SJ * (-0.9837), C)
        dG_dX = dG_dL * dL_dX + dG_dA * dA_dX + dG_dB * dB_dX + dG_dC * dC_dX
        dG_dY = dG_dL * dL_dY + dG_dA * dA_dY + dG_dB * dB_dY + dG_dC * dC_dY
        dG_dZ = dG_dL * dL_dZ + dG_dA * dA_dZ + dG_dB * dB_dZ + dG_dC * dC_dZ
        dJ_dX = dJ_dL * dL_dX + dJ_dA * dA_dX + dJ_dB * dB_dX + dJ_dC * dC_dX
        dJ_dY = dJ_dL * dL_dY + dJ_dA * dA_dY + dJ_dB * dB_dY + dJ_dC * dC_dY
        dJ_dZ = dJ_dL * dL_dZ + dJ_dA * dA_dZ + dJ_dB * dB_dZ + dJ_dC * dC_dZ
        jac = self.empty_matrix(ABC)
        jac[:, 0, 0] = dL_dX
        jac[:, 0, 1] = dL_dY
        jac[:, 0, 2] = dL_dZ
        jac[:, 1, 0] = dG_dX
        jac[:, 1, 1] = dG_dY
        jac[:, 1, 2] = dG_dZ
        jac[:, 2, 0] = dJ_dX
        jac[:, 2, 1] = dJ_dY
        jac[:, 2, 2] = dJ_dZ
        return jac


class TransformLGJE(Transform):
    """
    Transform from LGJOSA type coordinates to L_E, G_E, J_E.
    """
    def __init__(self, base):
        """
        Construct instance, setting base space.

        Parameters
        ----------
        base : Space
            The base colour space.
        """
        super(TransformLGJE, self).__init__(base)
        self.aL = 2.890
        self.bL = 0.015
        self.ac = 1.256
        self.bc = 0.050

    def to_base(self, ndata):
        """
        Convert from LGJE to LGJOSA (base).

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        LE = ndata[:, 0]
        GE = ndata[:, 1]
        JE = ndata[:, 2]
        CE = np.sqrt(GE**2 + JE**2)
        L = self.aL * (np.exp(self.bL * LE) - 1) / (10 * self.bL)
        C = self.ac * (np.exp(self.bc * CE) - 1) / (10 * self.bc)
        scale = misc.safe_div(C, CE)
        G = - scale * GE
        J = - scale * JE
        col = ndata.copy()
        col[:, 0] = L
        col[:, 1] = G
        col[:, 2] = J
        return col

    def from_base(self, ndata):
        """
        Transform from LGJOSA (base) to LGJE.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space (LGJOSA).

        Returns
        -------
        col : ndarray
            Colour data in the LGJOSA colour space.
        """
        L = ndata[:, 0]
        G = ndata[:, 1]
        J = ndata[:, 2]
        C = np.sqrt(G**2 + J**2)
        L_E = np.log(1 + 10 * L * self.bL / self.aL) / self.bL
        C_E = np.log(1 + 10 * C * self.bc / self.ac) / self.bc
        scale = misc.safe_div(C_E, C)
        G_E = - scale * G
        J_E = - scale * J
        col = ndata.copy()
        col[:, 0] = L_E
        col[:, 1] = G_E
        col[:, 2] = J_E
        return col

    def jacobian_base(self, data):
        """
        Return the Jacobian from LGJOSA (base), dLGJE^i/dLGJOSA^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        lgj = data.get_flattened(self.base)
        L = lgj[:, 0]
        G = lgj[:, 1]
        J = lgj[:, 2]
        C = np.sqrt(G**2 + J**2)
        lgj_e = data.get_flattened(self)
        C_E = np.sqrt(lgj_e[:, 1]**2 + lgj_e[:, 2]**2)
        dLE_dL = 10 / (self.aL + 10 * self.bL * L)
        dCE_dC = 10 / (self.ac + 10 * self.bc * C)
        dCEC_dC = misc.safe_div(dCE_dC * C - C_E, C**2)
        dC_dG = misc.safe_div(G, C)
        dC_dJ = misc.safe_div(J, C)
        dCEC_dG = dCEC_dC * dC_dG
        dCEC_dJ = dCEC_dC * dC_dJ
        dGE_dG = - misc.safe_div(C_E, C) - G * dCEC_dG
        dGE_dJ = - G * dCEC_dJ
        dJE_dG = - J * dCEC_dG
        dJE_dJ = - misc.safe_div(C_E, C) - J * dCEC_dJ
        jac = self.empty_matrix(lgj)
        jac[:, 0, 0] = dLE_dL
        jac[:, 1, 1] = dGE_dG
        jac[:, 1, 2] = dGE_dJ
        jac[:, 2, 1] = dJE_dG
        jac[:, 2, 2] = dJE_dJ
        return jac


class TransformLogCompressL(Transform):
    """
    Perform parametric logarithmic compression of lightness.

    As in the DIN99x formulae.
    """
    def __init__(self, base, aL, bL):
        """
        Construct instance, setting base space.

        Parameters
        ----------
        base : Space
            The base colour space.
        """
        super(TransformLogCompressL, self).__init__(base)
        self.aL = aL
        self.bL = bL

    def from_base(self, ndata):
        """
        Transform from Lab (base) to L'ab.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space (Lab).

        Returns
        -------
        col : ndarray
            Colour data in the La'b' colour space.
        """
        Lpab = ndata.copy()
        Lpab[:, 0] = self.aL * np.log(1 + self.bL * ndata[:, 0])
        return Lpab

    def to_base(self, ndata):
        """
        Transform from L'ab to Lab (base).

        Parameters
        ----------
        ndata : ndarray
            Colour data in L'ab colour space.

        Returns
        -------
        col : ndarray
            Colour data in the Lab colour space.
        """
        Lab = ndata.copy()
        Lab[:, 0] = (np.exp(ndata[:, 0] / self.aL) - 1) / self.bL
        return Lab

    def jacobian_base(self, data):
        """
        Return the Jacobian from Lab (base), dL'ab^i/dLab^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        lab = data.get_flattened(self.base)
        L = lab[:, 0]
        dLp_dL = self.aL * self.bL / (1 + self.bL * L)
        jac = self.empty_matrix(lab)
        jac[:, 0, 0] = dLp_dL
        jac[:, 1, 1] = 1
        jac[:, 2, 2] = 1
        return jac


class TransformLogCompressC(Transform):
    """
    Perform parametric logarithmic compression of chroma.

    As in the DIN99x formulae.
    """
    def __init__(self, base, aC, bC):
        """
        Construct instance, setting base space.

        Parameters
        ----------
        base : Space
            The base colour space.
        """
        super(TransformLogCompressC, self).__init__(base)
        self.aC = aC
        self.bC = bC

    def from_base(self, ndata):
        """
        Transform from Lab (base) to La'b'.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space (Lab).

        Returns
        -------
        col : ndarray
            Colour data in the La'b' colour space.
        """
        Lapbp = ndata.copy()
        C = np.sqrt(ndata[:, 1]**2 + ndata[:, 2]**2)
        Cp = self.aC * np.log(1 + self.bC * C)
        scale = misc.safe_div(Cp, C)
        ap = scale * ndata[:, 1]
        bp = scale * ndata[:, 2]
        Lapbp[:, 1] = ap
        Lapbp[:, 2] = bp
        return Lapbp

    def to_base(self, ndata):
        """
        Transform from La'b' to Lab (base).

        Parameters
        ----------
        ndata : ndarray
            Colour data in L'ab colour space.

        Returns
        -------
        col : ndarray
            Colour data in the Lab colour space.
        """
        Lab = ndata.copy()
        ap = ndata[:, 1]
        bp = ndata[:, 2]
        Cp = np.sqrt(ap**2 + bp**2)
        C = (np.exp(Cp / self.aC) - 1) / self.bC
        scale = misc.safe_div(Cp, C)
        a = scale * ap
        b = scale * bp
        Lab[:, 1] = a
        Lab[:, 2] = b
        return Lab

    def jacobian_base(self, data):
        """
        Return the Jacobian from Lab (base), dLa'b'^i/dLab^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        lab = data.get_flattened(self.base)
        lapbp = data.get_flattened(self)
        a = lab[:, 1]
        b = lab[:, 2]
        C = np.sqrt(a**2 + b**2)
        Cp = np.sqrt(lapbp[:, 1]**2 + lapbp[:, 2]**2)
        dC_da = misc.safe_div(a, C)
        dC_db = misc.safe_div(b, C)
        dCp_dC = self.aC * self.bC / (1 + self.bC * C)
        dCpC_dC = misc.safe_div(dCp_dC * C - Cp, C**2)
        dap_da = misc.safe_div(Cp, C) + a * (dCpC_dC * dC_da)
        dbp_db = misc.safe_div(Cp, C) + b * (dCpC_dC * dC_db)
        dap_db = a * dCpC_dC * dC_db
        dbp_da = b * dCpC_dC * dC_da
        jac = self.empty_matrix(lab)
        jac[:, 0, 0] = 1
        jac[:, 1, 1] = dap_da
        jac[:, 1, 2] = dap_db
        jac[:, 2, 1] = dbp_da
        jac[:, 2, 2] = dbp_db
        return jac


class TransformPoincareDisk(Transform):
    """
    Transform from Cartesian coordinates to Poincare disk coordinates.

    The coordinate transform only changes the radius (chroma, typically),
    and does so in a way that preserves the radial distance with respect to
    the Euclidean metric and the Poincare disk metric in the source and
    target spaces, respectively.
    """

    def __init__(self, base, R=1.):
        """
        Construct instance, setting base space and radius of curvature.

        Parameters
        ----------
        base : Space
            The base colour space.
        R : float
            The radius of curvature.
        """
        super(TransformPoincareDisk, self).__init__(base)
        self.R = R

    def to_base(self, ndata):
        """
        Transform from Poincare disk to base.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space

        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        Lab = ndata.copy()
        Lab[:, 1:] = 0
        x = ndata[:, 1]
        y = ndata[:, 2]
        r = np.sqrt(x**2 + y**2)
        for i in range(np.shape(Lab)[0]):
            if r[i] > 0:
                Lab[i, 1:] = ndata[i, 1:] * 2 * \
                    self.R * np.arctanh(r[i]) / r[i]
        return Lab

    def from_base(self, ndata):
        """
        Transform from base to Poincare disk

        Parameters
        ----------
        ndata : ndarray
            Colour data in the base colour space.

        Returns
        -------
        col : ndarray
            Colour data in the current colour space.
        """
        Lxy = ndata.copy()
        Lxy[:, 1:] = 0
        a = ndata[:, 1]
        b = ndata[:, 2]
        C = np.sqrt(a**2 + b**2)
        for i in range(np.shape(Lxy)[0]):
            if C[i] > 0:
                Lxy[i, 1:] = ndata[i, 1:] * np.tanh(C[i] / (2 * self.R)) / C[i]
        return Lxy

    def jacobian_base(self, data):
        """
        Return the Jacobian from CIELAB (base), dLxy^i/dCIELAB^j.

        The Jacobian is calculated at the given data points (of the
        Points class).

        Parameters
        ----------
        data : Points
            Colour data points for the jacobians to be computed.

        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        # TODO: bugfix!!!
        Lab = data.get_flattened(self.base)
        a = Lab[:, 1]
        b = Lab[:, 2]
        C = np.sqrt(a**2 + b**2)
        tanhC2R = np.tanh(C / (2. * self.R))
        tanhC2C = misc.safe_div(tanhC2R, C)
        dCda = misc.safe_div(a, C)
        dCdb = misc.safe_div(b, C)
        dtanhdC = misc.safe_div(C / (2. * self.R) *
                                (1 - tanhC2R**2) - tanhC2R, C**2)
        jac = self.empty_matrix(Lab)
        for i in range(np.shape(jac)[0]):
            jac[i, 0, 0] = 1        # dL/dL
            if C[i] == 0:
                jac[i, 1, 1] = .5   # dx/da
                jac[i, 2, 2] = .5   # dy/db
            else:
                jac[i, 1, 1] = tanhC2C[i] + \
                    a[i] * dtanhdC[i] * dCda[i]  # dx/da
                jac[i, 1, 2] = a[i] * dtanhdC[i] * dCdb[i]  # dx/db
                jac[i, 2, 1] = b[i] * dtanhdC[i] * dCda[i]  # dy/da
                jac[i, 2, 2] = tanhC2C[i] + \
                    b[i] * dtanhdC[i] * dCdb[i]  # dy/db
        return jac

# =============================================================================
# Colour space instances
# =============================================================================

# CIE based

xyz = XYZ()
xyY = TransformxyY(xyz)

cielab = TransformCIELAB(xyz)
cielch = TransformPolar(cielab)
cieluv = TransformCIELUV(xyz)
ciede00lab = TransformCIEDE00(cielab)
ciede00lch = TransformPolar(ciede00lab)
ciecat02 = TransformLinear(xyz,
                           np.array([[.7328, .4296, -.1624],
                                     [-.7036, 1.675, .0061],
                                     [.0030, .0136, .9834]]))
ciecat16 = TransformLinear(xyz,
                           np.array([[.401288, .650173, -.051461],
                                     [-.250268, 1.204414, .045854],
                                     [-.002079, .048952, .953127]]))

# sRGB

_srgb_linear = TransformLinear(
    xyz,
    np.array([[3.2404542, -1.5371385, -0.4985314],
              [-0.9692660,  1.8760108,  0.0415560],
              [0.0556434, -0.2040259,  1.0572252]]))
srgb = TransformSRGB(_srgb_linear)

# Adobe RGB

_rgb_adobe_linear = TransformLinear(
    xyz,
    np.array([[2.0413690, -0.5649464, -0.3446944],
              [-0.9692660, 1.8760108, 0.0415560],
              [0.0134474, -0.1183897, 1.0154096]]))
rgb_adobe = TransformGamma(_rgb_adobe_linear, 1 / 2.2)

# IPT

_ipt_lms = TransformLinear(
    xyz,
    np.array([[.4002, .7075, -.0807],
              [-.228, 1.15, .0612],
              [0, 0, .9184]]))
_ipt_lmsp = TransformGamma(_ipt_lms, .43)
ipt = TransformLinear(
    _ipt_lmsp,
    np.array([[.4, .4, .2],
              [4.455, -4.850, .3960],
              [.8056, .3572, -1.1628]]))

# OSA-UCS

lgj_osa = TransformLGJOSA(xyz)
lgj_e = TransformLGJE(lgj_osa)

# DIN99

_din99_lpab = TransformLogCompressL(cielab, 105.51, 0.0158)
_din99_lef = TransformLinear(
    _din99_lpab,
    np.array([[1, 0, 0],
              [0, np.cos(np.deg2rad(16.)),
               np.sin(np.deg2rad(16.))],
              [0, - 0.7 * np.sin(np.deg2rad(16.)),
               0.7 * np.cos(np.deg2rad(16.))]]))
din99 = TransformLogCompressC(_din99_lef, 1 / 0.045, 0.045)

# DIN99b

_din99b_lpab = TransformLogCompressL(cielab, 303.67, 0.0039)
_din99b_lef = TransformLinear(
    _din99b_lpab,
    np.array([[1, 0, 0],
              [0, np.cos(np.deg2rad(26.)), np.sin(np.deg2rad(26.))],
              [0, - 0.83 * np.sin(np.deg2rad(26.)),
               0.83 * np.cos(np.deg2rad(26.))]]))
_din99b_rot = TransformLogCompressC(_din99b_lef, 23.0, 0.075)
din99b = TransformLinear(
    _din99b_rot,
    np.array([[1, 0, 0],
              [0, np.cos(np.deg2rad(-26.)), np.sin(np.deg2rad(-26.))],
              [0, - np.sin(np.deg2rad(-26.)), np.cos(np.deg2rad(-26.))]]))

# DIN99c

_din99c_xyz = TransformLinear(xyz,
                              np.array([[1.1, 0, -0.1],
                                        [0, 1, 0],
                                        [0, 0, 1]]))
_din99c_white = np.dot(_din99c_xyz.M, _din99c_xyz.white_D65)
_din99c_lab = TransformCIELAB(_din99c_xyz, _din99c_white)
_din99c_lpab = TransformLogCompressL(_din99c_lab, 317.65, 0.0037)
_din99c_lef = TransformLinear(_din99c_lpab,
                              np.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, .94]]))
din99c = TransformLogCompressC(_din99c_lef, 23., 0.066)

# DIN99d

_din99d_xyz = TransformLinear(xyz,
                              np.array([[1.12, 0, -0.12],
                                        [0, 1, 0],
                                        [0, 0, 1]]))
_din99d_white = np.dot(_din99d_xyz.M, _din99d_xyz.white_D65)
_din99d_lab = TransformCIELAB(_din99d_xyz, _din99d_white)
_din99d_lpab = TransformLogCompressL(_din99c_lab, 325.22, 0.0036)
_din99d_lef = TransformLinear(
    _din99d_lpab,
    np.array([[1, 0, 0],
              [0, np.cos(np.deg2rad(50.)), np.sin(np.deg2rad(50.))],
              [0, - 1.14 * np.sin(np.deg2rad(50.)),
               1.14 * np.cos(np.deg2rad(50.))]]))
_din99d_rot = TransformLogCompressC(_din99d_lef, 23., 0.066)
din99d = TransformLinear(
    _din99d_rot,
    np.array([[1, 0, 0],
              [0, np.cos(np.deg2rad(-50.)), np.sin(np.deg2rad(-50.))],
              [0, - np.sin(np.deg2rad(-50.)), np.cos(np.deg2rad(-50.))]]))
