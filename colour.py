#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
colour: Colour spaces, colour metrics and colour data>

Copyright (C) 2011-2013 Ivar Farup

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

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

#==============================================================================
# Colour space classes
#
# Throughout the code, the name ndata is used for numerical data (numpy
# arrays), and data is used for objects of the type Data, and mdata for
# objects of the type MetricData.
#==============================================================================

class Space(object):
    """
    Base class for the colour space classes.
    """

    white_A = np.array([1.0985, 1., 0.35585])
    white_B = np.array([.990720, 1., .852230])
    white_C = np.array([.980740, 1., .82320])
    white_D50 = np.array([.964220, 1., .82510])
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
        """
        return np.zeros((np.shape(ndata)[0], 3, 3))

    def jacobian_XYZ(self, data):
        """
        Return the Jacobian to XYZ, dx^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class) by inverting the inverse Jacobian.
        """
        jac = self.inv_jacobian_XYZ(data)
        for i in range(np.shape(jac)[0]):
            jac[i] = np.linalg.inv(jac[i])
        return jac

    def inv_jacobian_XYZ(self, data):
        """
        Return the inverse Jacobian to XYZ, dXYZ^i/dx^j.

        The inverse Jacobian is calculated at the given data points
        (of the Data class) by inverting the Jacobian.
        """
        ijac = self.jacobian_XYZ(data)
        for i in range(np.shape(ijac)[0]):
            ijac[i] = np.linalg.inv(ijac[i])
        return ijac
    
    def metrics_to_XYZ(self, points_data, metrics_ndata):
        """
        Convert metric data to the XYZ colour space.
        """
        jacobian = self.jacobian_XYZ(points_data)
        new_metric = np.zeros(np.shape(metrics_ndata))
        for i in range(np.shape(jacobian)[0]):
            new_metric[i] = np.dot(jacobian[i].T,
                                   np.dot(metrics_ndata[i], jacobian[i]))
        return new_metric
    
    def metrics_from_XYZ(self, points_data, metrics_ndata):
        """
        Convert metric data from the XYZ colour space.
        """
        jacobian = self.inv_jacobian_XYZ(points_data)
        new_metric = np.zeros(np.shape(metrics_ndata))
        for i in range(np.shape(jacobian)[0]):
            new_metric[i] = np.dot(jacobian[i].T,
                                   np.dot(metrics_ndata[i], jacobian[i]))
        return new_metric

class SpaceXYZ(Space):
    """
    The XYZ colour space.

    Assumes that the CIE 1931 XYZ colour matching functions are
    used. The white point is D65. Serves a special role in the code in that
    it serves as a common reference point.
    """

    def to_XYZ(self, ndata):
        """
        Convert from current colour space to XYZ.
        """
        return ndata.copy()      # identity transform
    
    def from_XYZ(self, ndata):
        """
        Convert from XYZ to current colour space.
        """
        return ndata.copy()      # identity transform
        
    def jacobian_XYZ(self, data):
        """
        Return the Jacobian to XYZ, dx^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        jac = self.empty_matrix(data.linear_XYZ)
        jac[:] = np.eye(3)
        return jac

    def inv_jacobian_XYZ(self, data):
        """
        Return the inverse Jacobian to XYZ, dXYZ^i/dx^j.

        The inverse Jacobian is calculated at the given data points
        (of the Data class).
        """
        ijac = self.empty_matrix(data.linear_XYZ)
        ijac[:] = np.eye(3)
        return ijac

class SpaceTransform(Space):
    """
    Base class for colour space transforms.
    
    Real transforms (children) must implement to_base, from_base and either
    jacobian_base or inv_jacobian_base.
    """
    
    def __init__(self, base):
        """
        Construct instance and set base space for transformation
        """
        self.base = base
        
    def to_XYZ(self, ndata):
        """
        Transform data to XYZ by using the transformation to the base.
        """
        return self.base.to_XYZ(self.to_base(ndata))
        
    def from_XYZ(self, ndata):
        """
        Transform data from XYZ using the transformation to the base.
        """
        return self.from_base(self.base.from_XYZ(ndata))

    def jacobian_base(self, data):
        """
        Return the Jacobian to base, dx^i/dbase^j.

        The Jacobian is calculated at the given data points (of the
        Data class) by inverting the inverse Jacobian.
        """
        jac = self.inv_jacobian_base(data)
        for i in range(np.shape(jac)[0]):
            jac[i] = np.linalg.inv(jac[i])
        return jac

    def inv_jacobian_base(self, data):
        """
        Return the inverse Jacobian to base, dbase^i/dx^j.

        The inverse Jacobian is calculated at the given data points
        (of the Data class) by inverting the Jacobian.
        """
        ijac = self.jacobian_base(data)
        for i in range(np.shape(ijac)[0]):
            ijac[i] = np.linalg.inv(ijac[i])
        return ijac
        
    def jacobian_XYZ(self, data):
        """
        Return the Jacobian to XYZ, dx^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class) using the jacobian to the base and the Jacobian
        of the base space.
        """
        dxdbase = self.jacobian_base(data)
        dbasedXYZ = self.base.jacobian_XYZ(data)
        jac = self.empty_matrix(data.linear_XYZ)
        for i in range(np.shape(jac)[0]):
            jac[i] = np.dot(dxdbase[i], dbasedXYZ[i])
        return jac

    def inv_jacobian_XYZ(self, data):
        """
        Return the inverse Jacobian to XYZ, dXYZ^i/dx^j.

        The Jacobian is calculated at the given data points (of the
        Data class) using the inverse jacobian to the base and the 
        inverse Jacobian of the base space.
        """
        dXYZdbase = self.base.inv_jacobian_XYZ(data)
        dbasedx = self.inv_jacobian_base(data)
        ijac = self.empty_matrix(data.linear_XYZ)
        for i in range(np.shape(ijac)[0]):
            ijac[i] = np.dot(dXYZdbase[i], dbasedx[i])
        return ijac

class TransformxyY(SpaceTransform):
    """
    The XYZ to xyY projective transform.
    """

    def __init__(self, base):
        """
        Construct instance.
        """
        super(TransformxyY, self).__init__(base)
    
    def to_base(self, ndata):
        """
        Convert from xyY to XYZ.
        """
        xyz = np.zeros(np.shape(ndata))
        xyz[:,0] = ndata[:,0]*ndata[:,2]/ndata[:,1]                        # X
        xyz[:,1] = ndata[:,2]                                              # Y
        xyz[:,2] = (1 - ndata[:,0] - ndata[:,1]) * ndata[:,2] / ndata[:,1] # Z
        return xyz
    
    def from_base(self, ndata):
        """
        Convert from XYZ to xyY.
        """
        xyz = ndata
        xyY = np.zeros(np.shape(xyz))
        xyz_sum = np.sum(xyz, axis=1)
        xyY[:,0] = xyz[:,0] / xyz_sum # x
        xyY[:,1] = xyz[:,1] / xyz_sum # y
        xyY[:,2] = xyz[:,1]           # Y
        return xyY
    
    def jacobian_base(self, data):
        """
        Return the Jacobian to XYZ, dxyY^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        xyzdata = data.get_linear(self.base)
        jac = self.empty_matrix(xyzdata)
        for i in range(np.shape(jac)[0]):
            jac[i,0,0] = (xyzdata[i,1] + xyzdata[i,2]) / \
                (xyzdata[i,0] + xyzdata[i,1] + xyzdata[i,2]) ** 2
            jac[i,0,1] = -xyzdata[i,0] / \
                (xyzdata[i,0] + xyzdata[i,1] + xyzdata[i,2]) ** 2
            jac[i,0,2] = -xyzdata[i,0] / \
                (xyzdata[i,0] + xyzdata[i,1] + xyzdata[i,2]) ** 2           
            jac[i,1,0] = -xyzdata[i,1] / \
                (xyzdata[i,0] + xyzdata[i,1] + xyzdata[i,2]) ** 2           
            jac[i,1,1] = (xyzdata[i,0] + xyzdata[i,2]) / \
                (xyzdata[i,0] + xyzdata[i,1] + xyzdata[i,2]) ** 2           
            jac[i,1,2] = -xyzdata[i,1] / \
                (xyzdata[i,0] + xyzdata[i,1] + xyzdata[i,2]) ** 2           
            jac[i,2,1] = 1
        return jac
    
    def inv_jacobian_base(self, data):
        """
        Return the Jacobian from XYZ, dXYZ^i/dxyY^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        xyYdata = data.get_linear(self)
        jac = self.empty_matrix(xyYdata)
        for i in range(np.shape(jac)[0]):
            jac[i,0,0] = xyYdata[i,2] / xyYdata[i,1]
            jac[i,0,1] = - xyYdata[i,0] * xyYdata[i,2] / xyYdata[i,1] ** 2
            jac[i,0,2] = xyYdata[i,0] / xyYdata[i,1]
            jac[i,1,2] = 1
            jac[i,2,0] = - xyYdata[i,2] / xyYdata[i,1]
            jac[i,2,1] = xyYdata[i,2] * (xyYdata[i,0] - 1) / \
                xyYdata[i,1] ** 2
            jac[i,2,2] = (1 - xyYdata[i,0] - xyYdata[i,1]) / xyYdata[i,1]
        return jac

class TransformCIELAB(SpaceTransform):
    """
    The XYZ to CIELAB colour space transform.

    The white point is a parameter in the transform.
    """
    kappa = 24389. / 27.        # standard: 903.3
    epsilon = 216. / 24389.     # standard: 0.008856
    
    def __init__(self, base, white_point=Space.white_D65):
        """
        Construct instance by setting base space and magnification factor.
        """
        super(TransformCIELAB, self).__init__(base)
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
            (ndata[ndata > self.epsilon] ** (-2. /3)) / 3
        return df
    
    def to_base(self, ndata):
        """
        Convert from CIELAB to XYZ (base).
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
        # Combine into matrix
        xyz = np.zeros(np.shape(ndata))
        xyz[:,0] = xr * self.white_point[0]
        xyz[:,1] = yr * self.white_point[1]
        xyz[:,2] = zr * self.white_point[2]
        return xyz
    
    def from_base(self, ndata):
        """
        Convert from XYZ (base) to CIELAB.
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
        Data class).
        """
        d = data.get_linear(self.base)
        dr = d.copy()
        for i in range(3):
            dr[:, i] = dr[:, i] / self.white_point[i]
        df = self.dfdx(dr)
        jac = self.empty_matrix(d)
        jac[:, 0, 1] = 116 * df[:, 1] / self.white_point[1]  # dL/dY
        jac[:, 1, 0] = 500 * df[:, 0] / self.white_point[0]  # da/dX
        jac[:, 1, 1] = -500 * df[:, 1] / self.white_point[1] # da/dY
        jac[:, 2, 1] = 200 * df[:, 1] / self.white_point[1]  # db/dY
        jac[:, 2, 2] = -200 * df[:, 2] / self.white_point[2] # db/dZ
        return jac

class TransformCIELUV(SpaceTransform):
    """
    The XYZ to CIELUV colour space transform.

    The white point is a parameter in the transform.
    """
    kappa = 24389. / 27.        # standard: 903.3
    epsilon = 216. / 24389.     # standard: 0.008856
    
    def __init__(self, base, white_point=Space.white_D65):
        """
        Construct instance by setting base space and magnification factor.
        """
        super(TransformCIELUV, self).__init__(base)
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
            (ndata[ndata > self.epsilon] ** (-2. /3)) / 3
        return df
    
    def to_base(self, ndata):
        """
        Convert from CIELUV to XYZ (base).
        """
        luv = ndata
        fy = (luv[:, 0] + 16.) / 116.
        y = fy ** 3
        y[luv[:, 0] <= self.kappa * self.epsilon] = \
            luv[luv[:, 0] <= self.kappa * self.epsilon, 0] / self.kappa
        upr = 4 * self.white_point[0] / (self.white_point[0] + 15*self.white_point[1] + 3*self.white_point[2])
        vpr = 9 * self.white_point[1] / (self.white_point[0] + 15*self.white_point[1] + 3*self.white_pointS[2])
        a = (52*luv[:,0] / (luv[:,1] + 13*luv[:,0]*upr) - 1) / 3
        b = -5 * y
        c = -1/3.
        d = y * (39*luv[:,0] / (luv[:,2] + 13*luv[:,0]*vpr) - 5)
        x = (d - b) / (a - c)
        z = x * a + b
        # Combine into matrix
        xyz = np.zeros(np.shape(luv))
        xyz[:,0] = x
        xyz[:,1] = y
        xyz[:,2] = z
        return xyz
    
    def from_base(self, ndata):
        """
        Convert from XYZ (base) to CIELUV.
        """
        d = ndata
        luv = np.zeros(np.shape(d))
        fy = self.f(d[:, 1] / self.white_point[1])
        up = 4 * d[:,0] / (d[:,0] + 15*d[:,1] + 3*d[:,2])
        upr = 4 * self.white_point[0] / (self.white_point[0] + 15*self.white_point[1] + 3*self.white_point[2])
        vp = 9 * d[:,1] / (d[:,0] + 15*d[:,1] + 3*d[:,2])
        vpr = 9 * self.white_point[1] / (self.white_point[0] + 15*self.white_point[1] + 3*self.white_point[2])
        luv[:, 0] = 116. * fy - 16.
        luv[:, 1] = 13 * luv[:, 0] * (up - upr)
        luv[:, 2] = 13 * luv[:, 0] * (vp - vpr)
        return luv
    
    def jacobian_base(self, data):
        """
        Return the Jacobian to XYZ (base), dCIELUV^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        xyz = data.get_linear(spaceXYZ)
        luv = data.get_linear(spaceCIELUV)
        df = self.dfdx(xyz)
        jac = self.empty_matrix(xyz)
        # dL/dY:
        jac[:, 0, 1] = 116 * df[:, 1] / self.white_point[1]
        # du/dX:
        jac[:, 1, 0] = 13 * luv[:,0] * \
            (60 * xyz[:,1] + 12 * xyz[:,2]) / \
            (xyz[:,0] + 15 * xyz[:,1] + 3 * xyz[:,2]) ** 2
        # du/dY:
        jac[:, 1, 1] = 13 * luv[:,0] * \
            -60 * xyz[:,0] / \
            (xyz[:,0] + 15 * xyz[:,1] + 3 * xyz[:,2]) ** 2 + \
            13 * jac[:, 0, 1] * (
                4 * xyz[:,0] / (xyz[:,0] + 15 * xyz[:,1] + 3 * xyz[:,2]) -
                4 * self.white_point[0] / \
                (self.white_point[0] + 15 * self.white_point[1] + 3 * self.white_point[2]))
        # du/dZ:
        jac[:, 1, 2] = 13 * luv[:,0] * \
            -12 * xyz[:,0] / \
            (xyz[:,0] + 15 * xyz[:,1] + 3 * xyz[:,2]) ** 2
        # dv/dX:
        jac[:, 2, 0] = 13 * luv[:,0] * \
            -9 * xyz[:,1] / \
            (xyz[:,0] + 15 * xyz[:,1] + 3 * xyz[:,2]) ** 2
        # dv/dY:
        jac[:, 2, 1] = 13 * luv[:,0] * \
            (9 * xyz[:,0] + 27 * xyz[:,2]) / \
            (xyz[:,0] + 15 * xyz[:,1] + 3 * xyz[:,2]) ** 2 + \
            13 * jac[:, 0, 1] * (
                9 * xyz[:,1] / (xyz[:,0] + 15 * xyz[:,1] + 3 * xyz[:,2]) -
                9 * self.white_point[1] / \
                (self.white_point[0] + 15 * self.white_point[1] + 3 * self.white_point[2]))
        # dv/dZ:
        jac[:, 2, 2] = 13 * luv[:,0] * \
            -27 * xyz[:,1] / \
            (xyz[:,0] + 15 * xyz[:,1] + 3 * xyz[:,2]) ** 2
        return jac

class TransformLinear(SpaceTransform):
    """
    General linear transform, transformed = M * base
    """

    def __init__(self, base, M=np.eye(3)):
        """
        Construct instance, setting the matrix of the linear transfrom.
        """
        super(TransformLinear, self).__init__(base)
        self.M = M.copy()
        self.M_inv = np.linalg.inv(M)
    
    def to_base(self, ndata):
        """
        Convert from linear to XYZ.
        """
        xyz = np.zeros(np.shape(ndata))
        for i in range(np.shape(ndata)[0]):
            xyz[i] = np.dot(self.M_inv, ndata[i])
        return xyz
    
    def from_base(self, ndata):
        """
        Convert from XYZ to linear.
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
        Data class).
        """
        xyzdata = data.get_linear(spaceXYZ)
        jac = self.empty_matrix(xyzdata)
        jac[:] = self.M
        return jac
    
    def inv_jacobian_base(self, data):
        """
        Return the Jacobian from XYZ (base), dXYZ^i/dlinear^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        xyzdata = data.get_linear(spaceXYZ)
        jac = self.empty_matrix(xyzdata)
        jac[:] = self.M_inv
        return jac

class TransformGamma(SpaceTransform):
    """
    General gamma transform, transformed = base**gamma
    
    Uses absolute value and sign for negative base values:
    transformed = sign(base) * abs(base)**gamma
    """

    def __init__(self, base, gamma=1):
        """
        Construct instance, setting the gamma of the transfrom.
        """
        super(TransformGamma, self).__init__(base)
        self.gamma = float(gamma)
        self.gamma_inv = 1. / gamma
    
    def to_base(self, ndata):
        """
        Convert from gamma corrected to XYZ (base).
        """
        return np.sign(ndata) * np.abs(ndata)**self.gamma_inv

    def from_base(self, ndata):
        """
        Convert from XYZ to gamma corrected.
        """
        return np.sign(ndata) * np.abs(ndata)**self.gamma
        
    def jacobian_base(self, data):
        """
        Return the Jacobian to XYZ (base), dgamma^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        basedata = data.get_linear(self.base)
        jac = self.empty_matrix(basedata)
        for i in range(np.shape(basedata)[0]):
            jac[i, 0, 0] = self.gamma * np.abs(basedata[i, 0])**(self.gamma - 1)
            jac[i, 1, 1] = self.gamma * np.abs(basedata[i, 1])**(self.gamma - 1)
            jac[i, 2, 2] = self.gamma * np.abs(basedata[i, 2])**(self.gamma - 1)
        return jac

class TransformPolar(SpaceTransform):
    """
    Transform form Cartesian to polar coordinates in the two last variables.
    
    For example CIELAB to CIELCH.
    """
    
    def __init__(self, base):
        """
        Construct instance, setting base space.
        """
        super(TransformPolar, self).__init__(base)
    
    def to_base(self, ndata):
        """
        Convert from polar to rectangular.
        """
        Lab = np.zeros(np.shape(ndata))
        Lab[:,0] = ndata[:,0]
        C = ndata[:,1]
        h = ndata[:,2]
        Lab[:,1] = C * np.cos(h)
        Lab[:,2] = C * np.sin(h)
        return Lab
    
    def from_base(self, ndata):
        """
        Convert from rectangular (base) to polar.
        """
        LCh = np.zeros(np.shape(ndata))
        LCh[:,0] = ndata[:,0]
        x = ndata[:,1]
        y = ndata[:,2]
        LCh[:,1] = np.sqrt(x**2 + y**2)
        LCh[:,2] = np.arctan2(y, x)
        return LCh
        
    def inv_jacobian_base(self, data):
        """
        Return the Jacobian from CIELAB (base), dCIELAB^i/dCIELCH^j.
        
        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        LCh = data.get_linear(self)
        C = LCh[:,1]
        h = LCh[:,2]
        jac = self.empty_matrix(LCh)
        for i in range(np.shape(jac)[0]):
            jac[i,0,0] = 1 # dL/dL
            jac[i,1,1] = np.cos(h[i]) # da/dC
            jac[i,1,2] = -C[i] * np.sin(h[i]) # da/dh
            jac[i,2,1] = np.sin(h[i]) # db/dC
            jac[i,2,2] = C[i] * np.cos(h[i]) # db/dh
        return jac

class TransformCartesian(SpaceTransform):
    """
    Transform form polar to Cartesian coordinates in the two last variables.
    
    For example CIELCH to CIELAB.
    """
    
    def __init__(self, base):
        """
        Construct instance, setting base space.
        """
        super(TransformCartesian, self).__init__(base)
    
    def from_base(self, ndata):
        """
        Convert from polar to rectangular.
        """
        Lab = np.zeros(np.shape(ndata))
        Lab[:,0] = ndata[:,0]
        C = ndata[:,1]
        h = ndata[:,2]
        Lab[:,1] = C * np.cos(h)
        Lab[:,2] = C * np.sin(h)
        return Lab
    
    def to_base(self, ndata):
        """
        Convert from rectangular (base) to polar.
        """
        LCh = np.zeros(np.shape(ndata))
        LCh[:,0] = ndata[:,0]
        x = ndata[:,1]
        y = ndata[:,2]
        LCh[:,1] = np.sqrt(x**2 + y**2)
        LCh[:,2] = np.arctan2(y, x)
        return LCh
        
    def jacobian_base(self, data):
        """
        Return the Jacobian from CIELAB (base), dCIELAB^i/dCIELCH^j.
        
        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        LCh = data.get_linear(self.base)
        C = LCh[:,1]
        h = LCh[:,2]
        jac = self.empty_matrix(LCh)
        for i in range(np.shape(jac)[0]):
            jac[i,0,0] = 1 # dL/dL
            jac[i,1,1] = np.cos(h[i]) # da/dC
            jac[i,1,2] = -C[i] * np.sin(h[i]) # da/dh
            jac[i,2,1] = np.sin(h[i]) # db/dC
            jac[i,2,2] = C[i] * np.cos(h[i]) # db/dh
        return jac

#==============================================================================
# Colour space instances
#==============================================================================

#Basic

spaceXYZ = SpaceXYZ()
spacexyY = TransformxyY(spaceXYZ)
spaceCIELAB = TransformCIELAB(spaceXYZ)
spaceCIELCh = TransformPolar(spaceCIELAB)
spaceCIELUV = TransformCIELUV(spaceXYZ)
spaceIPT = TransformLinear(TransformGamma(TransformLinear(spaceXYZ,
                np.array([[.4002, .7075, -.0807],
                          [-.228, 1.15, .0612],
                          [0, 0, .9184]])),
                .43),
                np.array([[.4, .4, .2],
                          [4.455, -4.850, .3960],
                          [.8056, .3572, -1.1628]]))

#==============================================================================
# Colour data
#==============================================================================

class Data:
    """
    Class for keeping colour data in various colour spaces and shapes.
    """

    def __init__(self, space, ndata):
        """
        Construct new instance and set colour space and data.
        """
        self.set(space, ndata)

    def linearise(self, ndata):
        """
        Shape the data so that is becomes an PxC matrix or C vector.
        
        The data should be of the shape Mx...xNxC, where C is the
        number of colour channels. Returns the shaped data as a PxC
        matrix where P=Mx...xN, as well as the shape of the input
        data. Get back to original shape by reshape(data, shape).
        """
        sh = np.shape(ndata)
        sh_array = np.array(sh)
        P_data = np.prod(sh_array[:len(sh) - 1])
        C_data = sh[len(sh) - 1]
        return np.reshape(ndata, (P_data, C_data))

    def set(self, space, ndata):
        """
        Set colour space and data.

        A new dictionary is constructed, and the data are added in the
        provided colour space, as well as in the XYZ colour space
        (using the SpaceXYZ class).
        """
        ndata = np.array(ndata)
        self.data = dict()
        self.data[space] = ndata
        self.sh = ndata.shape
        linear_data = self.linearise(ndata)
        if space == spaceXYZ:
            self.linear_XYZ = linear_data
        else:
            self.linear_XYZ = space.to_XYZ(linear_data)
            self.data[spaceXYZ] = np.reshape(self.linear_XYZ, self.sh)

    def get(self, space):
        """
        Return colour data in required colour space.
        
        If the data do not currently exist in the required colour
        space, the necessary colour conversion will take place, and
        the results stored in the object or future use.
        """
        if self.data.has_key(space):
            return self.data[space]
        else:
            linear_data = space.from_XYZ(self.linear_XYZ)
            ndata = np.reshape(linear_data, self.sh)
            self.data[space] = ndata
            return ndata
            
    def get_linear(self, space):
        """
        Return colour data in required colour space in PxC format.
        
        If the data do not currently exist in the required colour
        space, the necessary colour conversion will take place, and
        the results stored in the object or future use.
        """
        return self.linearise(self.get(space))

class MetricData:
    """
    Class for keeping colour metric data in various colour spaces.
    """

    # Cross sectional planes for ellipses    
    plane_xy = np.s_[0:2]
    plane_ab = np.s_[1:3]
    plane_aL = np.s_[1::-1]
    plane_bL = np.s_[2::-2]

    def __init__(self, space, points_data, metrics_ndata):
        """
        Construct new instance and set colour space and data.
        """
        self.set(space, points_data, metrics_ndata)

    def set(self, space, points_data, metrics_ndata):
        """
        Set colour space, points, and metrics data.

        The points_data are taken care already of the type Data. A new
        dictionary is constructed, and the metrics_ndata are added in
        the provided colour space, as well as in the XYZ colour space
        (using the SpaceXYZ class).
        """
        self.points = points_data
        self.metrics = dict()
        self.metrics[space] = metrics_ndata
        if space != spaceXYZ:
            self.metrics[spaceXYZ] = \
                space.metrics_to_XYZ(points_data, metrics_ndata)

    def get(self, space):
        """
        Return metric data in required colour space.

        If the data do not currently exist in the required colour
        space, the necessary colour conversion will take place, and
        the results stored in the object or future use.
        """
        if self.metrics.has_key(space):
            return self.metrics[space]
        else:
            self.metrics[space] = \
                space.metrics_from_XYZ(self.points, self.metrics[spaceXYZ])
            return self.metrics[space]

    def get_ellipses(self, space, plane=plane_xy, scale=1):
        """
        Return Ellipse objects in the required plane of the given space.

        For now, plane is represented by a slice giving the correct
        range for the arrays. Should perhaps be changed in the future.
        """
        metrics = self.get(space).copy()
        points = self.points.get_linear(space).copy()
        metrics = metrics[:, plane, plane]
        points = points[:, plane]
        ells = []
        for i in range(np.shape(metrics)[0]):
            g11 = metrics[i, 0, 0]
            g22 = metrics[i, 1, 1]
            g12 = metrics[i, 0, 1]
            if g11 == g22:
                theta = 0
                a = 1 / np.sqrt(g22)
                b = 1 / np.sqrt(g11)
            else:
                theta = np.arctan2(2*g12, g11 - g22) * 0.5
                a = 1 / np.sqrt(g22 + g12 / np.tan(theta))
                b = 1 / np.sqrt(g11 - g12 / np.tan(theta))
            ells.append(Ellipse(points[i],
                                width=2*a*scale, height=2*b*scale,
                                angle=theta * 180 / np.pi))
        return ells

#==============================================================================
# Colour data sets
#==============================================================================

def resource_path(relative):
    """
    Extend relative path to full path (mainly for PyInstaller integration).
    """
    return os.path.join(
        os.environ.get(
            "_MEIPASS2",
            os.path.abspath(".")
        ),
        relative
    )
    

def read_csv_file(filename, pad=-np.inf):
    """
    Read a CSV file and return pylab array.

    Parameters
    ----------
    filename : string
        Name of the CSV file to read
    pad : float
        Value to pad for missing values.
    
    Returns
    -------
    csv_array : ndarray
        The content of the file plus padding.
    """
    f = open(resource_path(filename))
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].split(',')
        for j in range(len(data[i])):
            if data[i][j].strip() == '':
                data[i][j] = pad
            else:
                data[i][j] = float(data[i][j])
    return np.array(data)

def build_d_XYZ_31():
    """
    Read CIE XYZ 1931 functions.
    """
    xyz = read_csv_file('data/ciexyz31_1.csv')
    return Data(spaceXYZ, xyz[:,1:])

def build_d_XYZ_64():
    """
    Read CIE XYZ 1964 functions.
    """
    xyz = read_csv_file('data/ciexyz64_1.csv')
    return Data(spaceXYZ, xyz[:,1:])

def build_d_Melgosa():
    """
    The data points for the Melgosa Ellipsoids (RIT-DuPont).

    Copied verbatim from pdf of CRA paper. Uses the ellipsoids fitted
    in CIELAB and returns MetricData.
    """
    m_a = np.array([-1.403 ,-16.374, -0.782, -27.549, 12.606, 12.153,
                     35.646, 1.937, -10.011, -0.453, -30.732, 21.121,
                     -33.638, -13.440, 25.237, 31.509, 6.826, 0.307,
                     18.226])
    m_b = np.array([-27.810, -11.263, 1.049, 2.374, 20.571, -13.079, 
                     21.403, 35.638, 13.281, 0.421, -5.030, 17.804, 
                     -5.012, -25.897, 3.409, -0.183, -31.146, 0.214,
                     79.894])
    m_L = np.array([35.338, 50.259, 59.334, 55.618, 62.928, 46.389,
                    42.315, 78.023, 64.938, 14.140, 68.678, 28.893,
                    31.683, 59.904, 17.357, 58.109, 30.186, 83.481,
                    76.057])
    m_Lab = np.concatenate(([m_L], [m_a], [m_b]), axis=0).T
    return Data(spaceCIELAB, m_Lab)

# TODO:
#
# Colour data sets, as needed (instances of Data):
#     patches_Munsell ++
#     patches_OSA ++ ???
#     patches_Colour Checker ++

#==============================================================================
# Metric data sets
#==============================================================================

def build_g_MacAdam():
    """
    MacAdam ellipses (defined in xy, extended arbitrarily to xyY).
    
    Arbitrarily uses Y=0.4 and g33 = 1e3 for extension to 3D
    """
    import scipy.io
    rawdata = scipy.io.loadmat('metric_data/macdata(xyabtheta).mat')
    rawdata = rawdata['unnamed']
    xyY = rawdata[:,0:3].copy()
    xyY[:,2] = 0.4 # arbitrary!
    points = Data(spacexyY, xyY)
    a = rawdata[:,2]/1e3
    b = rawdata[:,3]/1e3
    theta = rawdata[:,4]*np.pi/180.
    g11 = (np.cos(theta)/a)**2 + (np.sin(theta)/b)**2
    g22 = (np.sin(theta)/a)**2 + (np.cos(theta)/b)**2
    g12 = np.cos(theta)*np.sin(theta)*(1/a**2 - 1/b**2)
    g = np.zeros((25,3,3))
    g[:, 0, 0] = g11
    g[:, 1, 1] = g22
    g[:, 2, 2] = 1e3 # arbitrary!
    g[:, 0, 1] = g12
    g[:, 1, 0] = g12
    return MetricData(spacexyY, points, g)

def build_g_ThreeObserver():
    """
    Wyszecki and Fielder's three observer data set.
    
    Arbitrarily uses Y=0.4 and g33 = 1e3 for extension to 3D. It seems by
    comparing the data file to the original paper by Wyszecki and Fielder
    (JOSA, 1971) that only one of the data sets (GW) is represented in the
    file. Also, the paper reports a full 3D metric, so the arbitrary extension
    to 3D used here is not really called for.
    """
    f = file('metric_data/3 observer.txt')
    rawdata = f.readlines()[:-1]
    for line in range(len(rawdata)):
        rawdata[line] = rawdata[line].split('\t')
        for item in range(len(rawdata[line])):
            rawdata[line][item] = float(rawdata[line][item].strip())
    rawdata = np.array(rawdata)
    xyY = rawdata[:,1:4].copy()
    xyY[:,2] = 0.4 # arbitrary!
    points = Data(spacexyY, xyY)
    a = rawdata[:,4]/1e3 # correct?
    b = rawdata[:,5]/1e3 # corect?
    theta = rawdata[:,3]*np.pi/180.
    g11 = (np.cos(theta)/a)**2 + (np.sin(theta)/b)**2
    g22 = (np.sin(theta)/a)**2 + (np.cos(theta)/b)**2
    g12 = np.cos(theta)*np.sin(theta)*(1/a**2 - 1/b**2)
    g = np.zeros((28,3,3))
    g[:, 0, 0] = g11
    g[:, 1, 1] = g22
    g[:, 2, 2] = 1e3 # arbitrary!
    g[:, 0, 1] = g12
    g[:, 1, 0] = g12
    return MetricData(spacexyY, points, g)

def build_g_Melgosa_Lab():
    """
    Melgosa's CIELAB-fitted ellipsoids for the RIT-DuPont data.

    Copied verbatim from pdf of CRA paper. Uses the ellipsoids fitted
    in CIELAB and returns MetricData.
    """
    m_gaa = np.array([0.6609, 0.3920, 1.3017, 0.1742, 0.5967, 0.5374,
                      0.2837, 0.6138, 0.7252, 1.6002, 0.1760, 0.8512,
                      0.0543, 0.3547, 0.2381, 0.1729, 0.7289, 0.9614,
                      0.2896])
    m_gbb = np.array([0.2387, 0.4286, 0.5241, 0.5847, 0.4543, 0.3048,
                      0.3717, 0.2465, 0.4370, 0.4790, 0.2589, 0.4054,
                      0.7178, 0.2057, 0.3801, 0.2532, 0.4255, 0.1984,
                      0.0522])
    m_gab = np.array([0.3080, -0.0386, 0.1837, 0.0632, -0.1913, 0.2772,
                      -0.1215, -0.0757, 0.1565, 0.0971, 0.0941, -0.2578,
                      -0.1148, 0.1671, 0.0229, 0.0362, 0.5275, 0.1822,
                      0.0023])
    m_gLa = np.array([-0.0144, 0.0812, -0.1435, 0.0996, -0.0008, -0.0115,
                       0.0644, 0.0315, 0.2465, -0.0120, 0.1255, 0.1046,
                       0.1319, 0.0924, 0.0952, -0.0134, 0.0128, -0.1378,
                       -0.0459])
    m_gLb = np.array([-0.1315, 0.0373, -0.1890, -0.1696, -0.1447, 0.0525,
                       -0.0927, -0.0833, -0.1251, 0.0357, -0.0153, 0.1334,
                       -0.1589, 0.1759, -0.1561, 0.0341, 0.0113, 0.0070,
                       -0.0288])
    m_gLL = np.array([1.1973, 1.6246, 1.3061, 1.0817, 1.1507, 1.2378, 0.9709,
                      0.7855, 1.3469, 0.6585, 0.9418, 0.9913, 0.8693, 0.8080,
                      0.8277, 0.5755, 0.9311, 0.5322, 0.4228])
    m_Lab_metric = np.zeros((19,3,3))
    m_Lab_metric[:, 0, 0] = m_gLL
    m_Lab_metric[:, 1, 1] = m_gaa
    m_Lab_metric[:, 2, 2] = m_gbb
    m_Lab_metric[:, 0, 1] = m_gLa
    m_Lab_metric[:, 1, 0] = m_gLa
    m_Lab_metric[:, 0, 2] = m_gLb
    m_Lab_metric[:, 2, 0] = m_gLb
    m_Lab_metric[:, 1, 2] = m_gab
    m_Lab_metric[:, 2, 1] = m_gab
    return MetricData(spaceCIELAB, build_d_Melgosa(), m_Lab_metric)
    
def build_g_Melgosa_xyY():
    """
    Melgosa's xyY-fitted ellipsoids for the RIT-DuPont data.

    Copied verbatim from pdf of CRA paper. Uses the ellipsoids fitted
    in xyY and returns MetricData.
    """
    m_g11 = np.array([10.074, 5.604, 18.738,3.718, 5.013, 7.462, 1.229,
                      7.634, 11.805, 3.578, 5.359, 1.770, 0.368, 9.407,
                      0.624, 2.531, 11.222, 26.497, 3.762])
    m_g22 = np.array([3.762, 6.589, 14.619, 3.310, 13.314, 3.533, 5.774,
                      11.162, 7.268, 3.007, 1.769, 6.549, 2.348, 3.485,
                      2.091, 4.122, 2.623, 16.086, 4.597])
    m_g12 = np.array([-5.498, -3.518, -12.571, 0.219, -4.689, -3.946, -0.365,
                      -6.096, -5.562, -2.698, -0.902, -2.052, 0.040, -4.170,
                      -0.434, -1.074, -4.884, -18.122, -1.715])
    m_g13 = np.array([-1.607, 0.001, -0.776, -0.078, -0.037, 0.212, 0.683,
                      0.049, 0.560, -0.103, 0.223, 2.341, 0.538, -0.240, 1.825,
                      0.285, -2.174, -0.361, 0.064])
    m_g23 = np.array([-0.509, -0.346, 0.147, -0.489, -0.121, -0.065, -1.676,
                      -0.020, -0.521, 0.831, -0.291, -1.436, -0.936, 0.480,
                      -3.806, -0.058, 0.659, 0.343, 0.088])
    m_g33 = np.array([5.745, 2.426, 1.146, 1.111, 0.845, 2.311, 2.878, 0.287,
                      0.912, 21.381, 0.517, 9.775, 3.823, 0.687, 23.949, 0.564,
                      6.283, 0.160, 0.169])
    m_xyY_metric = np.zeros((19,3,3))
    m_xyY_metric[:, 0, 0] = m_g11
    m_xyY_metric[:, 1, 1] = m_g22
    m_xyY_metric[:, 2, 2] = m_g33
    m_xyY_metric[:, 0, 1] = m_g12
    m_xyY_metric[:, 1, 0] = m_g12
    m_xyY_metric[:, 0, 2] = m_g13
    m_xyY_metric[:, 2, 0] = m_g13
    m_xyY_metric[:, 1, 2] = m_g23
    m_xyY_metric[:, 2, 1] = m_g23
    m_xyY_metric = 1e4*m_xyY_metric
    return MetricData(spacexyY, build_d_Melgosa(), m_xyY_metric)

# TODO:
#
# Metric data sets, as needed (instances of MetricData):
#     BrownMacAdam
#     BFD
#     +++

#==============================================================================
# Colour metrics
#==============================================================================

def metric_Euclidean(space, data):
    """
    Compute the general Euclidean metric in the given colour space as MetricData.
    """
    g = space.empty_matrix(data.linear_XYZ)
    for i in range(np.shape(g)[0]):
        g[i] = np.eye(3)
    return MetricData(space, data, g)

def metric_DEab(data):
    """
    Compute the DEab metric as MetricData for the given data points.
    """
    return metric_Euclidean(spaceCIELAB, data)

def metric_DEuv(data):
    """
    Compute the DEuv metric as MetricData for the given data points.
    """
    return metric_Euclidean(spaceCIELUV, data)

# TODO:
#
# Functions (returning MetricData):
#     metric_DE00(data)
#     metric_Stiles(data)
#     metric_Helmholz(data)
#     metric_Schrodinger(data)
#     metric_Vos(data)
#     metric_SVF(data)
#     metric_CIECAM02
#     metric_DIN99
#     +++

#==============================================================================
# Auxiliary functions
#==============================================================================

def plot_ellipses(ellipses, axis=None, alpha=1,
                  facecolor=[.5, .5, .5], edgecolor=[0, 0, 0], fill=False):
    """
    Plot the list of ellipses on the given axis.
    """
    if axis == None:
        axis = plt.gca()
    for e in ellipses:
        axis.add_artist(e)
        e.set_clip_box(axis.bbox)
        e.set_alpha(alpha)
        e.set_facecolor(facecolor)
        e.set_edgecolor(edgecolor)
        e.set_fill(fill)

#==============================================================================
# Main, for testing only
#==============================================================================

if __name__ == '__main__':
    g = build_g_MacAdam()
    spaceTmp = TransformCartesian(spaceCIELCh)
    d = g.points.get_linear(spaceCIELAB)
    plt.clf()
    plt.plot(d[:,1], d[:,2], '.')
    plot_ellipses(g.get_ellipses(spaceCIELAB, MetricData.plane_ab, scale=10))
    plt.axis('equal')
    plt.show()
