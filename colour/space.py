#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
space: Colour spaces, part of the colour package

Copyright (C) 2013 Ivar Farup

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
import data

#==============================================================================
# Colour space classes
#
# Throughout the code, the name ndata is used for numerical data (numpy
# arrays), and data is used for objects of the type Data.
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
        Data class) by inverting the inverse Jacobian.
        
        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to XYZ.
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

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians from XYZ.
        """
        ijac = self.jacobian_XYZ(data)
        for i in range(np.shape(ijac)[0]):
            ijac[i] = np.linalg.inv(ijac[i])
        return ijac
    
    def metrics_to_XYZ(self, points_data, metrics_ndata):
        """
        Convert metric data to the XYZ colour space.
        
        Parameters
        ----------
        points_data : Data
            The colour data points.
        metrics_ndata : ndarray
            Array of colour metric tensors in current colour space.
        
        Returns
        -------
        xyz_metrics : ndarray
            Array of colour metric tensors in XYZ.
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
        
        Parameters
        ----------
        points_data : Data
            The colour data points.
        metrics_ndata : ndarray
            Array of colour metric tensors in XYZ.
        
        Returns
        -------
        xyz_metrics : ndarray
            Array of colour metric tensors in the current colour space.
        """
        jacobian = self.inv_jacobian_XYZ(points_data)
        new_metric = np.zeros(np.shape(metrics_ndata))
        for i in range(np.shape(jacobian)[0]):
            new_metric[i] = np.dot(jacobian[i].T,
                                   np.dot(metrics_ndata[i], jacobian[i]))
        return new_metric

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
        Data class).

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to XYZ.
        """
        jac = self.empty_matrix(data.linear_XYZ)
        jac[:] = np.eye(3)
        return jac

    def inv_jacobian_XYZ(self, data):
        """
        Return the inverse Jacobian to XYZ, dXYZ^i/dx^j.

        The inverse Jacobian is calculated at the given data points
        (of the Data class).
        
        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians from XYZ.   
        """
        ijac = self.empty_matrix(data.linear_XYZ)
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
        Data class) by inverting the inverse Jacobian.

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
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
         
        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians from the base colour space.   
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

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to XYZ.

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

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians from XYZ.   
        """
        dXYZdbase = self.base.inv_jacobian_XYZ(data)
        dbasedx = self.inv_jacobian_base(data)
        ijac = self.empty_matrix(data.linear_XYZ)
        for i in range(np.shape(ijac)[0]):
            ijac[i] = np.dot(dXYZdbase[i], dbasedx[i])
        return ijac

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
        xyz[:,0] = ndata[:,0]*ndata[:,2]/ndata[:,1]
        xyz[:,1] = ndata[:,2]
        xyz[:,2] = (1 - ndata[:,0] - ndata[:,1]) * ndata[:,2] / ndata[:,1]
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
        xyY[:,0] = xyz[:,0] / xyz_sum # x
        xyY[:,1] = xyz[:,1] / xyz_sum # y
        xyY[:,2] = xyz[:,1]           # Y
        return xyY
    
    def jacobian_base(self, data):
        """
        Return the Jacobian to XYZ, dxyY^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
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

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
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
        xyz[:,0] = xr * self.white_point[0]
        xyz[:,1] = yr * self.white_point[1]
        xyz[:,2] = zr * self.white_point[2]
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
        Data class).

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
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
        upr = 4 * self.white_point[0] / (self.white_point[0] + 15*self.white_point[1] + 3*self.white_point[2])
        vpr = 9 * self.white_point[1] / (self.white_point[0] + 15*self.white_point[1] + 3*self.white_point[2])
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

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        xyz_ = data.get_linear(xyz)
        luv = data.get_linear(cieluv)
        df = self.dfdx(xyz_)
        jac = self.empty_matrix(xyz_)
        # dL/dY:
        jac[:, 0, 1] = 116 * df[:, 1] / self.white_point[1]
        # du/dX:
        jac[:, 1, 0] = 13 * luv[:,0] * \
            (60 * xyz_[:,1] + 12 * xyz_[:,2]) / \
            (xyz_[:,0] + 15 * xyz_[:,1] + 3 * xyz_[:,2]) ** 2
        # du/dY:
        jac[:, 1, 1] = 13 * luv[:,0] * \
            -60 * xyz_[:,0] / \
            (xyz_[:,0] + 15 * xyz_[:,1] + 3 * xyz_[:,2]) ** 2 + \
            13 * jac[:, 0, 1] * (
                4 * xyz_[:,0] / (xyz_[:,0] + 15 * xyz_[:,1] + 3 * xyz_[:,2]) -
                4 * self.white_point[0] / \
                (self.white_point[0] + 15 * self.white_point[1] + 3 * self.white_point[2]))
        # du/dZ:
        jac[:, 1, 2] = 13 * luv[:,0] * \
            -12 * xyz_[:,0] / \
            (xyz_[:,0] + 15 * xyz_[:,1] + 3 * xyz_[:,2]) ** 2
        # dv/dX:
        jac[:, 2, 0] = 13 * luv[:,0] * \
            -9 * xyz_[:,1] / \
            (xyz_[:,0] + 15 * xyz_[:,1] + 3 * xyz_[:,2]) ** 2
        # dv/dY:
        jac[:, 2, 1] = 13 * luv[:,0] * \
            (9 * xyz_[:,0] + 27 * xyz_[:,2]) / \
            (xyz_[:,0] + 15 * xyz_[:,1] + 3 * xyz_[:,2]) ** 2 + \
            13 * jac[:, 0, 1] * (
                9 * xyz_[:,1] / (xyz_[:,0] + 15 * xyz_[:,1] + 3 * xyz_[:,2]) -
                9 * self.white_point[1] / \
                (self.white_point[0] + 15 * self.white_point[1] + 3 * self.white_point[2]))
        # dv/dZ:
        jac[:, 2, 2] = 13 * luv[:,0] * \
            -27 * xyz_[:,1] / \
            (xyz_[:,0] + 15 * xyz_[:,1] + 3 * xyz_[:,2]) ** 2
        return jac

class TransformCIEDE00(Transform):
    """
    The CIELAB to CIEDE00 L'a'b' colour space transform.
    """

    def __init__(self, base):
        """
        Construct instance by setting base space and white point.
        
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
        print 'No conversion of CIEDE00 Lab to CIELAB implemented (yet).'
    
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
        Cab = np.sqrt(lab[:,1]**2 + lab[:,2]**2)
        G = .5 * (1 - np.sqrt(Cab**7 / (Cab**7 + 25**7)))
        labp[:,1] = lab[:,1] * (1 + G)
        return labp
    
    def jacobian_base(self, data):
        """
        Return the Jacobian to CIELAB (base), dCIEDE00^i/dCIELAB^j.

        The Jacobian is calculated at the given data points (of the
        Data class).

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        lab = data.get_linear(cielab)
        lch = data.get_linear(cielch)
        a = lab[:,1]
        b = lab[:,2]
        C = lch[:,1]
        G = .5 * (1 - np.sqrt(C**7 / (C**7 + 25**7)))
        jac = self.empty_matrix(lab)
        jac[:,0,0] = 1 # dLp/dL
        jac[:,2,2] = 1 # dbp/db
        jac[:,1,1] = 1 + G - a**2 / C * (7 * 25**7 * C**(5/2.) / (4 * (C**7 + 25**7)**(3/2.))) # dap/da
        jac[C == 0, 1, 1] = 1
        jac[:,1,2] = - a * b / C * (7 * 25**7 * C**(5/2.) / (4 * (C**7 + 25**7)**(3/2.)))
        jac[C == 0, 1, 2] = 0
        return jac
    
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
        Data class).

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        xyzdata = data.get_linear(xyz)
        jac = self.empty_matrix(xyzdata)
        jac[:] = self.M
        return jac
    
    def inv_jacobian_base(self, data):
        """
        Return the Jacobian from XYZ (base), dXYZ^i/dlinear^j.

        The Jacobian is calculated at the given data points (of the
        Data class).

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        xyzdata = data.get_linear(xyz)
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
        Data class).

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        basedata = data.get_linear(self.base)
        jac = self.empty_matrix(basedata)
        for i in range(np.shape(basedata)[0]):
            jac[i, 0, 0] = self.gamma * np.abs(basedata[i, 0])**(self.gamma - 1)
            jac[i, 1, 1] = self.gamma * np.abs(basedata[i, 1])**(self.gamma - 1)
            jac[i, 2, 2] = self.gamma * np.abs(basedata[i, 2])**(self.gamma - 1)
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
        Lab[:,0] = ndata[:,0]
        C = ndata[:,1]
        h = ndata[:,2]
        Lab[:,1] = C * np.cos(h)
        Lab[:,2] = C * np.sin(h)
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

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
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
        Lab[:,0] = ndata[:,0]
        C = ndata[:,1]
        h = ndata[:,2]
        Lab[:,1] = C * np.cos(h)
        Lab[:,2] = C * np.sin(h)
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
        LCh[:,0] = ndata[:,0]
        x = ndata[:,1]
        y = ndata[:,2]
        LCh[:,1] = np.sqrt(x**2 + y**2)
        LCh[:,2] = np.arctan2(y, x)
        return LCh
        
    def jacobian_base(self, data):
        """
        Return the Jacobian from CIELCh (base), dCIELAB^i/dCIELCH^j.
        
        The Jacobian is calculated at the given data points (of the
        Data class).

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
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
        self.space_ABC = TransformLinear(self.base, np.array([[0.6597, 0.4492, -0.1089],
                                                              [-0.3053, 1.2126, 0.0927],
                                                              [-0.0374, 0.4795, 0.5579]]))
        self.space_xyY = TransformxyY(self.base)
        
    def to_base(self, ndata):
        """
        Convert from LGJOSA to XYZ (base). Not implemented yet.

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space
        
        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        print 'No conversion of LGJOSA to XYZ implemented (yet).'
        
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
        dat = data.Data(self.base, ndata)
        abc = dat.get_linear(self.space_ABC)
        A = abc[:,0]
        B = abc[:,1]
        C = abc[:,2]
        xyY = dat.get_linear(self.space_xyY)
        x = xyY[:,0] 
        y = xyY[:,1] 
        Y = xyY[:,2]
        Y_0 = 100 * Y * (4.4934 * x**2 + 4.3034 * y**2 - 4.2760 * x * y -
                         1.3744 * x - 2.5643 * y + 1.8103)
        L_osa = (5.9 * ((Y_0**(1/3.) - (2/3.)) + 
                        0.0042 * np.sign(Y_0 - 30) * np.abs(Y_0 - 30)**(1/3.)) - 14.4) / np.sqrt(2)   
        G = -2 * (0.764 * L_osa + 9.2521) * (0.9482 * (np.log(A) - np.log(0.9366 * B)) -
                                             0.3175 * (np.log(B) - np.log(0.9807 * C)))
        J = 2 * (0.5735 * L_osa + 7.0892) * (0.1792 * (np.log(A) - np.log(0.9366 * B)) +
                                             0.9237 * (np.log(B) - np.log(0.9807 * C)))
        col = np.zeros(np.shape(ndata))
        col[:,0] = L_osa
        col[:,1] = G
        col[:,2] = J
        return col
    
    def jacobian_base(self, data):
        """
        Return the Jacobian from XYZ (base), dLGJOSA^i/dXYZ^j.
        
        The Jacobian is calculated at the given data points (of the
        Data class). Like the colour space, a terrible mess...

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        ABC = data.get_linear(self.space_ABC)
        xyY = data.get_linear(self.space_xyY)
        x = xyY[:,0]
        y = xyY[:,1]
        Y = xyY[:,2]
        A = ABC[:,0]
        B = ABC[:,1]
        C = ABC[:,2]
        dxyY_dXYZ = self.space_xyY.jacobian_base(data)
        dx_dX = dxyY_dXYZ[:,0,0]
        dx_dY = dxyY_dXYZ[:,0,1]
        dx_dZ = dxyY_dXYZ[:,0,2]
        dy_dX = dxyY_dXYZ[:,1,0]
        dy_dY = dxyY_dXYZ[:,1,1]
        dy_dZ = dxyY_dXYZ[:,1,2]
        dY_dX = dxyY_dXYZ[:,2,0]
        dY_dY = dxyY_dXYZ[:,2,1]
        dY_dZ = dxyY_dXYZ[:,2,2]
        dABC_dXYZ = self.space_ABC.jacobian_base(data)
        dA_dX = dABC_dXYZ[:,0,0]
        dA_dY = dABC_dXYZ[:,0,1]
        dA_dZ = dABC_dXYZ[:,0,2]
        dB_dX = dABC_dXYZ[:,1,0]
        dB_dY = dABC_dXYZ[:,1,1]
        dB_dZ = dABC_dXYZ[:,1,2]
        dC_dX = dABC_dXYZ[:,2,0]
        dC_dY = dABC_dXYZ[:,2,1]
        dC_dZ = dABC_dXYZ[:,2,2]
        Y_0 = 100 * Y * (4.4934 * x**2 + 4.3034 * y**2 - 4.2760 * x * y -
                         1.3744 * x - 2.5643 * y + 1.8103)
        L = (5.9 * ((Y_0**(1/3.) - (2/3.)) + \
                    0.0042 * np.sign(Y_0 - 30) * np.abs(Y_0 - 30)**(1/3.)) - 14.4) / np.sqrt(2)   
        dL_dY0 = 5.9 * (Y_0**(-2./3) + 0.042 * np.sign(Y_0 - 30) * np.abs(Y_0 - 30)**(-2./3) / 3) / np.sqrt(2)
        dY0_dx = 100 * Y * (4.4934 * 2 * x - 4.2760 * y - 1.3744)
        dY0_dy = 100 * Y * (4.3034 * 2 * y - 4.2760 * x - 2.5643)
        dY0_dY = 100 * (4.4934 * x**2 + 4.3034 * y**2 - 4.2760 * x * y -
                        1.3744 * x - 2.5643 * y + 1.8103)
        dL_dX = dL_dY0 * (dY0_dx * dx_dX + dY0_dy * dy_dX + dY0_dY * dY_dX)
        dL_dY = dL_dY0 * (dY0_dx * dx_dY + dY0_dy * dy_dY + dY0_dY * dY_dY)
        dL_dZ = dL_dY0 * (dY0_dx * dx_dZ + dY0_dy * dy_dZ + dY0_dY * dY_dZ)
        TG = 0.9482 * (np.log(A) - np.log(0.9366 * B)) - 0.3175 * (np.log(B) - np.log(0.9807 * C))
        TJ = 0.1792 * (np.log(A) - np.log(0.9366 * B)) + 0.9237 * (np.log(B) - np.log(0.9807 * C))
        SG = - 2 * (0.764 * L + 9.2521) 
        SJ =  2 * (0.5735 * L + 7.0892) 
        dG_dL = - 2 * 0.764 * TG
        dJ_dL = 2 * 0.57354 * TJ
        dG_dA = SG * 0.9482 / A
        dG_dB = SG * (-0.9482 - 0.3175) / B
        dG_dC = SG * 0.3175 / C
        dJ_dA = SJ * 0.1792 / A
        dJ_dB = SJ * (-0.1792 + 0.9837) / B 
        dJ_dC = SJ * (-0.9837) / C
        dG_dX = dG_dL * dL_dX + dG_dA * dA_dX + dG_dB * dB_dX + dG_dC * dC_dX
        dG_dY = dG_dL * dL_dY + dG_dA * dA_dY + dG_dB * dB_dY + dG_dC * dC_dY
        dG_dZ = dG_dL * dL_dZ + dG_dA * dA_dZ + dG_dB * dB_dZ + dG_dC * dC_dZ
        dJ_dX = dJ_dL * dL_dX + dJ_dA * dA_dX + dJ_dB * dB_dX + dJ_dC * dC_dX
        dJ_dY = dJ_dL * dL_dY + dJ_dA * dA_dY + dJ_dB * dB_dY + dJ_dC * dC_dY
        dJ_dZ = dJ_dL * dL_dZ + dJ_dA * dA_dZ + dJ_dB * dB_dZ + dJ_dC * dC_dZ
        jac = self.empty_matrix(ABC)
        jac[:,0,0] = dL_dX
        jac[:,0,1] = dL_dY
        jac[:,0,2] = dL_dZ
        jac[:,1,0] = dG_dX
        jac[:,1,1] = dG_dY
        jac[:,1,2] = dG_dZ
        jac[:,2,0] = dJ_dX
        jac[:,2,1] = dJ_dY
        jac[:,2,2] = dJ_dZ
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
        Convert from LGJE to LGJOSA (base). Not implemented yet!

        Parameters
        ----------
        ndata : ndarray
            Colour data in the current colour space
        
        Returns
        -------
        col : ndarray
            Colour data in the base colour space
        """
        print 'No conversion of LGJE to LGJOSA implemented (yet).'
        
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
        L = ndata[:,0]
        G = ndata[:,1]
        J = ndata[:,2]
        C = np.sqrt(G**2 + J**2)
        L_E = np.log(1 + 10 * L * self.bL / self.aL) / self.bL
        C_E = np.log(1 + 10 * C * self.bc / self.ac) / self.bc
        scale = np.zeros(np.shape(C))
        scale[C == 0] = 1
        scale[C != 0] = C_E[C != 0] / C[C != 0]
        G_E = - scale * G
        J_E = - scale * J
        col = ndata.copy()
        col[:,0] = L_E
        col[:,1] = G_E
        col[:,2] = J_E
        return col
    
    def jacobian_base(self, data):
        """
        Return the Jacobian from LGJOSA (base), dLGJE^i/dLGJOSA^j.
        
        The Jacobian is calculated at the given data points (of the
        Data class).

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        lgj = data.get(lgj_osa)
        L = lgj[:,0]
        G = lgj[:,1]
        J = lgj[:,2]
        C = np.sqrt(G**2 + J**2)
        lgj_e = data.get(self)
        C_E = np.sqrt(lgj_e[:,1]**2 + lgj_e[:,2]**2)
        dLE_dL = 10 / (self.aL + 10 * self.bL * L)
        dCE_dC = 10 / (self.ac + 10 * self.bc * C)
        dCEC_dC = (dCE_dC * C - C_E) / C**2
        dC_dG = G / C
        dC_dJ = J / C
        dCEC_dG = dCEC_dC * dC_dG
        dCEC_dJ = dCEC_dC * dC_dJ
        dGE_dG = - C_E / C - G * dCEC_dG
        dGE_dJ = - G * dCEC_dJ
        dJE_dG = - J * dCEC_dG
        dJE_dJ = - C_E / C - J * dCEC_dJ
        jac = self.empty_matrix(lgj)
        jac[:,0,0] = dLE_dL
        jac[:,1,1] = dGE_dG
        jac[:,1,2] = dGE_dJ
        jac[:,2,1] = dJE_dG
        jac[:,2,2] = dJE_dJ
        return jac

class TransformPoincareDisk(Transform):
    """
    Transform from Cartesian coordinates to Poincare disk coordinates.
    
    The coordinate transform only changes the radius (chroma, typically),
    and does so in a way that preserves the radial distance with respect to
    the Euclidean metric and the Poincare disk metric in the source and
    target spaces, respectively.
    """

    def __init__(self, base):
        """
        Construct instance, setting base space.
        
        Parameters
        ----------
        base : Space
            The base colour space.
        """
        super(TransformPoincareDisk, self).__init__(base)
        
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
        Lab[:,1:] = 0
        x = ndata[:,1]
        y = ndata[:,2]
        r = np.sqrt(x**2 + y**2)
        for i in range(np.shape(Lab)[0]):
            if r[i] > 0:
                Lab[i,1:] = ndata[i,1:] * (2 * np.arctanh(r[i])) / r[i]
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
        Lxy[:,1:] = 0
        a = ndata[:,1]
        b = ndata[:,2]
        C = np.sqrt(a**2 + b**2)
        for i in range(np.shape(Lxy)[0]):
            if C[i] > 0:
                Lxy[i,1:] = ndata[i,1:] * np.tanh(C[i] / 2) / C[i]
        return Lxy
    
    def jacobian_base(self, data):
        """
        Return the Jacobian from CIELAB (base), dLxy^i/dCIELAB^j.
        
        The Jacobian is calculated at the given data points (of the
        Data class).

        Parameters
        ----------
        data : Data
            Colour data points for the jacobians to be computed.
        
        Returns
        -------
        jacobian : ndarray
            The list of Jacobians to the base colour space.
        """
        Lab = data.get_linear(self.base)
        a = Lab[:,1]
        b = Lab[:,2]
        C = np.sqrt(a**2 + b**2)
        tanhC2 = np.tanh(C / 2.)
        tanhC2C = tanhC2 / C
        dCda = a / C
        dCdb = b / C
        dtanhdC = ((C / 2 * (1 - np.tanh(C / 2))**2) - np.tanh(C / 2)) / C**2
        jac = self.empty_matrix(Lab)
        for i in range(np.shape(jac)[0]):
            jac[i, 0, 0] = 1 # dL/dL
            if C[i] == 0:
                jac[i, 1, 1] = .5 # dx/da
                jac[i, 2, 2] = .5 # dy/db
            else:
                jac[i, 1, 1] = tanhC2C[i] + a[i] * dtanhdC[i] * dCda[i] # dx/da
                jac[i, 1, 2] = a[i] * dtanhdC[i] * dCdb[i] # dx/db
                jac[i, 2, 1] = b[i] * dtanhdC[i] * dCda[i] # dy/da
                jac[i, 2, 2] = tanhC2C[i] + b[i] * dtanhdC[i] * dCdb[i] # dy/db
        return jac

#==============================================================================
# Colour space instances
#==============================================================================

xyz = XYZ()
xyY = TransformxyY(xyz)
cielab = TransformCIELAB(xyz)
cielch= TransformPolar(cielab)
cieluv = TransformCIELUV(xyz)
ciede00lab = TransformCIEDE00(cielab)
ciede00lch = TransformPolar(ciede00lab)
ipt = TransformLinear(TransformGamma(TransformLinear(xyz,
                np.array([[.4002, .7075, -.0807],
                          [-.228, 1.15, .0612],
                          [0, 0, .9184]])),
                .43),
                np.array([[.4, .4, .2],
                          [4.455, -4.850, .3960],
                          [.8056, .3572, -1.1628]]))
lgj_osa = TransformLGJOSA(xyz)
lgj_e = TransformLGJE(lgj_osa)
# First attemt at Euclidean for Poincare transform:
ui = TransformLinear(TransformGamma(TransformLinear(xyz,
                np.array([[0.1551646, 0.5430763, -0.0370161],
                          [-0.1551646, 0.4569237, 0.0296946],
                          [0, 0, 0.0073215]])),
                .43),
                np.array([[1.1032e+00, 5.0900e-01, 5.0840e-03],
                          [2.2822e+00, -4.2580e+00, 6.2844e+00],
                          [9.6110e+00, -1.2199e+01, -2.3843e+00]]))

# For testing only:
_test_space_cartesian = TransformCartesian(cieluv)
_test_space_poincare_disk = TransformPoincareDisk(cielab)
_test_space_gamma = TransformGamma(xyz, .43)

#==============================================================================
# Test module
#==============================================================================

def test():
    """
    Test entire module, and print report.
    """
    col = np.array([[1e-10, 1e-10, 1e-10],
                    [.95, 1., 1.08],
                    [.5, .5, .5]])
    test_spaces = [xyz, xyY, cielab, cieluv, cielch, ipt,
                  _test_space_cartesian, _test_space_poincare_disk,
                  _test_space_gamma]
    print "Colour transformations:"
    for sp in test_spaces:
        c2 = sp.to_XYZ(sp.from_XYZ(col))
        err = np.max(np.abs(col - c2))
        if err > 1e-14:
            print sp, ": ", err, " !!!"
        else:
            print sp, ": OK"
    print "\nJacobians:"
    col_data = data.Data(xyz, col)
    test_spaces = [xyz, xyY, cielab, cieluv, cielch, ipt, ciede00lab,
                  _test_space_cartesian, _test_space_poincare_disk,
                  _test_space_gamma]
    for sp in test_spaces:
        jac1 = sp.jacobian_XYZ(col_data)
        jac2 = sp.inv_jacobian_XYZ(col_data)
        prod = np.zeros(np.shape(jac1))
        for i in range(np.shape(jac1)[0]):
            prod[i] = np.dot(jac1[i], jac2[i])
            prod[i] = np.abs(prod[i] - np.eye(3))
        err = np.max(prod)
        if err > 1e-6:
            print sp, ": ", err, " !!!"
        else:
            print sp, ": OK"
        