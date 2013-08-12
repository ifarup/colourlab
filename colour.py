#!/usr/bin/env python
"""
Colour spaces, colour metrics and colour data

Copyright (C) 2011-2012 Ivar Farup

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

import tc
import pylab as pl
from matplotlib.patches import Ellipse

#==============================================================================
# White points (numeric data in CIEXYZ)
#==============================================================================

white_A = pl.array([1.0985, 1., 0.35585])
white_B = pl.array([.990720, 1., .852230])
white_C = pl.array([.980740, 1., .82320])
white_D50 = pl.array([.964220, 1., .82510])
white_D55 = pl.array([.956820, 1., .921490])
white_D65 = pl.array([.950470, 1., 1.088830])
white_D75 = pl.array([.949720, 1., 1.226380])
white_E = pl.array([1., 1., 1.])
white_F2 = pl.array([.991860, 1., .673930])
white_F7 = pl.array([.950410, 1., 1.087470])
white_F11 = pl.array([1.009620, 1., .643500])

#==============================================================================
# Colour spaces
#
# Throughout the code, the name ndata is used for numerical data (numpy
# arrays), and data is used for objects of the type Data, and mdata for
# objects of the type MetricData.
#==============================================================================

class Space:
    """
    Base class for the colour space classes.

    Contains auxiliary methods for converting data to the desired
    shapes, as well as for combining Jacobi matrices.
    """
    
    @classmethod
    def preshape(cls, ndata):
        """
        Shape the data so that is becomes an PxC matrix or C vector.
        
        The data should be of the shape Mx...xNxC, where C is the
        number of colour channels. Returns the shaped data as a PxC
        matrix where P=Mx...xN, as well as the shape of the input
        data. Get back to original shape by reshape(data, shape).
        """
        sh = pl.shape(ndata)
        sh_array = pl.array(sh)
        P_data = pl.prod(sh_array[:len(sh) - 1])
        C_data = sh[len(sh) - 1]
        return pl.reshape(ndata, (P_data, C_data)), sh

    @classmethod
    def preshape_matrix(cls, ndata):
        """
        Shape the matrix data so the it becomes a PxC1xC1 matrix.

        The data should be on the shape Mx...xNxC1xC2, where Cx is the
        number of colour channels. Returns the shaped data as a
        PxC1xC2 matrix where P=Mx...xN, as well as the shape of the
        input data. Get back original shape by reshape(data, shape).
        """
        sh = pl.shape(ndata)
        sh_array = pl.array(sh)
        P_data = pl.prod(sh_array[:len(sh) - 2])
        C1_data = sh[len(sh) - 2]
        C2_data = sh[len(sh) - 1]
        return pl.reshape(ndata, (P_data, C1_data, C2_data)), sh
    
    @classmethod
    def empty_matrix(cls, ndata):
        """
        Make empty Jabobian with appropriate dimensions.
        
        Similar to preshape, but gives allocates an extra dimension
        for the Jacobian matrix. Get back to good shape by
        reshape(data, shape).
        """
        ndata, shape = cls.preshape(ndata)
        sh = pl.array(pl.shape(ndata))
        C = sh[1]
        jacobian = pl.zeros((sh[0], C, C))
        shape = list(shape)
        shape.append(C)
        shape = tuple(shape)
        return jacobian, shape

    @classmethod
    def inv_jacobian_XYZ(cls, data):
        """
        Return the inverse Jacobian to the XYZ colour space.

        The inverse Jacobian is calculated at the given data points
        (of the Data class). Default method that works simply by
        inverting the matrix given by the cls.jacobian_XYZ(data).
        """
        jac, sh = cls.preshape_matrix(cls.jacobian_XYZ(data))
        for i in range(pl.shape(jac)[0]):
            jac[i] = pl.inv(jac[i])
        return pl.reshape(jac, sh)

    @classmethod
    def jacobian(cls, space, data):
        """
        Compute the Jacobian matrix between two colour spaces.
        
        Compute the Jacobian matrix for the transformation from
        current (i.e., class) colour space to the given colour space
        at the points specified by the colour data, In other
        words, it calculates d(cls^i)/d(space^j)(data).
        """
        # TODO: works only when both colour spaces have the same
        # dimensions due to sh1 vs. sh2. Not always the case, e.g., in
        # DEE and CIEDE2000!
        #
        # Is this function really needed? Ever used?
        jac1, sh1 = cls.preshape_matrix(cls.jacobian_XYZ(data))
        jac2, sh2 = space.preshape_matrix(space.inv_jacobian_XYZ(data))
        jac = pl.zeros(pl.shape(jac1))
        for i in range(pl.shape(jac)[0]):
            jac[i] = pl.dot(jac1[i], jac2[i])
        return pl.reshape(jac, sh1)

    @classmethod
    def metrics_to_XYZ(cls, points_data, metrics_ndata):
        """
        Convert metric data to the XYZ colour space.
        """
        # TODO: works only when both colour spaces have the same
        # dimensions due to sh_jac vs. sh_met. Not always the case,
        # e.g., in DEE and CIEDE2000! Or?
        jacobian, sh_jacobian = \
            cls.preshape_matrix(cls.jacobian_XYZ(points_data))
        metric, sh_metric = cls.preshape_matrix(metrics_ndata)
        new_metric = pl.zeros(pl.shape(metric))
        for i in range(pl.shape(jacobian)[0]):
            new_metric[i] = pl.dot(jacobian[i].T,
                                   pl.dot(metric[i], jacobian[i]))
        return pl.reshape(new_metric, sh_metric)

    @classmethod
    def metrics_from_XYZ(cls, points_data, metrics_ndata):
        """
        Convert metric data from the XYZ colour space.
        """
        # TODO: works only when both colour spaces have the same
        # dimensions due to sh_jac vs. sh_met. Not always the case,
        # e.g., in DEE and CIEDE2000!
        jacobian, sh_jacobian = \
            cls.preshape_matrix(cls.inv_jacobian_XYZ(points_data))
        metric, sh_metric = cls.preshape_matrix(metrics_ndata)
        new_metric = pl.zeros(pl.shape(metric))
        for i in range(pl.shape(jacobian)[0]):
            new_metric[i] = pl.dot(jacobian[i].T,
                                   pl.dot(metric[i], jacobian[i]))
        return pl.reshape(new_metric, sh_metric)

class SpaceLinearTransform(Space):
    """
    A general colour space defined by a linear transformation.
    
    The resulting space itself is not necessarily linear (understood as a
    linear transformation from, e.g., XYZ), but is instead defined by a
    linear transformation from any colour space. This class must be
    derived in order to define the parameters (matrix) of the linear
    transformation and the colour space from which to transform.
    
    This colour space is mainly thought to be used as a component for tool
    other colour spaces. Both simple linear ones such as LMS or linear RGB,
    but also to create components of more complex colour spaces such as
    OSA-UCS.
    
    The class variables base_space, trans_matrix and inv_trans_matrix
    must be set in derived classes.
    """
    
    @classmethod
    def from_XYZ(cls, ndata):
        """
        Convert from XYZ to current space.
        """
        base_data = Data(SpaceXYZ, ndata).get(cls.base_space)
        d, s = cls.preshape(base_data)
        lin = pl.dot(cls.trans_matrix, d.T).T
        return pl.reshape(lin, s)
    
    @classmethod
    def to_XYZ(cls, ndata):
        """
        Convert from current space to XYZ.
        """
        d, s = cls.preshape(ndata)
        base_data = pl.dot(cls.inv_trans_matrix, d.T).T
        xyz = cls.base_space.to_XYZ(base_data)
        return pl.reshape(xyz, s)
    
    @classmethod
    def jacobian_XYZ(cls, data):
        """
        Return the Jacobian to XYZ, dLinear^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        xyzdata = data.get(SpaceXYZ)
        jac, s = cls.empty_matrix(xyzdata)
        jac[:] = cls.trans_matrix
        return pl.reshape(jac, s)

class SpaceXYZ(Space):
    """
    The XYZ colour space.

    Assumes that the CIE 1931 XYZ colour matching functions are
    used. The white point is D65. Serves a special role in the code in that
    it serves as a common reference point.
    """
    
    @classmethod
    def to_XYZ(cls, ndata):
        """
        Convert from current colour space to XYZ.
        """
        return ndata.copy()      # identity transform

    @classmethod
    def from_XYZ(cls, ndata):
        """
        Convert from XYZ to current colour space.
        """
        return ndata.copy()      # identity transform

    @classmethod
    def jacobian_XYZ(cls, data):
        """
        Return the Jacobian to XYZ, dx^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        d, s = cls.empty_matrix(data.get(cls))
        d[:] = pl.eye(3)
        return pl.reshape(d, s)

    @classmethod
    def inv_jacobian_XYZ(cls, data):
        """
        Return the inverse Jacobian to XYZ, dXYZ^i/dx^j.

        The inverse Jacobian is calculated at the given data points
        (of the Data class).
        """
        d, s = cls.empty_matrix(data.get(cls))
        d[:] = pl.eye(3)
        return pl.reshape(d, s)

class SpacexyY(Space):
    """
    The xyY chromaticity colour space.

    Assumes that the CIE 1931 XYZ colour matching functions are used.
    """
    
    @classmethod
    def to_XYZ(cls, ndata):
        """
        Convert from xyY to XYZ.
        """
        d, s = cls.preshape(ndata)
        xyz = pl.zeros(pl.shape(d))
        xyz[:,0] = d[:,0]*d[:,2]/d[:,1]                    # X
        xyz[:,1] = d[:,2]                                  # Y
        xyz[:,2] = (1 - d[:,0] - d[:,1]) * d[:,2] / d[:,1] # Z
        return pl.reshape(xyz, s)

    @classmethod
    def from_XYZ(cls, ndata):
        """
        Convert from XYZ to xyY.
        """
        xyz, s = cls.preshape(ndata)
        xyY = pl.zeros(pl.shape(xyz))
        xyz_sum = pl.sum(xyz, axis=1)
        xyY[:,0] = xyz[:,0] / xyz_sum # x
        xyY[:,1] = xyz[:,1] / xyz_sum # y
        xyY[:,2] = xyz[:,1]           # Y
        return pl.reshape(xyY, s)

    @classmethod
    def jacobian_XYZ(cls, data):
        """
        Return the Jacobian to XYZ, dxyY^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        xyzdata, sh = cls.preshape(data.get(SpaceXYZ))
        jac, s = cls.empty_matrix(data.get(SpaceXYZ))
        for i in range(pl.shape(jac)[0]):
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
        return pl.reshape(jac, s)

    @classmethod
    def inv_jacobian_XYZ(cls, data):
        """
        Return the Jacobian from XYZ, dXYZ^i/dxyY^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        xyYdata, sh = cls.preshape(data.get(SpacexyY))
        jac, s = cls.empty_matrix(data.get(SpacexyY))
        for i in range(pl.shape(jac)[0]):
            jac[i,0,0] = xyYdata[i,2] / xyYdata[i,1]
            jac[i,0,1] = - xyYdata[i,0] * xyYdata[i,2] / xyYdata[i,1] ** 2
            jac[i,0,2] = xyYdata[i,0] / xyYdata[i,1]
            jac[i,1,2] = 1
            jac[i,2,0] = - xyYdata[i,2] / xyYdata[i,1]
            jac[i,2,1] = xyYdata[i,2] * (xyYdata[i,0] - 1) / \
                xyYdata[i,1] ** 2
            jac[i,2,2] = (1 - xyYdata[i,0] - xyYdata[i,1]) / xyYdata[i,1]
        return pl.reshape(jac, s)

class SpaceCIELAB(Space):
    """
    The CIELAB colour space.

    Assumes that the CIE 1931 XYZ colour matching functions are
    used. The white point in the conversion is D65, not D50!
    """
    kappa = 24389. / 27.        # standard: 903.3
    epsilon = 216. / 24389.     # standard: 0.008856

    @classmethod
    def f(cls, ndata):
        """
        Auxiliary function for the conversion.
        """
        fx = (cls.kappa * ndata + 16.) / 116.
        fx[ndata > cls.epsilon] = ndata[ndata > cls.epsilon] ** (1. / 3)
        return fx

    @classmethod
    def dfdx(cls, ndata):
        """
        Auxiliary function for the Jacobian.

        Returns the derivative of the function f above. Works for arrays.
        """
        df = cls.kappa / 116. * pl.ones(pl.shape(ndata))
        df[ndata > cls.epsilon] = \
            (ndata[ndata > cls.epsilon] ** (-2. /3)) / 3
        return df

    @classmethod
    def to_XYZ(cls, ndata):
        """
        Convert from CIELAB to XYZ.
        """
        d, s = cls.preshape(ndata)
        fy = (d[:, 0] + 16.) / 116.
        fx = d[:, 1] / 500. + fy
        fz = fy - d[:, 2] / 200.
        xr = fx ** 3
        xr[xr <= cls.epsilon] = ((116 * fx[xr <= cls.epsilon] - 16) /
                                 cls.kappa)
        yr = fy ** 3
        yr[d[:, 0] <= cls.kappa * cls.epsilon] = \
            d[d[:, 0] <= cls.kappa * cls.epsilon, 0] / cls.kappa
        zr = fz ** 3
        zr[zr <= cls.epsilon] = ((116 * fz[zr <= cls.epsilon] - 16) /
                                 cls.kappa)
        # Combine into matrix
        xyz = pl.zeros(pl.shape(d))
        xyz[:,0] = xr * white_D65[0]
        xyz[:,1] = yr * white_D65[1]
        xyz[:,2] = zr * white_D65[2]
        return pl.reshape(xyz, s)

    @classmethod
    def from_XYZ(cls, ndata):
        """
        Convert from XYZ to CIELAB.
        """
        d, s = cls.preshape(ndata)
        lab = pl.zeros(pl.shape(d))
        fx = cls.f(d[:, 0] / white_D65[0])
        fy = cls.f(d[:, 1] / white_D65[1])
        fz = cls.f(d[:, 2] / white_D65[2])
        lab[:, 0] = 116. * fy - 16.
        lab[:, 1] = 500. * (fx - fy)
        lab[:, 2] = 200. * (fy - fz)
        return pl.reshape(lab, s)

    @classmethod
    def jacobian_XYZ(cls, data):
        """
        Return the Jacobian to XYZ, dCIELAB^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        xyzdata = data.get(SpaceXYZ)
        d, s = cls.preshape(xyzdata)
        dr = d.copy()
        for i in range(3):
            dr[:, i] = dr[:, i] / white_D65[i]
        df = cls.dfdx(dr)
        jac, s = cls.empty_matrix(xyzdata)
        jac[:, 0, 1] = 116 * df[:, 1] / white_D65[1]  # dL/dY
        jac[:, 1, 0] = 500 * df[:, 0] / white_D65[0]  # da/dX
        jac[:, 1, 1] = -500 * df[:, 1] / white_D65[1] # da/dY
        jac[:, 2, 1] = 200 * df[:, 1] / white_D65[1]  # db/dY
        jac[:, 2, 2] = -200 * df[:, 2] / white_D65[2] # db/dZ
        return pl.reshape(jac, s)

class SpaceCIELUV(Space):
    """
    The CIELUV colour space.

    Assumes that the CIE 1931 XYZ colour matching functions are
    used. The white point in the conversion is D65, not D50!
    """
    kappa = 24389. / 27.        # standard: 903.3
    epsilon = 216. / 24389.     # standard: 0.008856

    @classmethod
    def f(cls, ndata):
        """
        Auxiliary function for the conversion.
        """
        fx = (cls.kappa * ndata + 16.) / 116.
        fx[ndata > cls.epsilon] = ndata[ndata > cls.epsilon] ** (1. / 3)
        return fx

    @classmethod
    def dfdx(cls, ndata):
        """
        Auxiliary function for the Jacobian.

        Returns the derivative of the function f above. Works for arrays.
        """
        df = cls.kappa / 116. * pl.ones(pl.shape(ndata))
        df[ndata > cls.epsilon] = \
            (ndata[ndata > cls.epsilon] ** (-2. /3)) / 3
        return df

    @classmethod
    def to_XYZ(cls, ndata):
        """
        Convert from CIELUV to XYZ.
        """
        luv, s = cls.preshape(ndata)
        fy = (luv[:, 0] + 16.) / 116.
        y = fy ** 3
        y[luv[:, 0] <= cls.kappa * cls.epsilon] = \
            luv[luv[:, 0] <= cls.kappa * cls.epsilon, 0] / cls.kappa
        upr = 4 * white_D65[0] / (white_D65[0] + 15*white_D65[1] + 3*white_D65[2])
        vpr = 9 * white_D65[1] / (white_D65[0] + 15*white_D65[1] + 3*white_D65[2])
        a = (52*luv[:,0] / (luv[:,1] + 13*luv[:,0]*upr) - 1) / 3
        b = -5 * y
        c = -1/3.
        d = y * (39*luv[:,0] / (luv[:,2] + 13*luv[:,0]*vpr) - 5)
        x = (d - b) / (a - c)
        z = x * a + b
        # Combine into matrix
        xyz = pl.zeros(pl.shape(luv))
        xyz[:,0] = x
        xyz[:,1] = y
        xyz[:,2] = z
        return pl.reshape(xyz, s)

    @classmethod
    def from_XYZ(cls, ndata):
        """
        Convert from XYZ to CIELUV.
        """
        d, s = cls.preshape(ndata)
        luv = pl.zeros(pl.shape(d))
        fy = cls.f(d[:, 1] / white_D65[1])
        up = 4 * d[:,0] / (d[:,0] + 15*d[:,1] + 3*d[:,2])
        upr = 4 * white_D65[0] / (white_D65[0] + 15*white_D65[1] + 3*white_D65[2])
        vp = 9 * d[:,1] / (d[:,0] + 15*d[:,1] + 3*d[:,2])
        vpr = 9 * white_D65[1] / (white_D65[0] + 15*white_D65[1] + 3*white_D65[2])
        luv[:, 0] = 116. * fy - 16.
        luv[:, 1] = 13 * luv[:, 0] * (up - upr)
        luv[:, 2] = 13 * luv[:, 0] * (vp - vpr)
        return pl.reshape(luv, s)

    @classmethod
    def jacobian_XYZ(cls, data):
        """
        Return the Jacobian to XYZ, dCIELUV^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        xyzdata = data.get(SpaceXYZ)
        luvdata = data.get(SpaceCIELUV)
        xyz, s = cls.preshape(xyzdata)
        luv, s = cls.preshape(luvdata)
        df = cls.dfdx(xyz)
        jac, s = cls.empty_matrix(xyzdata)
        # dL/dY:
        jac[:, 0, 1] = 116 * df[:, 1] / white_D65[1]
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
                4 * white_D65[0] / \
                (white_D65[0] + 15 * white_D65[1] + 3 * white_D65[2]))
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
                9 * white_D65[1] / \
                (white_D65[0] + 15 * white_D65[1] + 3 * white_D65[2]))
        # dv/dZ:
        jac[:, 2, 2] = 13 * luv[:,0] * \
            -27 * xyz[:,1] / \
            (xyz[:,0] + 15 * xyz[:,1] + 3 * xyz[:,2]) ** 2
        return pl.reshape(jac, s)

        
class SpaceLGJ_ABC(SpaceLinearTransform):
    """
    The linear ABC space used in the OSA-UCS transformation.
    """
    
    base_space = SpaceXYZ
    trans_matrix = pl.array([[0.6597, 0.4492, -0.1089],
                             [-0.3053, 1.2126, 0.0927],
                             [-0.0374, 0.4795, 0.5579]])
    inv_trans_matrix = pl.inv(trans_matrix)

class SpaceLGJ(Space):
    """
    The OSA-UCS colour space.
    """

    @classmethod
    def from_XYZ(cls, ndata):
        """
        Convert from XYZ to LGJ.
        """
        xyz, s = cls.preshape(ndata)
        xyY = SpacexyY.from_XYZ(xyz)
        abc = SpaceLGJ_ABC.from_XYZ(xyz)
        lgj = pl.zeros(pl.shape(xyz))
        Y0 = 100*xyY[:,2]*(4.4934*xyY[:,0]**2 + 4.3034*xyY[:,1]**2 -
                           4.2760*xyY[:,0]*xyY[:,1] - 1.3744*xyY[:,0] -
                           2.5643*xyY[:,1] + 1.8103)
        lgj[:,0] = (5.9*((Y0**(1/3.) - 2/3.) +
                     0.042*pl.sign(Y0-30)*abs(Y0-30)**(1/3.)) -
                     14.4)/pl.sqrt(2)
        lgj[:,2] = (2*(0.5735*lgj[:,0] + 7.0892) *
                    (0.1792*(pl.log(abc[:,0]) - pl.log(0.9366*abc[:,1])) +
                     0.9837*(pl.log(abc[:,1]) - pl.log(0.9807*abc[:,2]))))
        lgj[:,1] = (-2*(0.764*lgj[:,0] + 9.2521) *
                    (0.9482*(pl.log(abc[:,0]) - pl.log(0.9366*abc[:,1])) -
                     0.3175*(pl.log(abc[:,1]) - pl.log(0.9807*abc[:,2]))))     
        return pl.reshape(lgj, s)
    
    # TODO: to_XYZ missing due to non-invertibility of L_OSA
    
    @classmethod
    def jacobian_XYZ(cls, data):
        """
        Return the Jacobian to XYZ, dLGJ^i/dXYZ^j.

        The Jacobian is calculated at the given data points (of the
        Data class).
        """
        xyz = data.get(SpaceXYZ)
        jac, jac_s = cls.empty_matrix(xyz)
        xyz, d_s = cls.preshape(xyz)
        xyY = SpacexyY.from_XYZ(xyz)
        abc = SpaceLGJ_ABC.from_XYZ(xyz)
        lgj = cls.from_XYZ(xyz)
        xyz_d = Data(SpaceXYZ, xyz)
        dxyY_dXYZ = SpacexyY.jacobian_XYZ(xyz_d)
        dABC_dXYZ = SpaceLGJ_ABC.jacobian_XYZ(xyz_d)
        
        Y0 = 100*xyY[:,2]*(4.4934*xyY[:,0]**2 + 4.3034*xyY[:,1]**2 -
                           4.2760*xyY[:,0]*xyY[:,1] - 1.3744*xyY[:,0] -
                           2.5643*xyY[:,1] + 1.8103)                           

        dL_dY0 = 5.9*(Y0**(-2/3.) + 0.0042*abs(Y0 - 30)**(-2/3.))/(3*pl.sqrt(2))       

        dY0_dx = 100*xyY[:,2]*(4.4934*2*xyY[:,0] - 4.2760*xyY[:,1] - 1.3744)
        dY0_dy = 100*xyY[:,2]*(4.3034*2*xyY[:,1] - 4.2760*xyY[:,0] - 2.5643)
        dY0_dY = 100*(4.4934*xyY[:,0]**2 + 4.3034*xyY[:,1]**2 -
                      4.2760*xyY[:,0]*xyY[:,1] - 1.3744*xyY[:,0] -
                      2.5643*xyY[:,1] + 1.8103)

        dL_dx = dL_dY0 * dY0_dx
        dL_dy = dL_dY0 * dY0_dy
        dL_dY = dL_dY0 * dY0_dY
        
        dL_dX = (dL_dx[:] * dxyY_dXYZ[:,0,0] +
                 dL_dy[:] * dxyY_dXYZ[:,1,0] +
                 dL_dY[:] * dxyY_dXYZ[:,2,0])
        dL_dZ = (dL_dx[:] * dxyY_dXYZ[:,0,2] +
                 dL_dy[:] * dxyY_dXYZ[:,1,2] +
                 dL_dY[:] * dxyY_dXYZ[:,2,2])

        TG = (0.9482*(pl.log(abc[:,0]) - pl.log(0.9366*abc[:,1])) -
              0.3175*(pl.log(abc[:,1]) - pl.log(0.9807*abc[:,2])))
        TJ = (0.1792*(pl.log(abc[:,0]) - pl.log(0.9366*abc[:,1])) +
              0.9837*(pl.log(abc[:,1]) - pl.log(0.9807*abc[:,2])))
        SG = -2 * (0.764  * lgj[:,0] + 9.2521)
        SJ =  2 * (0.5735 * lgj[:,0] + 7.0892)
        
        dG_dL = TG * -2 * 0.764
        dG_dA = SG * 0.9482 / abc[:,0]
        dG_dB = SG * (-0.9482 - 0.3175) / abc[:,1]
        dG_dC = SG * 0.3175 / abc[:,2]
        dJ_dL = TJ *  2 * 0.57354
        dJ_dA = SJ * 0.1792 / abc[:,0]
        dJ_dB = SJ * (-0.1792 + 0.9837) / abc[:,1]
        dJ_dC = SJ * -0.9837 / abc[:,2]
        
        dG_dX = (dG_dL * dL_dX +
                 dG_dA * dABC_dXYZ[:,0,0] +
                 dG_dB * dABC_dXYZ[:,1,0] +
                 dG_dC * dABC_dXYZ[:,2,0])
        dG_dY = (dG_dL * dL_dY +
                 dG_dA * dABC_dXYZ[:,0,1] +
                 dG_dB * dABC_dXYZ[:,1,1] +
                 dG_dC * dABC_dXYZ[:,2,1])
        dG_dZ = (dG_dL * dL_dZ +
                 dG_dA * dABC_dXYZ[:,0,2] +
                 dG_dB * dABC_dXYZ[:,1,2] +
                 dG_dC * dABC_dXYZ[:,2,2])
        
        dJ_dX = (dJ_dL * dL_dX +
                 dJ_dA * dABC_dXYZ[:,0,0] +
                 dJ_dB * dABC_dXYZ[:,1,0] +
                 dJ_dC * dABC_dXYZ[:,2,0])
        dJ_dY = (dJ_dL * dL_dY +
                 dJ_dA * dABC_dXYZ[:,0,1] +
                 dJ_dB * dABC_dXYZ[:,1,1] +
                 dJ_dC * dABC_dXYZ[:,2,1])
        dJ_dZ = (dJ_dL * dL_dZ +
                 dJ_dA * dABC_dXYZ[:,0,2] +
                 dJ_dB * dABC_dXYZ[:,1,2] +
                 dJ_dC * dABC_dXYZ[:,2,2])

        for i in range(len(dL_dX)):
            jac[i] = pl.array([[dL_dX[i], dL_dY[i], dL_dZ[i]],
                               [dG_dX[i], dG_dY[i], dG_dZ[i]],
                               [dJ_dX[i], dJ_dY[i], dJ_dZ[i]]])
        return pl.reshape(jac, jac_s)
        
class SpaceLGJE(Space):
    """
    The log-compressed OSA-UCS colour space for the DEE metric.
    """
    
    @classmethod
    def from_XYZ(cls, ndata):
        """
        Convert from XYZ to LGJE (using LJG).
        """
        xyz, s = cls.preshape(ndata)
        lgj = SpaceLGJ.from_XYZ(xyz)
        lgje = pl.zeros(pl.shape(xyz))
        C_OSA = pl.sqrt(lgj[:,1]**2 + lgj[:,2]**2)
        C_E = pl.log(1 + 0.050/1.256*(10*C_OSA))/0.050       
        h = pl.arctan2(-lgj[:,2], lgj[:,1])
        lgje[:,0] = pl.log(1 + 0.015/2.980*(10*lgj[:,0]))/0.015
        lgje[:,1] = -C_E*pl.cos(h)
        lgje[:,2] = C_E*pl.sin(h)
        return pl.reshape(lgje, s)

# TODO:
#
# - Parametrise CIELAB, CIELUV etc. for white point?
#
# - Make a general linear colour space, and let XYZ, LUV, RGB etc be
#   defined as instances of that space?
#
# - Make von Kries/Bradford etc. adapted XYZ colour spaces as
#   instances of a generalised adapted XYZ? Or directly as instances
#   of the general linear colour space (above)?
#
# Classes:
#     SpaceSRGB
#     SpaceLogOSAUCS
#     SpaceDE2000 (which, how?)
#     SpaceLinearTransform (using M and M_inv class attributes)
#          SpaceLMS_...
#          SpaceXYZ_D50_Bradford
#          SpaceXYZ_D55_Bradford
#          ...
#          SpaceXYZ_D50_vonKries
#          SpaceXYZ_D55_vonKries
#          ...
#          SpaceLinearRGB_sRGB
#          SpaceLinearRGB_CIE
#          ...
#     SpaceGammaTransform (using lin_space and gamma class attributes)
#          SpaceRGB_Adobe
#          ...
#     Combined spaces
#          SpaceUI (experimental)
#          SpaceIPT
#     SpaceCIECAM (needs instances; parameters for surround etc.)

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

    def set(self, space, ndata):
        """
        Set colour space and data.

        A new dictionary is constructed, and the data are added in the
        provided colour space, as well as in the XYZ colour space
        (using the SpaceXYZ class).
        """
        self.data = dict()
        self.data[space] = ndata
        if space != SpaceXYZ:
            self.data[SpaceXYZ] = space.to_XYZ(ndata)

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
            self.data[space] = space.from_XYZ(self.data[SpaceXYZ])
            return self.data[space]

class MetricData:
    """
    Class for keeping colour metric data in various colour spaces.
    """
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
        if space != SpaceXYZ:
            self.metrics[SpaceXYZ] = \
                space.metrics_to_XYZ(points_data, metrics_ndata)

    def get_points(self, space):
        """
        Return colour data in required colour space.

        Taken care of by the Data class.
        """
        return self.points.get(space)

    def get_metrics(self, space):
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
                space.metrics_from_XYZ(self.points, self.metrics[SpaceXYZ])
            return self.metrics[space]

    def get_ellipses(self, space, plane=pl.s_[0:2], scale=1):
        """
        Return Ellipse objects in the required plane of the given space.

        For now, plane is represented by a slice giving the correct
        range for the arrays. Should perhaps be changed in the future.
        
        a,b: s_[1:3]
        a,L: s_[1::-1]
        b,L= s_[2::-2]
        x,y: s_[0:2]
        """
        metrics, sh = Space.preshape_matrix(self.get_metrics(space).copy())
        points, sh = Space.preshape(self.get_points(space).copy())
        metrics = metrics[:, plane, plane]
        points = points[:, plane]
        ells = []
        for i in range(sh[0]):
            g11 = metrics[i, 0, 0]
            g22 = metrics[i, 1, 1]
            g12 = metrics[i, 0, 1]
            if g11 == g22:
                theta = 0
                a = 1 / pl.sqrt(g22)
                b = 1 / pl.sqrt(g11)
            else:
                theta = pl.arctan2(2*g12, g11 - g22) * 0.5
                a = 1 / pl.sqrt(g22 + g12 / pl.tan(theta))
                b = 1 / pl.sqrt(g11 - g12 / pl.tan(theta))
            ells.append(Ellipse(points[i],
                                width=2*a*scale, height=2*b*scale,
                                angle=theta * 180 / pl.pi))
        return ells

#==============================================================================
# Colour data sets
#==============================================================================

def build_d_XYZ(field_size=2, age=32, resolution=1):
    """
    Read CIE XYZ31 functions by TC1-82 and return Data object.
    """
    xyz, cc, lms, trans_mat, lambda_ref_min = tc.xyz(field_size, age, resolution)
    return Data(SpaceXYZ, xyz[:,1:])

def build_d_Melgosa():
    """
    The data points for the Melgosa Ellipsoids (RIT-DuPont).

    Copied verbatim from pdf of CRA paper. Uses the ellipsoids fitted
    in CIELAB and returns MetricData.
    """
    m_a = pl.array([-1.403 ,-16.374, -0.782, -27.549, 12.606, 12.153,
                     35.646, 1.937, -10.011, -0.453, -30.732, 21.121,
                     -33.638, -13.440, 25.237, 31.509, 6.826, 0.307,
                     18.226])
    m_b = pl.array([-27.810, -11.263, 1.049, 2.374, 20.571, -13.079, 
                     21.403, 35.638, 13.281, 0.421, -5.030, 17.804, 
                     -5.012, -25.897, 3.409, -0.183, -31.146, 0.214,
                     79.894])
    m_L = pl.array([35.338, 50.259, 59.334, 55.618, 62.928, 46.389,
                    42.315, 78.023, 64.938, 14.140, 68.678, 28.893,
                    31.683, 59.904, 17.357, 58.109, 30.186, 83.481,
                    76.057])
    m_Lab = pl.concatenate(([m_L], [m_a], [m_b]), axis=0).T
    return Data(SpaceCIELAB, m_Lab)

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
    points = Data(SpacexyY, xyY)
    a = rawdata[:,2]/1e3
    b = rawdata[:,3]/1e3
    theta = rawdata[:,4]*pl.pi/180.
    g11 = (pl.cos(theta)/a)**2 + (pl.sin(theta)/b)**2
    g22 = (pl.sin(theta)/a)**2 + (pl.cos(theta)/b)**2
    g12 = pl.cos(theta)*pl.sin(theta)*(1/a**2 - 1/b**2)
    g = pl.zeros((25,3,3))
    g[:, 0, 0] = g11
    g[:, 1, 1] = g22
    g[:, 2, 2] = 1e3 # arbitrary!
    g[:, 0, 1] = g12
    g[:, 1, 0] = g12
    return MetricData(SpacexyY, points, g)
    
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
    rawdata = pl.array(rawdata)
    xyY = rawdata[:,1:4].copy()
    xyY[:,2] = 0.4 # arbitrary!
    points = Data(SpacexyY, xyY)
    a = rawdata[:,4]/1e3 # correct?
    b = rawdata[:,5]/1e3 # corect?
    theta = rawdata[:,3]*pl.pi/180.
    g11 = (pl.cos(theta)/a)**2 + (pl.sin(theta)/b)**2
    g22 = (pl.sin(theta)/a)**2 + (pl.cos(theta)/b)**2
    g12 = pl.cos(theta)*pl.sin(theta)*(1/a**2 - 1/b**2)
    g = pl.zeros((28,3,3))
    g[:, 0, 0] = g11
    g[:, 1, 1] = g22
    g[:, 2, 2] = 1e3 # arbitrary!
    g[:, 0, 1] = g12
    g[:, 1, 0] = g12
    return MetricData(SpacexyY, points, g)

def build_g_Melgosa_Lab():
    """
    Melgosa's CIELAB-fitted ellipsoids for the RIT-DuPont data.

    Copied verbatim from pdf of CRA paper. Uses the ellipsoids fitted
    in CIELAB and returns MetricData.
    """
    m_gaa = pl.array([0.6609, 0.3920, 1.3017, 0.1742, 0.5967, 0.5374,
                      0.2837, 0.6138, 0.7252, 1.6002, 0.1760, 0.8512,
                      0.0543, 0.3547, 0.2381, 0.1729, 0.7289, 0.9614,
                      0.2896])
    m_gbb = pl.array([0.2387, 0.4286, 0.5241, 0.5847, 0.4543, 0.3048,
                      0.3717, 0.2465, 0.4370, 0.4790, 0.2589, 0.4054,
                      0.7178, 0.2057, 0.3801, 0.2532, 0.4255, 0.1984,
                      0.0522])
    m_gab = pl.array([0.3080, -0.0386, 0.1837, 0.0632, -0.1913, 0.2772,
                      -0.1215, -0.0757, 0.1565, 0.0971, 0.0941, -0.2578,
                      -0.1148, 0.1671, 0.0229, 0.0362, 0.5275, 0.1822,
                      0.0023])
    m_gLa = pl.array([-0.0144, 0.0812, -0.1435, 0.0996, -0.0008, -0.0115,
                       0.0644, 0.0315, 0.2465, -0.0120, 0.1255, 0.1046,
                       0.1319, 0.0924, 0.0952, -0.0134, 0.0128, -0.1378,
                       -0.0459])
    m_gLb = pl.array([-0.1315, 0.0373, -0.1890, -0.1696, -0.1447, 0.0525,
                       -0.0927, -0.0833, -0.1251, 0.0357, -0.0153, 0.1334,
                       -0.1589, 0.1759, -0.1561, 0.0341, 0.0113, 0.0070,
                       -0.0288])
    m_gLL = pl.array([1.1973, 1.6246, 1.3061, 1.0817, 1.1507, 1.2378, 0.9709,
                      0.7855, 1.3469, 0.6585, 0.9418, 0.9913, 0.8693, 0.8080,
                      0.8277, 0.5755, 0.9311, 0.5322, 0.4228])
    m_Lab_metric = pl.zeros((19,3,3))
    m_Lab_metric[:, 0, 0] = m_gLL
    m_Lab_metric[:, 1, 1] = m_gaa
    m_Lab_metric[:, 2, 2] = m_gbb
    m_Lab_metric[:, 0, 1] = m_gLa
    m_Lab_metric[:, 1, 0] = m_gLa
    m_Lab_metric[:, 0, 2] = m_gLb
    m_Lab_metric[:, 2, 0] = m_gLb
    m_Lab_metric[:, 1, 2] = m_gab
    m_Lab_metric[:, 2, 1] = m_gab
    return MetricData(SpaceCIELAB, build_d_Melgosa(), m_Lab_metric)
    
def build_g_Melgosa_xyY():
    """
    Melgosa's xyY-fitted ellipsoids for the RIT-DuPont data.

    Copied verbatim from pdf of CRA paper. Uses the ellipsoids fitted
    in xyY and returns MetricData.
    """
    m_g11 = pl.array([10.074, 5.604, 18.738,3.718, 5.013, 7.462, 1.229,
                      7.634, 11.805, 3.578, 5.359, 1.770, 0.368, 9.407,
                      0.624, 2.531, 11.222, 26.497, 3.762])
    m_g22 = pl.array([3.762, 6.589, 14.619, 3.310, 13.314, 3.533, 5.774,
                      11.162, 7.268, 3.007, 1.769, 6.549, 2.348, 3.485,
                      2.091, 4.122, 2.623, 16.086, 4.597])
    m_g12 = pl.array([-5.498, -3.518, -12.571, 0.219, -4.689, -3.946, -0.365,
                      -6.096, -5.562, -2.698, -0.902, -2.052, 0.040, -4.170,
                      -0.434, -1.074, -4.884, -18.122, -1.715])
    m_g13 = pl.array([-1.607, 0.001, -0.776, -0.078, -0.037, 0.212, 0.683,
                      0.049, 0.560, -0.103, 0.223, 2.341, 0.538, -0.240, 1.825,
                      0.285, -2.174, -0.361, 0.064])
    m_g23 = pl.array([-0.509, -0.346, 0.147, -0.489, -0.121, -0.065, -1.676,
                      -0.020, -0.521, 0.831, -0.291, -1.436, -0.936, 0.480,
                      -3.806, -0.058, 0.659, 0.343, 0.088])
    m_g33 = pl.array([5.745, 2.426, 1.146, 1.111, 0.845, 2.311, 2.878, 0.287,
                      0.912, 21.381, 0.517, 9.775, 3.823, 0.687, 23.949, 0.564,
                      6.283, 0.160, 0.169])
    m_xyY_metric = pl.zeros((19,3,3))
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
    return MetricData(SpacexyY, build_d_Melgosa(), m_xyY_metric)

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
    g, sh = Space.empty_matrix(data.get(SpaceXYZ))
    for i in range(pl.shape(g)[0]):
        g[i] = pl.eye(3)
    g = pl.reshape(g, sh)
    return MetricData(space, data, g)

def metric_DEab(data):
    """
    Compute the DEab metric as MetricData for the given data points.
    """
    return metric_Euclidean(SpaceCIELAB, data)

def metric_DEuv(data):
    """
    Compute the DEuv metric as MetricData for the given data points.
    """
    return metric_Euclidean(SpaceCIELUV, data)

def metric_DEE(data):
    """
    Compute the DEuv metric as MetricData for the given data points.
    """
    return metric_Euclidean(SpaceLGJE, data)

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
        axis = pl.gca()
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
    g_data = build_g_Melgosa_Lab()
    g_metric = metric_DEab(g_data.points)
    
    d = Data(SpacexyY, [[.4, .3, .4], [.1,.2,.3]])
    print SpaceLGJ.jacobian_XYZ(d)
    
    # Planes:
    #
    # a,b: s_[1:3]
    # a,L: s_[1::-1]
    # b,L= s_[2::-2]
    # x,y: s_[0:2]

#    space = SpaceCIELAB
#    plane = pl.s_[2::-2]
#    
#    pl.clf()
#    xyz = build_d_XYZ()
#    plot_ellipses(g_data.get_ellipses(space, plane, scale=3))
#    plot_ellipses(g_metric.get_ellipses(space, plane, scale=5), edgecolor=[1,0,0])
##    pl.plot(xyz.get(space)[:,0], xyz.get(space)[:,1], 'k')
#    pl.axis('equal')
#    pl.grid()
##    pl.ylim(-100,100)
#    pl.xlim(-120,120)
#    pl.show()