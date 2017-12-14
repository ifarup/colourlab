#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data: Colour data, part of the colourlab package

Copyright (C) 2013-2017 Ivar Farup

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

import os
import re
import numpy as np
import inspect
from matplotlib.patches import Ellipse
from . import space, misc


# =============================================================================
# Colour data
# =============================================================================


class Points:
    """
    Class for keeping colour data in various colour spaces and shapes.
    """

    def __init__(self, sp, ndata):
        """
        Construct new instance and set colour space and data.

        Parameters
        ----------
        sp : space.Space
            The colour space for the given instanisiation data.
        ndata : ndarray
            The colour data in the given space.
        """
        self.data = None
        self.sh = None
        self.flattened_XYZ = None
        self.set(sp, ndata)

    def flatten(self, ndata):
        """
        Shape the data so that is becomes an PxC matrix or C vector.

        The data should be of the shape M x ... x N x C, where C is the
        number of colour channels. Returns the shaped data as a P x C
        matrix where P = M x ... x N, as well as the shape of the input
        data. Get back to original shape by reshape(data, shape).

        Parameters
        ----------
        ndata : ndarray
            M x ... x N x C array of colour data

        Returns
        -------
        ndata : ndarray
            P x C array of colour data, P = M * ... * N
        """
        sh = np.shape(ndata)
        sh_array = np.array(sh)
        P_data = np.prod(sh_array[:len(sh) - 1])
        C_data = sh[len(sh) - 1]
        return np.reshape(ndata, [P_data, C_data])

    def set(self, sp, ndata):
        """
        Set colour space and data.

        A new dictionary is constructed, and the data are added in the
        provided colour space, as well as in the XYZ colour space
        (using the SpaceXYZ class).

        Parameters
        ----------
        sp : space.Space
            The colour space for the given instanisiation data.
        ndata : ndarray
            The colour data in the given space.
        """
        ndata = np.array(ndata)
        self.data = dict()
        self.data[sp] = ndata
        self.sh = ndata.shape
        flattened_data = self.flatten(ndata)
        if sp == space.xyz:
            self.flattened_XYZ = flattened_data
        else:
            self.flattened_XYZ = sp.to_XYZ(flattened_data)
            self.data[space.xyz] = np.reshape(self.flattened_XYZ, self.sh)

    def get(self, sp):
        """
        Return colour data in required colour space.

        If the data do not currently exist in the required colour
        space, the necessary colour conversion will take place, and
        the results stored in the object or future use.

        Parameters
        ----------
        sp : space.Space
            The colour space for the returned data.

        Returns
        -------
        ndata : ndarray
            The colour data in the given colour space.
        """
        if sp in self.data:
            return self.data[sp]
        else:
            flattened_data = sp.from_XYZ(self.flattened_XYZ)
            ndata = np.reshape(flattened_data, self.sh)
            self.data[sp] = ndata
            return ndata

    def get_flattened(self, sp):
        """
        Return colour data in required colour space in PxC format.

        If the data do not currently exist in the required colour
        space, the necessary colour conversion will take place, and
        the results stored in the object or future use.

        Parameters
        ----------
        sp : space.Space
            The colour space for the returned data.

        Returns
        -------
        ndata : ndarray
            The flattend colour data in the given colour space.
        """
        return self.flatten(self.get(sp))

    def new_white_point(self, sp, from_white, to_white):
        """
        Return new data set with new white point.

        The transformation is done using the von Kries transformation
        in the given colour space.

        Parameters
        ----------
        sp : space.Space
            The colour space for the von Kries transformation.
        from_white : data.Points
            The white point of the current data set.
        to_white : data.Points
            The white point of the new data set.

        Returns
        -------
        data : data.Points
            The new colour data with changed white point.
        """
        wh_in = from_white.get(sp)
        wh_out = to_white.get(sp)
        von_kries_mat = np.array([[wh_out[0] / wh_in[0], 0, 0],
                                  [0, wh_out[1] / wh_in[1], 0],
                                  [0, 0, wh_out[2] / wh_in[2]]])
        return Points(sp, self.get(space.TransformLinear(sp, von_kries_mat)))


class Vectors:
    """
    Class for keeping contravariant vector data in various colour spaces.
    """

    def __init__(self, sp, vectors_ndata, points_data):
        """
        Construct new instance and set colour space and data.

        Parameters
        ----------
        sp: space.Space
           The colour space for the given vector data
        metrics_ndata : ndarray
            The tensor data in the given colour space at the given points.
        points_data : space.Points
            The colour points for the given vector data.
        """
        self.points = None
        self.vectors = None
        self.sh = None
        self.flattened_XYZ = None
        self.set(sp, vectors_ndata, points_data)

    def flatten(self, ndata):
        """
        Shape the data so that is becomes an PxC matrix or C vector.

        The data should be of the shape M x ... x N x C, where C is
        the number of colour channels. Returns the shaped data as a P
        x C matrix where P = M x ... x N, as well as the shape of the
        input data. Get back to original shape by reshape(data,
        shape).

        Parameters
        ----------
        ndata : ndarray
            M x ... x N x C array of colour data

        Returns
        -------
        ndata : ndarray
            P x C array of colour data, P = M * ... * N
        """
        sh = np.shape(ndata)
        sh_array = np.array(sh)
        P_data = np.prod(sh_array[:len(sh) - 1])
        C_data = sh[len(sh) - 1]
        return np.reshape(ndata, [P_data, C_data])

    def set(self, sp, vectors_ndata, points_data):
        """
        Set colour sp, points, and vectorss data.

        The points_data are taken care already of the type Points. A new
        dictionary is constructed, and the vectors_ndata are added in
        the provided colour space, as well as in the XYZ colour space
        (using the SpaceXYZ class).

        Parameters
        ----------
        sp : space.Space
            The colour space for the given tensor data.
        vectors_ndata : ndarray
            The vector data in the given colour space at the given points.
        points_data : data.Points
            The colour points for the given tensor data.
        """

        self.points = points_data
        self.vectors = dict()
        vectors_ndata = np.array(vectors_ndata)
        self.vectors[sp] = vectors_ndata
        self.sh = vectors_ndata.shape
        flattened_data = self.flatten(vectors_ndata)
        if sp == space.xyz:
            self.flattened_XYZ = flattened_data
        else:
            self.flattened_XYZ = sp.vectors_to_XYZ(self.points, flattened_data)
            self.vectors[space.xyz] = np.reshape(self.flattened_XYZ, self.sh)

    def get(self, sp):
        """
        Return colour vector data in required colour space.

        If the data do not currently exist in the required colour
        space, the necessary colour conversion will take place, and
        the results stored in the object or future use.

        Parameters
        ----------
        sp : space.Space
            The colour space for the returned data.

        Returns
        -------
        ndata : ndarray
            The colour vector data in the given colour space.
        """
        if sp in self.vectors:
            return self.vectors[sp]
        else:
            flattened_data = sp.vectors_from_XYZ(self.points, self.flattened_XYZ)
            ndata = np.reshape(flattened_data, self.sh)
            self.vectors[sp] = ndata
            return ndata

    def get_flattened(self, sp):
        """
        Return colour vector data in required colour space in PxC format.

        If the data do not currently exist in the required colour
        space, the necessary colour conversion will take place, and
        the results stored in the object or future use.

        Parameters
        ----------
        sp : space.Space
            The colour space for the returned data.

        Returns
        -------
        ndata : ndarray
            The flattend colour vector data in the given colour space.
        """
        return self.flatten(self.get(sp))


class Tensors:
    """
    Class for keeping colour metric data in various colour spaces.
    """

    # Cross sectional planes for ellipses
    plane_01 = np.s_[0:2]
    plane_12 = np.s_[1:3]
    plane_10 = np.s_[1::-1]
    plane_20 = np.s_[2::-2]

    plane_xy = plane_01
    plane_ab = plane_12
    plane_aL = plane_10
    plane_bL = plane_20

    def __init__(self, sp, metrics_ndata, points_data):
        """
        Construct new instance and set colour space and data.

        Parameters
        ----------
        sp : space.Space
            The colour space for the given tensor data.
        metrics_ndata : ndarray
            The tensor data in the given colour space at the given points.
        points_data : data.Points
            The colour points for the given tensor data.
        """
        self.points = None
        self.metrics = None
        self.sh = None
        self.flattened_XYZ = None
        self.set(sp, metrics_ndata, points_data)

    def flatten(self, ndata):
        """
        Shape the data so that is becomes an PxCxC matrix or CxC matrix

        The data should be of the shape M x ... x N x C x D, where C is the
        number of colour channels. Returns the shaped data as a P x C
        matrix where P = M x ... x N, as well as the shape of the input
        data. Get back to original shape by reshape(data, shape).

        Parameters
        ----------
        ndata : ndarray
            M x ... x N x C x C array of colour metrics

        Returns
        -------
        ndata : ndarray
            P x C x C array of colour metrics, P = M * ... * N
        """
        sh = np.shape(ndata)
        sh_array = np.array(sh)
        P_data = np.prod(sh_array[:len(sh) - 2])
        C_data = sh[len(sh) - 2:]
        return np.reshape(ndata, [P_data, C_data[0], C_data[1]])

    def set(self, sp, metrics_ndata, points_data):
        """
        Set colour sp, points, and metrics data.

        The points_data are taken care already of the type Points. A new
        dictionary is constructed, and the metrics_ndata are added in
        the provided colour space, as well as in the XYZ colour space
        (using thespace.SpaceXYZ class).

        Parameters
        ----------
        sp : space.Space
            The colour space for the given tensor data.
        metrics_ndata : ndarray
            The tensor data in the given colour space at the given points.
        points_data : data.Points
            The colour points for the given tensor data.
        """
        self.points = points_data
        self.metrics = dict()
        self.sh = metrics_ndata.shape
        self.metrics[sp] = metrics_ndata
        flattened_data = self.flatten(metrics_ndata)
        if sp == space.xyz:
            self.flattened_XYZ = flattened_data
        else:
            self.flattened_XYZ = sp.metrics_to_XYZ(points_data, flattened_data)
            self.metrics[space.xyz] = np.reshape(self.flattened_XYZ, self.sh)

    def get(self, sp):
        """
        Return metric data in required colour space.

        If the data do not currently exist in the required colour
        space, the necessary colour conversion will take place, and
        the results stored in the object or future use.

        Parameters
        ----------
        sp : space.Space
            The colour space in which to return the tensor data.

        Returns
        -------
        tensors : ndarray
            Array of tensors in the given colour space.
        """
        if sp in self.metrics:
            return self.metrics[sp]
        else:
            flattened_metrics = sp.metrics_from_XYZ(self.points, self.flattened_XYZ)
            metrics_ndata = np.reshape(flattened_metrics, self.sh)
            self.metrics[sp] = metrics_ndata
            return metrics_ndata

    def get_flattened(self, sp):
        """
        Return colour data in required colour space in PxC format.

        If the data do not currently exist in the required colour
        space, the necessary colour conversion will take place, and
        the results stored in the object or future use.

        Parameters
        ----------
        sp : space.Space
            The colour space for the returned data.

        Returns
        -------
        ndata : ndarray
            The flattend colour data in the given colour space.
        """
        return self.flatten(self.get(sp))

    def get_ellipse_parameters(self, sp, plane=plane_xy, scale=1):
        """
        Return ellipse parameters a, b, theta in the required plane.

        The plane is in the given space. For now, plane is represented
        by a slice giving the correct range for the arrays. Should
        perhaps be changed in the future.

        Parameters
        ----------
        sp : space.Space
            The space in which to give the ellipse parameters.
        plane : slice
            The principal plan for the ellipsoid cross sections.
        scale : float
            The scaling (magnification) factor for the ellipses.

        Returns
        -------
        a_b_theta : ndarray
            N x 3 array of a, b, theta ellipse parameters.
        """
        metrics = self.get_flattened(sp).copy()
        points = self.points.get_flattened(sp).copy()
        a_b_theta = np.zeros(np.shape(points))
        metrics = metrics[..., plane, plane]
        points = points[:, plane]
        for i in range(np.shape(metrics)[0]):
            g11 = metrics[i, 0, 0]
            g22 = metrics[i, 1, 1]
            g12 = metrics[i, 0, 1]
            theta = np.arctan2(2*g12, g11 - g22) * 0.5
            if theta == 0:
                a = 1 / np.sqrt(g11)
                b = 1 / np.sqrt(g22)
            else:
                a = 1 / np.sqrt(g22 + g12 / np.tan(theta))
                b = 1 / np.sqrt(g11 - g12 / np.tan(theta))
            a_b_theta[i, 0] = a * scale
            a_b_theta[i, 1] = b * scale
            a_b_theta[i, 2] = theta
        return a_b_theta

    def get_ellipses(self, sp, plane=plane_xy, scale=1):
        """
        Return Ellipse objects in the required plane of the given space.

        For now, plane is represented by a slice giving the correct
        range for the arrays. Should perhaps be changed in the future.

        Parameters
        ----------
        sp : space.Space
            The space in which to give the ellipse parameters.
        plane : slice
            The principal plan for the ellipsoid cross sections.
        scale : float
            The scaling (magnification) factor for the ellipses.

        Returns
        -------
        ellipses : list
            List of Ellipse objects.
        """
        a_b_theta = self.get_ellipse_parameters(sp, plane, scale)
        points = self.points.get_flattened(sp).copy()
        points = points[:, plane]
        ells = []
        for i in range(np.shape(a_b_theta)[0]):
            ells.append(Ellipse(points[i],

                                width=2 * a_b_theta[i, 0],
                                height=2 * a_b_theta[i, 1],
                                angle=a_b_theta[i, 2] * 180 / np.pi))
        return ells

    def inner(self, sp, vec1, vec2):
        """
        Return the inner product of the two vectors computed in the given space.

        The result should in theory be invariant with respect to the colour space.

        Parameters
        ----------
        sp : space.Space
            The space in which to compute the inner product
        vec1: Vectors
            The first vector
        vec2: Vectors
            The second vector

        Returns
        -------
        inner : ndarray
            The inner products (scalars)
        """
        return np.einsum('...ij,...i,...j', self.get(sp),
                         vec1.get(sp), vec2.get(sp))

    def norm_sq(self, sp, vec):
        """
        Compute the squared norm of a vector data set with a given metric tensor.

        The vector set and the tensor data set must have corresponding dimensions.

        Parameters
        ----------
        sp : space.Space
            The space in which to compute the inner product
        vec: Vectors
            The vectors

        Returns
        -------
        norms: ndarray
            Array with numerical (scalar) values of the squared norm.
        """
        return self.inner(sp, vec, vec)
    
    def norm(self, sp, vec):
        """
        Compute the norm of a vector data set with a given metric tensor.

        The vector set and the tensor data set must have corresponding dimensions.

        Parameters
        ----------
        sp : space.Space
            The space in which to compute the inner product
        vec: Vectors
            The vectors

        Returns
        -------
        norms: ndarray
            Array with numerical (scalar) values of the norm.
        """
        return np.sqrt(self.inner(sp, vec, vec))


# =============================================================================
# Colour data sets
# =============================================================================

def resource_path(relative):
    """
    Extend relative path to full path (mainly for setuptools integration).

    Parameters
    ----------
    relative : string
        The relative path name.

    Returns
    -------
    absolute : string
        The absolute path name.
    """
    return os.path.dirname(
        os.path.abspath(
            inspect.getsourcefile(resource_path))) + '/' + relative


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
    f.close()
    for i in range(len(data)):
        data[i] = data[i].split(',')
        for j in range(len(data[i])):
            if data[i][j].strip() == '':
                data[i][j] = pad
            else:
                data[i][j] = float(data[i][j])
    return np.array(data)

# White points:

white_A = Points(space.xyz, space.Space.white_A)
white_B = Points(space.xyz, space.Space.white_B)
white_C = Points(space.xyz, space.Space.white_C)
white_D50 = Points(space.xyz, space.Space.white_D50)
white_D55 = Points(space.xyz, space.Space.white_D55)
white_D65 = Points(space.xyz, space.Space.white_D65)
white_D75 = Points(space.xyz, space.Space.white_D75)
white_E = Points(space.xyz, space.Space.white_E)
white_F2 = Points(space.xyz, space.Space.white_F2)
white_F7 = Points(space.xyz, space.Space.white_F7)
white_F11 = Points(space.xyz, space.Space.white_F11)


def d_XYZ_31():
    """
    Read CIE XYZ 1931 functions.

    Returns
    -------
    xyz_31 : data.Points
        The XYZ 1931 colour matching functions.
    """
    xyz_ = read_csv_file('colour_data/ciexyz31_1.csv')
    return Points(space.xyz, xyz_[:, 1:])


def d_XYZ_64():
    """
    Read CIE XYZ 1964 functions.

    Returns
    -------
    xyz_64 : data.Points
        The XYZ 1964 colour matching functions.
    """
    xyz_ = read_csv_file('colour_data/ciexyz64_1.csv')
    return Points(space.xyz, xyz_[:, 1:])


def d_Melgosa():
    """
    The data points for the Melgosa Ellipsoids (RIT-DuPont).

    Copied verbatim from pdf of CRA paper. Uses the ellipsoids fitted
    in CIELAB and returns Tensors.

    Returns
    -------
    d_Melgosa : data.Points
        The centre points of Melgosa's RIT-DuPont ellipsoids.
    """
    m_a = np.array([-1.403, -16.374, -0.782, -27.549, 12.606, 12.153,
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
    return Points(space.cielab, m_Lab)


def d_Munsell(dataset='real'):
    """
    The Munsell renotation data under illuminant C for the 2 degree observer.

    Parameters
    ----------
    dataset : string
        Which data set. Either 'all', 'real', or '1929'. See
        http://www.cis.rit.edu/research/mcsl2/online/munsell.php
        for details.

    Returns
    -------
    d_Munsell : data.Points
        The Munsell colours.
    munsell_names : list
        The standard Munsell value names (H, V, C).
    munsell_lab : ndarray
        Numeric version of the Munsell values names in a normalised
        Lab type coordinate system. Follows the layout of McCann
        J. Elect. Imag. 1999
    """
    if dataset == 'all' or dataset == 'real' or dataset == '1929':
        fname = 'colour_data/' + dataset + '.dat'
    else:
        raise RuntimeError('Non-existing Munsell data set: ' + str(dataset))
    infile = open(resource_path(fname), 'r')
    data = infile.readlines()
    infile.close()
    data = data[1:]
    for i in range(len(data)):
        data[i] = data[i].split()
    munsell_names = list(data)
    for i in range(len(munsell_names)):
        munsell_names[i] = munsell_names[i][0:3]
    munsell_hlc = list(munsell_names)
    for i in range(len(data)):
        data[i] = data[i][3:]
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    data = np.array(data)
    data[:, 2] = data[:, 2] / 100.
    data[data == 0] = 1e-16
    hue_list = ['10RP',
                '2.5R', '5R', '7.5R', '10R',
                '2.5YR', '5YR', '7.5YR', '10YR',
                '2.5Y', '5Y', '7.5Y', '10Y',
                '2.5GY', '5GY', '7.5GY', '10GY',
                '2.5G', '5G', '7.5G', '10G',
                '2.5BG', '5BG', '7.5BG', '10BG',
                '2.5B', '5B', '7.5B', '10B',
                '2.5PB', '5PB', '7.5PB', '10PB',
                '2.5P', '5P', '7.5P', '10P',
                '2.5RP', '5RP', '7.5RP']
    hue_lut = dict(zip(hue_list, 2 * np.pi * np.arange(len(hue_list)) /
                       float(len(hue_list))))
    for i in range(len(munsell_hlc)):
        munsell_hlc[i][0] = hue_lut[munsell_hlc[i][0]]
        for j in range(3):
            munsell_hlc[i][j] = float(munsell_hlc[i][j])
    munsell_hlc = np.array(munsell_hlc)
    munsell_hlc[:, 1] = munsell_hlc[:, 1] / 10.
    munsell_hlc[:, 2] = munsell_hlc[:, 2] / 20.
    munsell_lab = np.zeros(np.shape(munsell_hlc))
    munsell_lab[:, 0] = munsell_hlc[:, 1]
    munsell_lab[:, 1] = munsell_hlc[:, 2] * np.cos(munsell_hlc[:, 0])
    munsell_lab[:, 2] = munsell_hlc[:, 2] * np.sin(munsell_hlc[:, 0])
    return Points(space.xyY, data), munsell_names, munsell_lab


def d_regular(sp, x_val, y_val, z_val):
    """
    Build regular data set of colour data in the given colour space.

    x_val, y_val, and z_val should be one-dimensional arrays.

    Parameters
    ----------
    sp : space.Space
        The given colour space.
    x_val : ndarray
        Array of x values.
    y_val : ndarray
        Array of y values.
    z_val : ndarray
        Array of z values.

    Returns
    -------
    data : data.Points
        Regular structure of colour data in the given colour space.
    """
    x_len = np.shape(x_val)[0]
    y_len = np.shape(y_val)[0]
    z_len = np.shape(z_val)[0]
    tot_len = x_len * y_len * z_len
    ndata = np.zeros((tot_len, 3))
    l = 0
    for i in range(x_len):
        for j in range(y_len):
            for k in range(z_len):
                ndata[l, 0] = x_val[i]
                ndata[l, 1] = y_val[j]
                ndata[l, 2] = z_val[k]
                l = l + 1
    return Points(sp, ndata)

# TODO:
#
# Colour data sets, as needed (instances of Points):
#     patches_Munsell ++
#     patches_OSA ++ ???
#     patches_Colour Checker ++


# =============================================================================
# Metric data sets
# =============================================================================


def g_MacAdam():
    """
    MacAdam ellipses (defined in xy, extended arbitrarily to xyY).

    Arbitrarily uses Y=0.4 and g33 = 1e3 for extension to 3D.

    Returns
    -------
    MacAdam : Tensors
        The metric tensors corresponding to the MacAdam ellipsoids.
    """
    from scipy.io import loadmat
    rawdata = loadmat(resource_path('tensor_data/macdata(xyabtheta).mat'))
    rawdata = rawdata['unnamed']
    xyY = rawdata[:, 0:3].copy()
    xyY[:, 2] = 0.4             # arbitrary!
    points = Points(space.xyY, xyY)
    a = rawdata[:, 2]/1e3
    b = rawdata[:, 3]/1e3
    theta = rawdata[:, 4]*np.pi/180.
    g11 = (np.cos(theta)/a)**2 + (np.sin(theta)/b)**2
    g22 = (np.sin(theta)/a)**2 + (np.cos(theta)/b)**2
    g12 = np.cos(theta)*np.sin(theta)*(1/a**2 - 1/b**2)
    g = np.zeros((25, 3, 3))
    g[:, 0, 0] = g11
    g[:, 1, 1] = g22
    g[:, 2, 2] = 1e3            # arbitrary!
    g[:, 0, 1] = g12
    g[:, 1, 0] = g12
    return Tensors(space.xyY, g, points)


def g_three_observer():
    """
    Wyszecki and Fielder's three observer data set.

    Arbitrarily uses Y=0.4 and g33 = 1e3 for extension to 3D. It seems by
    comparing the data file to the original paper by Wyszecki and Fielder
    (JOSA, 1971) that only one of the data sets (GW) is represented in the
    file. Also, the paper reports a full 3D metric, so the arbitrary extension
    to 3D used here is not really called for.

    Returns
    -------
    threeObserver : Tensors
        The metric tensors corresponding to the three observer ellipsoids.
    """
    f = open(resource_path('tensor_data/3 observer.txt'))
    rawdata = f.readlines()[:-1]
    f.close()
    for line in range(len(rawdata)):
        rawdata[line] = rawdata[line].split('\t')
        for item in range(len(rawdata[line])):
            rawdata[line][item] = float(rawdata[line][item].strip())
    rawdata = np.array(rawdata)
    xyY = rawdata[:, 1:4].copy()
    xyY[:, 2] = 0.4             # arbitrary!
    points = Points(space.xyY, xyY)
    a = rawdata[:, 4] / 1e3     # correct?
    b = rawdata[:, 5] / 1e3     # corect?
    theta = rawdata[:, 3] * np.pi / 180.
    g11 = (np.cos(theta) / a)**2 + (np.sin(theta) / b)**2
    g22 = (np.sin(theta) / a)**2 + (np.cos(theta) / b)**2
    g12 = np.cos(theta)*np.sin(theta)*(1 / a**2 - 1 / b**2)
    g = np.zeros((28, 3, 3))
    g[:, 0, 0] = g11
    g[:, 1, 1] = g22
    g[:, 2, 2] = 1e3            # arbitrary!
    g[:, 0, 1] = g12
    g[:, 1, 0] = g12
    return Tensors(space.xyY, g, points)


def g_Melgosa_Lab():
    """
    Melgosa's CIELAB-fitted ellipsoids for the RIT-DuPont data.

    Copied verbatim from pdf of CRA paper. Uses the ellipsoids fitted
    in CIELAB and returns Tensors.

    Returns
    -------
    Melgosa : Tensors
        The metric tensors corresponding to Melgosa's ellipsoids.
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
    m_Lab_metric = np.zeros((19, 3, 3))
    m_Lab_metric[:, 0, 0] = m_gLL
    m_Lab_metric[:, 1, 1] = m_gaa
    m_Lab_metric[:, 2, 2] = m_gbb
    m_Lab_metric[:, 0, 1] = m_gLa
    m_Lab_metric[:, 1, 0] = m_gLa
    m_Lab_metric[:, 0, 2] = m_gLb
    m_Lab_metric[:, 2, 0] = m_gLb
    m_Lab_metric[:, 1, 2] = m_gab
    m_Lab_metric[:, 2, 1] = m_gab
    return Tensors(space.cielab, m_Lab_metric, d_Melgosa())


def g_Melgosa_xyY():
    """
    Melgosa's xyY-fitted ellipsoids for the RIT-DuPont data.

    Copied verbatim from pdf of CRA paper. Uses the ellipsoids fitted
    in xyY and returns Tensors.

    Returns
    -------
    Melgosa : Tensors
        The metric tensors corresponding to Melgosa's ellipsoids.
    """
    m_g11 = np.array([10.074, 5.604, 18.738, 3.718, 5.013, 7.462, 1.229,
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
    m_xyY_metric = np.zeros((19, 3, 3))
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
    return Tensors(space.xyY, m_xyY_metric, d_Melgosa())


def g_BFD(dataset='P'):
    """
    Return the BFD data set ellipses of the required type.

    Parameters
    ----------
    dataset : string
        The data set to use, either 'P', 'A', or '2', for perceptual,
        accept, and both, respectively.

    Returns
    -------
    bfd : Tensors
        The BDF data set of the required type
    """
    if dataset == 'P':
        file_name = resource_path('tensor_data/BFD_P.txt')
    elif dataset == 'A':
        file_name = resource_path('tensor_data/BFD_A.txt')
    elif dataset == '2':
        file_name = resource_path('tensor_data/BFD (2).txt')
    f = open(file_name, 'r')
    rawdata = f.readlines()
    f.close()
    for line in range(len(rawdata)):
        rawdata[line] = re.sub(r'\s+', ' ', rawdata[line]).strip()
        rawdata[line] = rawdata[line].split(' ')
        for item in range(len(rawdata[line])):
            rawdata[line][item] = float(rawdata[line][item])
    rawdata = np.array(rawdata)
    xyY = rawdata[:, 0:3].copy()
    xyY[:, 2] = xyY[:, 2] / 100
    points = Points(space.xyY, xyY)
    a = rawdata[:, 3] / 1e4     # correct?
    b = a / rawdata[:, 4]       # corect?
    theta = rawdata[:, 5] * np.pi / 180.
    g11 = (np.cos(theta) / a)**2 + (np.sin(theta) / b)**2
    g22 = (np.sin(theta) / a)**2 + (np.cos(theta) / b)**2
    g12 = np.cos(theta) * np.sin(theta) * (1 / a**2 - 1 / b**2)
    g = np.zeros((np.shape(rawdata)[0], 3, 3))
    g[:, 0, 0] = g11
    g[:, 1, 1] = g22
    g[:, 2, 2] = 1e3            # arbitrary!
    g[:, 0, 1] = g12
    g[:, 1, 0] = g12
    return Tensors(space.xyY, g, points)


# =============================================================================
# Metric datasets
# =============================================================================


def m_rit_dupont():
    """
    Read the full RIT-DuPont individual colour difference data from file.

    Returns
    -------
    rit_dupont : dict
        Dictionary with two datasets, dV, weights, and various metrics.
    """
    dat = read_csv_file(
        'metric_data/Mio_RIT_DuPont_Individual_Color_Difference_Data.csv')
    lab1 = dat[:, 0:3]
    lab2 = dat[:, 3:6]
    rit_dupont = dict()
    rit_dupont['data1'] = Points(space.cielab, lab1)
    rit_dupont['data2'] = Points(space.cielab, lab2)
    rit_dupont['dE_ab'] = dat[:, 6].copy()
    rit_dupont['dE_00'] = dat[:, 7].copy()
    rit_dupont['dE_94'] = dat[:, 8].copy()
    rit_dupont['dV'] = dat[:, 9].copy()
    rit_dupont['weights'] = dat[:, 10].copy()
    return rit_dupont


def m_rit_dupont_T50():
    """
    Read the reduced RIT-DuPont T50 colour difference data from file.

    Returns
    -------
    rit_dupont : dict
        Dictionary with two datasets and dV.
    """
    dat = read_csv_file('metric_data/Data_RIT-DuPont.csv')
    rit_dupont = dict()
    rit_dupont['data1'] = Points(space.cielab, dat[:, 0:3].copy())
    rit_dupont['data2'] = Points(space.cielab, dat[:, 3:6].copy())
    rit_dupont['dV'] = dat[:, 6].copy()
    return rit_dupont

# TODO:
#
# Metric data sets, as needed (instances of Tensors):
#     BrownMacAdam
#     +++
