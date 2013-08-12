#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIE TC1-82 Computations

Copyright (C) 2012 Ivar Farup and Jan Henrik Wold

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
import scipy.optimize
import scipy.interpolate

#==============================================================================
# Tabulated and derived visual data
#==============================================================================

def my_round(x,n=0):
    """
    Round array x to n decimal points using round half away from zero.
    
    Parameters
    ----------
    x : ndarray
        Array to be rounded
    n : int
        Number of decimal points
    
    Returns
    -------
    y : ndarray
        Rounded array
    """
    s = np.sign(x)
    return s*np.floor(np.absolute(x)*10**n + 0.5)/10**n
    
def significant_digits(x,n=0):
    """
    Round x to n significant digits (not decimal points).
    
    Parameters
    ----------
    x : int, float or ndarray
        Number or array to be rounded.
    
    Returns
    -------
    t : float or ndarray
        Rounded number or array.
    """
    if type(x) == float or type(x) == int:
        if x == 0.:
            return 0
        else:
            b = np.ceil(np.log10(x))
            return 10**b*my_round(x/10**b, n)
    b = x.copy()
    b[x == 0] = 0
    b[x != 0] = np.ceil(np.log10(abs(x[x != 0])))
    return 10**b*my_round(x/10**b, n)

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

def chromaticities_from_XYZ(xyz31, xyz64):
    """
    Compute chromaticities and knots (for interpolation) from XYZ functions.

    Parameters
    ----------
    xyz31 : ndarray
        CIE 1931 colour matching functions
    xyz64 : ndarray
        CIE 1964 colour matching functions

    Returns
    -------
    cc31 : ndarray
        Chromaticities for the 1931 colour matching functions.
    cc64 : ndarray
        Chromaticities for the 1964 colour matching functions.
    cc31knots : ndarray
        Knots for interpolating the chromaticities.
    cc64knots : ndarray
        Knots for interpolating the chromaticities.
    """
    cc31 = xyz31.copy()
    cc31sum = np.sum(cc31[:,1:], axis=1)
    cc64 = xyz64.copy()
    cc64sum = np.sum(cc64[:,1:], axis=1)
    for i in range(1,4):
        cc31[:,i] = cc31[:,i]/cc31sum
        cc64[:,i] = cc64[:,i]/cc64sum

#    cc31[:,1:] = my_round(cc31[:,1:], 6) # Skip !!!
#    cc64[:,1:] = my_round(cc64[:,1:], 6) # Skip !!!

    cc31knots = np.array([cc31[0,0],
                          cc31[np.argmin(cc31[:,1]),0],
                          cc31[np.argmax(cc31[:,2]),0],
                          700,
                          cc31[-1,0]])
                          
    cc64knots = np.array([cc64[0,0],
                          cc64[np.argmin(cc64[:,1]),0],
                          cc64[np.argmax(cc64[:,2]),0],
                          700,
                          cc64[-1,0]])
    return cc31, cc64, cc31knots, cc64knots

def gauss_func(param, docul2):
    b = param[0]
    x0 = param[1]
    f = 4*np.exp(-b*(docul2[:,0]-x0)**2)
    return f

def rms_error(param, docul2):
    f = gauss_func(param, docul2)
    return sum((f - docul2[:,1])**2)

def docul_fine(ocular_sum_32, docul2):
    """
    Calculate the two parts of docul.
    
    Parameters
    ----------
    ocular_sum_32 : ndarray
        Sum of two ocular functions
    docul2 : ndarray
        
    Returns
    -------
    docul1_fine : ndarray
        Tabulated docul1 with high resolution
    docul2_fine : ndarray
        Tabulated docul2 with high resolution
    """
    param = scipy.optimize.fmin(rms_error, [8e-4, 350], (docul2,), disp=False)
    docul2_add = np.array([[390, 4*np.exp(-param[0]*(390 - param[1])**2)],
                           [395, 4*np.exp(-param[0]*(395 - param[1])**2)]])
    docul2_pad = np.zeros((75,2))
    docul2_pad[:,0] = np.arange(460, 835, 5)
    docul2_pad[:,1] = 0
    docul2 = np.concatenate((docul2_add, docul2, docul2_pad))
    spl = scipy.interpolate.InterpolatedUnivariateSpline(docul2[:,0],
                                                         docul2[:,1])
    docul2_fine = ocular_sum_32.copy()
    docul2_fine[:,1] = spl(ocular_sum_32[:,0])
    docul1_fine = ocular_sum_32.copy()
    docul1_fine[:,1] = ocular_sum_32[:,1] - docul2_fine[:,1]
    return docul1_fine, docul2_fine
    
class VisualData:
    absorbance = read_csv_file('data/ssabance_fine.csv')
    ocular_sum_32 = read_csv_file('data/lensss_fine.csv') # 32 years only!!!
    macula = read_csv_file('data/macss_fine.csv')
    lms10_log_quant = read_csv_file('data/ss10q_fine_8dp.csv')
    lms10_lin_energ = read_csv_file('data/linss10e_fine_8dp.csv', 0)
    lms2_log_quant = read_csv_file('data/ss2_10q_fine_8dp.csv')
    lms2_lin_energ = read_csv_file('data/linss2_10e_fine_8dp.csv', 0)
    vlambdaLM_10_lin_energ = read_csv_file('data/linCIE2008v10e_fine_8dp.csv')
    vlambdaLM_2_lin_energ = read_csv_file('data/linCIE2008v2e_fine_8dp.csv')
    vlambdaLM_10_log_quant = read_csv_file('data/logCIE2008v10q_fine_8dp.csv')
    vlambdaLM_2_log_quant = read_csv_file('data/logCIE2008v2q_fine_8dp.csv')
    xyz31 = read_csv_file('data/ciexyz31_1.csv')
    xyz64 = read_csv_file('data/ciexyz64_1.csv')
    docul2 = read_csv_file('data/docul2.csv')

    cc31, cc64, cc31knots, cc64knots = chromaticities_from_XYZ(xyz31, xyz64)
    docul1_fine, docul2_fine = docul_fine(ocular_sum_32, docul2)
    
#==============================================================================
# Compute absorptance data from tabulated cone fundamentals; do se need these?
#==============================================================================

def absorptance_from_lms10q():
    """
    Compute the absorptance from quantal lms 10 for reference.
    """
    absorptance = VisualData.lms10_log_quant.copy()
    absorptance[:,1:] = 10**(absorptance[:,1:])
    for i in range(1,4):
        absorptance[:,i] = absorptance[:,i]/ \
            10**(-d_mac_max(10)*VisualData.macula[:,1]/.35 -
                 VisualData.ocular_sum_32[:,1])
        absorptance[:,i] = absorptance[:,i]/absorptance[:,i].max()
    return absorptance

def absorbance_from_lms10q():
    """
    Compute the absorbance from quantal lms 10 for reference.
    """
    absorbance = absorptance_from_lms10q(VisualData.lms10_log_quant)
    absorbance[:,1] = np.log10(1 - absorbance[:,1] * \
                                   (1 - 10**-d_LM_max(10))) / \
                                   -d_LM_max(10)
    absorbance[:,2] = np.log10(1 - absorbance[:,2] * \
                                   (1 - 10**-d_LM_max(10))) / \
                                   -d_LM_max(10)
    absorbance[:,3] = np.log10(1 - absorbance[:,3] * \
                                   (1 - 10**-d_S_max(10))) / \
                                   -d_S_max(10)
    return absorbance

#==============================================================================
# Functions of age and field size
#==============================================================================

def chromaticity_interpolated(field_size):
    """
    Compute the spectral chromaticity coordinates by interpolation for
    reference.
    
    Parameters
    ----------
    field_size : float
        The field size in degrees.
         
    Returns
    -------
    chromaticity : ndarray
        The xyz chromaticities, with wavelenghts in first column.
    """
    alpha = (field_size - 2)/8.
    knots = (1 - alpha)*VisualData.cc31knots + alpha*VisualData.cc64knots
    lambd = np.arange(360, 831)

    lambda31_func = scipy.interpolate.interp1d(knots, VisualData.cc31knots)
    lambda64_func = scipy.interpolate.interp1d(knots, VisualData.cc64knots)
    lambda31 = lambda31_func(lambd)
    lambda64 = lambda64_func(lambd)
    # x values
    cc31x_func = scipy.interpolate.interp1d(VisualData.cc31[:,0],
                                            VisualData.cc31[:,1],
                                            kind='cubic')
    cc64x_func = scipy.interpolate.interp1d(VisualData.cc64[:,0],
                                            VisualData.cc64[:,1],
                                            kind='cubic')
    cc31x = cc31x_func(lambda31)
    cc64x = cc64x_func(lambda64)
    xvalues = (1-alpha)*cc31x + alpha*cc64x
    # y values
    cc31y_func = scipy.interpolate.interp1d(VisualData.cc31[:,0],
                                            VisualData.cc31[:,2],
                                            kind='cubic')
    cc64y_func = scipy.interpolate.interp1d(VisualData.cc64[:,0],
                                            VisualData.cc64[:,2],
                                            kind='cubic')
    cc31y = cc31y_func(lambda31)
    cc64y = cc64y_func(lambda64)
    yvalues = (1-alpha)*cc31y + alpha*cc64y
    zvalues = 1 - xvalues - yvalues
    return np.concatenate((np.reshape(lambd, (471,1)),
                           np.reshape(xvalues, (471,1)),
                           np.reshape(yvalues, (471,1)),
                           np.reshape(zvalues, (471,1))), 1)

def ocular(age):
    """
    The optical density of the ocular media as a function of age.
    
    Computes a weighted average of docul1 and docul2.
    
    Parameters
    ----------
    age : float
        Age in years.
        
    Returns
    -------
    ocular : ndarray
        The optical density of the ocular media with wavelength in first column.
    """
    ocul = VisualData.docul2_fine.copy()
    if age < 60:
        ocul[:,1] = (1 + 0.02*(age - 32)) * VisualData.docul1_fine[:,1] + \
            VisualData.docul2_fine[:,1]
    else:
        ocul[:,1] = (1.56 + 0.0667*(age - 60)) * VisualData.docul1_fine[:,1] + \
            VisualData.docul2_fine[:,1]
    return ocul

def d_mac_max(field_size):
    """
    Maximum optical density of the macular pigment (function of field size).
    
    Parameters
    ----------
    field_size : float
        Field size in degrees.

    Returns
    -------
    d_mac_max : float
        Maximum optical density of the macular pigment.
    """
    return my_round(0.485*np.exp(-field_size/6.132), 3)

def d_LM_max(field_size):
    """
    Maximum optical density of the visual pigment (function of field size).
    
    Parameters
    ----------
    field_size : float
        Field size in degrees.

    Returns
    -------
    d_LM_max : float
        Maximum optical density of the visual pigment.
    """
    return my_round(0.38 + 0.54*np.exp(-field_size/1.333), 3)

def d_S_max(field_size):
    """
    Maximum optical density of the visual pigment (function of field size).
    
    Parameters
    ----------
    field_size : float
        Field size in degrees.

    Returns
    -------
    d_S_max : float
        Maximum optical density of the visual pigment.            
    """
    return my_round(0.30 + 0.45*np.exp(-field_size/1.333), 3)

def absorpt(field_size):
    """
    Compute quantal absorptance as a function of field size.
    
    Parameters
    ----------   
    field_size : float
        Field size in degrees.
        
    Returns
    -------
    absorpt : ndarray
        The computed lms functions, with wavelengths in first column.
    """
    abt = VisualData.absorbance.copy()
    abt[:,1] = 1 - 10**(-d_LM_max(field_size)*10**(VisualData.absorbance[:,1])) # L
    abt[:,2] = 1 - 10**(-d_LM_max(field_size)*10**(VisualData.absorbance[:,2])) # M
    abt[:,3] = 1 - 10**(-d_S_max(field_size)*10**(VisualData.absorbance[:,3]))  # S
    return abt

def lms_quantal(field_size, age):
    """
    Compute quantal cone fundamentals as a function of field size and age.
    
    Parameters
    ----------   
    field_size : float
        Field size in degrees.
    age : float
        Age in years.

    Returns
    -------
    lms : ndarray
        The computed lms functions, with wavelengths in first column.
    """
    abt = absorpt(field_size)
    lmsq = abt.copy()
    ocul = ocular(age)
    for i in range(1,4):
        lmsq[:,i] = abt[:,i] * \
            10**(-d_mac_max(field_size)*VisualData.macula[:,1]/.35 - ocul[:,1])
        lmsq[:,i] = lmsq[:,i]/(lmsq[:,i].max())
    return lmsq

def lms_energy(field_size, age):
    """
    Compute energy cone fundamentals as a function of field size and age.
    
    Parameters
    ----------   
    field_size : float
        Field size in degrees.
    age : float
        Age in years.

    Returns
    -------
    lms : ndarray
        The computed lms functions, with wavelengths in first column.
    lms_max : ndarray
        Max values of the lms functions before renormalisation.
    """
    if age == 32:
        if field_size == 2:
            return VisualData.lms2_lin_energ.copy(), 0  # dummy max value
        elif field_size == 10:
            return VisualData.lms10_lin_energ.copy(), 0 # dummy max value
    lms = lms_quantal(field_size, age)
    lms_max = []
    for i in range(1,4):
        lms[:,i] = lms[:,i]*lms[:,0]
        lms_max.append(lms[:,i].max())
        lms[:,i] = lms[:,i]/lms[:,i].max()    
    return significant_digits(lms, 9), np.array(lms_max)

def v_lambda_quantal(field_size, age):
    """
    Compute the V(lambda) function as a function of field size and age.
    
    Parameters
    ----------
    field_size : float
        Field size in degrees.
    age : float
        Age in years.
        
    Returns
    -------
    v_lambda : ndarray
        The computed v_lambda function, with wavelengths in first column.
    """
    lms = lms_quantal(field_size, age)
    v_lambda = np.zeros((np.shape(lms)[0], 2))
    v_lambda[:,0] = lms[:,0]
    v_lambda[:,1] = 1.89*lms[:,1] + lms[:,2]
    v_lambda[:,1] = v_lambda[:,1]/v_lambda[:,1].max()
    return v_lambda

def v_lambda_energy_from_quantal(field_size, age):
    """
    Compute the V(lambda) function as a function of field size and age.
    
    Starting from quantal V(lambda).
    
    Parameters
    ----------
    field_size : float
        Field size in degrees.
    age : float
        Age in years.
        
    Returns
    -------
    v_lambda : ndarray
        The computed v_lambda function, with wavelengths in first column.
    """
    if age == 32:
        if field_size == 2:
            return VisualData.vlambdaLM_2_log_quant.copy()
        elif field_size == 10:
            return VisualData.vlambdaLM_10_log_quant.copy()
    v_lambda = v_lambda_quantal(field_size, age)
    v_lambda[:,1] = v_lambda[:,1]*v_lambda[:,0]
    v_lambda[:,1] = v_lambda[:,1]/v_lambda[:,1].max()
    return v_lambda

def v_lambda_energy_from_lms(field_size, age):
    """
    Compute the V(lambda) function as a function of field size and age.
    
    Starting from engergy scale LMS.
    
    Parameters
    ----------
    field_size : float
        Field size in degrees.
    age : float
        Age in years.
        
    Returns
    -------
    v_lambda : ndarray
        The computed v_lambda function, with wavelengths in first column.
    weights : ndarray
        The two weighting factors in V(lambda) = a21*L(lambda) + \
                                                 a22*M(lambda)
    """
    if age == 32:
        if field_size == 2:
            return VisualData.vlambdaLM_2_lin_energ.copy(), \
            np.array([0.68990272, 0.34832189])
        elif field_size == 10:
            return VisualData.vlambdaLM_10_lin_energ.copy(), \
            np.array([0.69283932, 0.34967567])            
    lms, lms_max = lms_energy(field_size, age)
    v_lambda = np.zeros((np.shape(lms)[0], 2))
    v_lambda[:,0] = lms[:,0]
    v_lambda[:,1] = 1.89*lms_max[0]*lms[:,1] + lms_max[1]*lms[:,2]
    m = v_lambda[:,1].max()
    a21 = my_round(1.89*lms_max[0]/m, 8)
    a22 = my_round(lms_max[1]/m, 8)
    v_lambda[:,1] = significant_digits(a21*lms[:,1] + a22*lms[:,2], 7)
    return v_lambda, np.array([a21, a22])

def square_sum(a13, a21, a22, a33, l_spline, m_spline, s_spline, v_spline,
               lambdas, lambda_ref_min, cc_ref, full_results=False):
    """
    Function to be optimised for a13.
    
    Parameters
    ----------
    a13 : ndarray
        1x1 array with parameter to optimise.
    a21, a22, a33 : float
        Parameters in matrix for LMS to XYZ conversion.
    l_spline, m_spline, s_spline, v_spline: InterPolatedUnivariateSpline
        LMS and V(lambda)
    lambdas : ndarray
        Tabulated lambda values according to chosen resolution.
    lambda_ref_min : float
        Lambda value for x(lambda_ref_min) = x_ref_min.
    cc_ref : ndarray
        Tabulated reference chromaticity coordinates at 1 nm steps.
    full_results : bool
        Return all or just the computed error.
    
    Returns
    -------
    err : float
        Computed error.
    trans_mat : ndarray
        Transformation matrix.
    ok : bool
        Hit the correct minimum wavelength.
    """
    # Stripping reference values according to Stockman-Sharpe
    cc_ref_trunk = cc_ref[30:,1:].T.copy()
    x_ref_min = cc_ref_trunk[0,:].min()
    # Computed by Mathematica, don't ask...:
    a11 = (-m_spline(lambda_ref_min)*v_spline(lambdas).sum() +
          a13*(s_spline(lambda_ref_min)*m_spline(lambdas).sum() -
          m_spline(lambda_ref_min)*s_spline(lambdas).sum())*(-1 +
          x_ref_min) + (a21*l_spline(lambda_ref_min) +
          a33*s_spline(lambda_ref_min))*m_spline(lambdas).sum()*x_ref_min +
          m_spline(lambda_ref_min)*(a22*m_spline(lambdas).sum() +
          v_spline(lambdas).sum())*x_ref_min) / ((m_spline(lambda_ref_min)*
          l_spline(lambdas).sum() - l_spline(lambda_ref_min) *
          m_spline(lambdas).sum()) * (-1 + x_ref_min))
    a12 = (l_spline(lambda_ref_min)*v_spline(lambdas).sum() -
          a13*(s_spline(lambda_ref_min)*l_spline(lambdas).sum() -
          l_spline(lambda_ref_min)*s_spline(lambdas).sum())*(-1 +
          x_ref_min) - ((a21*l_spline(lambda_ref_min) +
          a22*m_spline(lambda_ref_min) + a33*s_spline(lambda_ref_min)) *
          l_spline(lambdas).sum() +
          l_spline(lambda_ref_min)*v_spline(lambdas).sum())*x_ref_min) / \
          ((m_spline(lambda_ref_min)*
          l_spline(lambdas).sum() - l_spline(lambda_ref_min) *
          m_spline(lambdas).sum()) * (-1 + x_ref_min))
    a11 = my_round(a11[0], 8)
    a12 = my_round(a12[0], 8)
    a13 = my_round(a13[0], 8)
    trans_mat = np.array([[a11, a12, a13], [a21, a22, 0], [0, 0, a33]])
    lms = np.array([l_spline(np.arange(390, 831)),
                    m_spline(np.arange(390, 831)),
                    s_spline(np.arange(390, 831))])
    xyz = np.dot(trans_mat, lms)
    xyz = significant_digits(xyz, 7)
    cc = np.array([xyz[0,:]/(xyz[0,:] + xyz[1,:] + xyz[2,:]),
                   xyz[1,:]/(xyz[0,:] + xyz[1,:] + xyz[2,:]),
                   xyz[2,:]/(xyz[0,:] + xyz[1,:] + xyz[2,:])])
#    cc = my_round(cc, 6) # Skip!!!
    err = ((cc - cc_ref_trunk)**2).sum()
    lambda_found = np.arange(390, 831)[cc[0,:].argmin()]
    ok = (lambda_found == lambda_ref_min)
    if full_results:
        return err, trans_mat, lambda_found, ok
    else:
        return err

def xyz(field_size, age, resolution=1):
    """
    Compute XYZ as a function of fiels size and age.
    
    Parameters
    ----------
    field_size : float
        Field size in degrees.
    age : float
        Age in years.
    resolution : float
        Resolution of tabulated results in nm.
        
    Returns
    -------
    xyz : ndarray
        The computed colour matching functions.
    cc : ndarray
        The chromaticity coordinates.
    lms : ndarray
        The computed LMS functions at the given resolution.
    mat : ndarray
        The 3x3 matrix for converting from LMS to XYZ.
    lambda_min : int
        The wavelength of minimum x chromaticity value.
    """
    lms, tmp = lms_energy(field_size, age)
    v_lambda, weights = v_lambda_energy_from_lms(field_size, age)

    l_spline = scipy.interpolate.InterpolatedUnivariateSpline(lms[:,0], lms[:,1])
    m_spline = scipy.interpolate.InterpolatedUnivariateSpline(lms[:,0], lms[:,2])
    s_spline = scipy.interpolate.InterpolatedUnivariateSpline(lms[:,0], lms[:,3])
    v_spline = scipy.interpolate.InterpolatedUnivariateSpline(v_lambda[:,0],
                                                              v_lambda[:,1])
    lambdas = np.arange(390, 830 + resolution, resolution)
    s_values = s_spline(lambdas)
    v_values = v_spline(lambdas)

    a21 = weights[0]
    a22 = weights[1]
    a33 = my_round(v_values.sum() / s_values.sum(), 8)

    cc_ref = chromaticity_interpolated(field_size)
    ok = False
    lambda_ref_min = 500
    while not ok:
        a13 = scipy.optimize.fmin(square_sum, 0.39,
                                  (a21, a22, a33, l_spline, m_spline, s_spline,
                                   v_spline, lambdas, lambda_ref_min, cc_ref),
                                   xtol=1e-10, disp=False)
        err, trans_mat, lambda_ref_min, ok = \
            square_sum(a13, a21, a22, a33, l_spline, m_spline,
                       s_spline, v_spline, lambdas,
                       lambda_ref_min, cc_ref, True)
    lms = np.array([l_spline(lambdas),
                    m_spline(lambdas),
                    s_spline(lambdas)])
    xyz = np.dot(trans_mat, lms)
    xyz = significant_digits(xyz, 7)
    cc = np.array([xyz[0,:]/(xyz[0,:] + xyz[1,:] + xyz[2,:]),
                   xyz[1,:]/(xyz[0,:] + xyz[1,:] + xyz[2,:]),
                   xyz[2,:]/(xyz[0,:] + xyz[1,:] + xyz[2,:])])
    cc = my_round(cc, 5)
    # Reshape
    lms = np.concatenate((lambdas.reshape((1,len(lambdas))), lms)).T
    xyz = np.concatenate((lambdas.reshape((1,len(lambdas))), xyz)).T
    cc = np.concatenate((lambdas.reshape((1,len(lambdas))), cc)).T
    return xyz, cc, lms, trans_mat, lambda_ref_min

def projective_lms_to_cc_matrix(trans_mat):
    """
    Compute the matrtix for the projective transformation from lms to cc.
    
    Parameters
    ----------
    trans_mat : ndarray
        Transformation matrix from lms to xyz.
    
    Returns
    -------
    mat : ndarray
        Transformation matrix directly from lms to cc.
    """
    mat = trans_mat.copy()
    mat[2,0] = trans_mat[0,0] + trans_mat[1,0] + trans_mat[2,0]
    mat[2,1] = trans_mat[0,1] + trans_mat[1,1] + trans_mat[2,1]
    mat[2,2] = trans_mat[0,2] + trans_mat[1,2] + trans_mat[2,2]
    return mat

def boynton_macleod(lms, v_lambda, trans_mat):
    """
    Compute the Boynton-MacLeod diagram from lms and the transformation.
    
    Parameters
    ----------
    lms : ndarray
        LMS functions.
    trans_mat : ndarray
        Transformation matrix from LMS to XYZ.
        
    Returns
    -------
    bm : ndarray
        The Boynton-MacLeod data.
    """
    bm = lms.copy()
    bm[:,1] = trans_mat[1,0]*lms[:,1]/v_lambda[:,1]
    bm[:,2] = trans_mat[1,1]*lms[:,2]/v_lambda[:,1]
    bm[:,3] = lms[:,3]/v_lambda[:,1]
    m = bm[:,3].max()
    bm[:,3] = bm[:,3]/m
    return bm    
    
#==============================================================================
# For testing purposes only
#==============================================================================

if __name__ == '__main__':
    xyz, cc, lms, trans_mat, lambda_ref_min = xyz(4, 32, .1)
