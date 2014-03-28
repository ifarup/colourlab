#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test: Test various features of the colour package (continuously updated)

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
import colour
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

def xyz2lxy(params, xyz):
    a21 = params[0]
    a23 = params[1]
    a31 = params[2]
    a33 = params[3]
    gamma = params[4]
    b21 = params[5]
    b23 = params[6]
    b31 = params[7]
    b33 = params[8]
    w = colour.space.white_D65.get(colour.space.xyz)
    wx = w[0]
    wz = w[2]
    A = np.array([[0, 1, 0],
                  [a21, 1 - a21 * wx - a23 * wz, a23],
                  [a31, 1 - a31 * wx - a33 * wz, a33]])
    B = np.array([[1, 0, 0],
                  [b21, - b21 - b23, b23],
                  [b31, - b31 - b33, b33]])
    u = np.dot(A, xyz.T)
    v = np.sign(u) * np.abs(u)**gamma
    return np.dot(B, v).T

def cost_function(params, xyz, m_lab):
    lxy = xyz2lxy(params, xyz)
    cost = colour.statistics.dataset_distance(lxy, m_lab).sum()
    return cost

m_data, m_names, m_lab = colour.data.d_Munsell('1929')
m_corr = m_data.new_white_point(colour.space.ciecat02, colour.space.white_C, colour.space.white_D65)
xyz = m_corr.get(colour.space.xyz)
cost_function(np.array([1,0,0,1,.43,5,0,0,-2]), xyz, m_lab)
params = scipy.optimize.fmin(cost_function, np.array([1,0,0,1,.43,5,0,0,-2]), (xyz, m_lab))
print cost_function(params, xyz, m_lab)
print params
lxy = xyz2lxy(params,xyz)
print xyz2lxy(params,colour.space.white_D65.get(colour.space.xyz) * .4)
plt.plot(m_lab[:,1], m_lab[:,2] , 'k.')
plt.plot(lxy[:,1], lxy[:,2], '.')
plt.grid()
plt.axis('equal')
plt.figure()
plt.plot(np.sqrt(m_lab[:,1]**2 + m_lab[:,2]**2), m_lab[:,0] , 'k.')
plt.plot(np.sqrt(lxy[:,1]**2 + lxy[:,2]**2), lxy[:,0], '.')
plt.grid()
plt.axis('equal')
plt.show()
