#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test: Test various features of the colour package (continuously updated)

Copyright (C) 2014 Ivar Farup

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
from mpl_toolkits.mplot3d import Axes3D

# Points defined in the sRGB colour space:
rgbpoints = np.array([[.25, .25, .25],
                      [.9, .25, .25],
                      [.25, .9, .25],
                      [.25, .25, .9],
                      [.9, .9, .25],
                      [.9, .25, .9],
                      [.25, .9, .9],
                      [1, 1, 1]])

# Colour data object (any colour space):
points = colour.data.Data(colour.space.srgb, rgbpoints)

# Compute the gamut of the points in XYZ:
gamut = colour.data.Gamut(colour.space.xyz, points)

# Plot the gamut in CIELAB
fig = plt.figure()
ax = fig.gca(projection='3d')
labpoints = points.get_linear(colour.space.cielab)
ax.plot_trisurf(labpoints[:, 1], labpoints[:, 2], labpoints[:, 0],
                triangles=gamut.simplices)
plt.show()
