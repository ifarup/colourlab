#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test: Test various features of the colour package (continuously updated)

Copyright (C) 2016 Ivar Farup

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
import colour
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dat1 = colour.data.Data(colour.space.cielab, [50, 0, 0])
dat2 = colour.data.Data(colour.space.cielab, [51, 1, 1])
vec = colour.data.VectorData(colour.space.cielab, dat1, [1, 1, 1])

print(vec.vectors[colour.space.xyz])
print(dat2.get(colour.space.xyz) - dat1.get(colour.space.xyz))

dat = colour.data.d_regular(colour.space.srgb,
                            np.linspace(0, 1, 5),
                            np.linspace(0, 1, 5),
                            np.linspace(0, 1, 5))

gamut = colour.data.Gamut(colour.space.cielab, dat)
for i in range(gamut.hull.simplices.shape[0]):
    tri = art3d.Poly3DCollection([gamut.hull.points[gamut.hull.simplices[i]]])
    ax.add_collection(tri)
ax.set_xlim([0, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([-100, 100])
plt.show()
