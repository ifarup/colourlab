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

im = colour.data.Data(colour.space.srgb, .25 * np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]))
print(im.dip(colour.space.cielab).get(colour.space.cielab))
print(im.dim(colour.space.cielab).get(colour.space.cielab))
print(im.dic(colour.space.cielab).get(colour.space.cielab))
print(im.djp(colour.space.cielab).get(colour.space.cielab))
print(im.djm(colour.space.cielab).get(colour.space.cielab))
print(im.djc(colour.space.cielab).get(colour.space.cielab))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
