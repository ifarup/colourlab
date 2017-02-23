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

g = np.array([[0, .25, .5], [.25, .5, .75], [.5, .75, 1]])
r = np.zeros(np.shape(g))
b = np.zeros(np.shape(g))

im1_rgb = np.dstack((r,g,b))
im2_rgb = im1_rgb.copy() + .001
im2_rgb[im2_rgb > 1] = 1

im1 = colour.data.Data(colour.space.srgb, im1_rgb)
im2 = colour.data.Data(colour.space.srgb, im2_rgb)
d = im1.diff(colour.space.cielab, im2)
g = colour.tensor.euclidean(colour.space.srgb, im1)
print(g.inner(colour.space.srgb, d, d))
