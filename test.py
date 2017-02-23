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

im_rgb = np.dstack((r,g,b))
im = colour.data.Data(colour.space.srgb, im_rgb)
im_lab = im.get(colour.space.cielab)

t = colour.tensor.dE_ab(im)
t_srgb = t.get(colour.space.srgb)
print(t_srgb.shape)
print(t_srgb[0, 0, ...])