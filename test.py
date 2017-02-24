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

im_rgb = plt.imread('lena.png')
im = colour.data.Data(colour.space.srgb, im_rgb)

sp = colour.space.cielab
g = colour.tensor.dE_ab(im)

di = im.dip(sp)
dj = im.djp(sp)

di2 = g.inner(sp, di, di)
dj2 = g.inner(sp, dj, dj)
didj = g.inner(sp, di, dj)

s11, s12, s22 = g.structure_tensor(colour.space.cielab)
d11, d12, d22 = g.diffusion_tensor(colour.space.srgb, 1e-8)
