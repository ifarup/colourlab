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

im_rgb = plt.imread('lena.png')
im = colour.image.Image(colour.space.srgb, im_rgb)

sp = colour.space.cielab
g = colour.tensor.dE_ab(im)

d11, d12, d22 = im.diffusion_tensor(colour.space.srgb, 1e-4, g)
d11 = np.dstack((d11, d11, d11))
d12 = np.dstack((d12, d12, d12))
d22 = np.dstack((d22, d22, d22))

for i in range(500):
    print(i)
    gx = colour.misc.dic(im_rgb)
    gy = colour.misc.djc(im_rgb)
    gxx = colour.misc.dic(d11 * gx + d12 * gy)
    gyy = colour.misc.djc(d12 * gx + d22 * gy)
    tv = gxx + gyy
    im_rgb[1:-1, 1:-1, :] = im_rgb[1:-1, 1:-1, :] + tv[1:-1, 1:-1, :]
    plt.imsave('im_%03d.png' % i, im_rgb)
