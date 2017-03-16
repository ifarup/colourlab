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
img = im.get(colour.space.cielab)[..., 0] / 100
d11, d12, d22 = im.diffusion_tensor(colour.space.cielab)

for i in range(100):
    print(i)
    gi = colour.misc.dip(img)
    gj = colour.misc.djp(img)
    gii = colour.misc.dim(d11 * gi + d12 * gj)
    gjj = colour.misc.djm(d12 * gi + d22 * gj)
    tv = gii + gjj
    img += .25 * tv

plt.imshow(img, plt.cm.gray)
plt.show()
