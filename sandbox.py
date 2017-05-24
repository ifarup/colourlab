#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sandbox: Play with various features of the colour package (continuously updated)

Copyright (C) 2012-2017 Ivar Farup

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
import colourspace as col
import matplotlib.pyplot as plt

# im_rgb = plt.imread('lena.png')
# im = col.image.Image(col.space.srgb, im_rgb)
# img = im.get(col.space.cielab)[..., 0] / 100
# d11, d12, d22 = im.diffusion_tensor(col.space.cielab)

# for i in range(100):
#     print(i)
#     gi = col.misc.dip(img)
#     gj = col.misc.djp(img)
#     gii = col.misc.dim(d11 * gi + d12 * gj)
#     gjj = col.misc.djm(d12 * gi + d22 * gj)
#     tv = gii + gjj
#     img += .25 * tv

# plt.imshow(img, plt.cm.gray)
# plt.show()

l = np.array([[0, 0, 0], [3, 3, 3]])
p = np.array([1, 1, 2])
print(col.gamut.Gamut.in_line(l, p))
