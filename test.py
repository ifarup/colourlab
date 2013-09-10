#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test: Test various features of the colour package (continuosly updated)

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

import numpy as np
from matplotlib import pyplot as plt
import colour

disk = colour.TransformPoincareDisk(colour.TransformLinear(colour.spaceUi, 1.5 * np.eye(3)))

bfd = colour.build_g_BFD('P')
p = bfd.points.get_linear(disk)

plt.clf()
plt.plot(p[:,1], p[:,2], '.')
colour.plot_ellipses(bfd.get_ellipses(disk, plane=bfd.plane_ab))
plt.axis('equal')
plt.grid()
plt.show()