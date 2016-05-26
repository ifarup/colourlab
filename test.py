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

bfd = colour.data.g_BFD()
bfd_points = bfd.points.get(colour.space.xyY)
plt.plot(bfd_points[:, 0], bfd_points[:, 1], '.')
colour.misc.plot_ellipses(bfd.get_ellipses(colour.space.xyY, bfd.plane_xy, 1.5))
plt.axis('equal')
plt.show()
