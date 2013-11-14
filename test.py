#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test: Test various features of the colour package (continuously updated)

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
import colour
import matplotlib.pyplot as plt
import numpy as np

pd = colour.space.TransformPoincareDisk(colour.space.cielab, 5)
d0 = colour.data.Data(colour.space.cielab, np.array([50, 0, 0]))
d1 = colour.data.Data(colour.space.cielab, np.array([50, 0, 10]))
d2 = colour.data.Data(colour.space.cielab, np.array([50, 10, 0]))
print colour.metric.poincare_disk(pd, d0, d1)
print colour.metric.poincare_disk(pd, d0, d2)
print colour.metric.poincare_disk(pd, d1, d2)

step = 20
d = colour.data.build_d_regular(colour.space.cielab, [50], np.arange(-100, 100 + step, step), np.arange(-100, 100 + step, step))
gab = colour.tensor.dE_ab(d)
gp = colour.tensor.poincare_disk(pd, d)
col = d.get(colour.space.cielab)
plt.figure(1)
plt.plot(col[:,1], col[:,2] ,'.')
plt.grid()
plt.axis('equal')
colour.misc.plot_ellipses(gab.get_ellipses(colour.space.cielab, gab.plane_ab, step))
colour.misc.plot_ellipses(gp.get_ellipses(colour.space.cielab, gab.plane_ab, step), edgecolor=[1,0,0])
plt.figure(2)
col = d.get(pd)
plt.plot(col[:,1], col[:,2] ,'.')
plt.axis('equal')
colour.misc.plot_ellipses(gab.get_ellipses(pd, gab.plane_ab, step))
colour.misc.plot_ellipses(gp.get_ellipses(pd, gab.plane_ab, step), edgecolor=[1,0,0])
plt.show()

