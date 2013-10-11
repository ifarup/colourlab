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

import colour
import matplotlib.pyplot as plt
import numpy as np

step = 10
d = colour.data.build_d_regular(colour.space.cielab, [50], np.arange(-100, 100, step), np.arange(-100, 100, step))
pd = colour.space.TransformPoincareDisk(colour.space.cielab, 100)
g = colour.tensor.dE_ab(d)
sp = pd
col = d.get(sp)
plt.plot(col[:,1], col[:,2] ,'.')
plt.grid()
plt.axis('equal')
colour.misc.plot_ellipses(g.get_ellipses(sp, g.plane_ab, step))
plt.show()