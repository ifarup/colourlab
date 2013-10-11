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

step = 20
d1 = colour.data.build_d_regular(colour.space.cielab, [50], np.arange(-100, 100 + step, step), [0])
d2 = colour.data.Data(colour.space.cielab, d1.get(colour.space.cielab) + np.array([0,0,.01]))
sp = colour.space.TransformPoincareDisk(colour.space.cielab, 10)
pdiff = colour.metric.poincare_disk(sp, d1, d2)
lindiff = colour.metric.linear(sp, d1, d2, lambda dat: colour.tensor.poincare_disk(sp, dat))
plt.plot(pdiff, lindiff)
plt.axis('equal')
plt.grid()
plt.show()