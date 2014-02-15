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

col1d = np.array([50, 0, 0])
col2d = np.array([[50, 0, 0], [25, 10, 10]])
col3d = np.array([[[50, 0, 0], [25, 10, 10]],
                  [[50, 0, 0], [25, 10, 10]]])
col4d = np.array([[[[50, 0, 0], [25, 10, 10]],
                  [[50, 0, 0], [25, 10, 10]]],
                  [[[50, 0, 0], [25, 10, 10]],
                  [[50, 0, 0], [25, 10, 10]]]])

col1d_1 = colour.data.Data(colour.space.cielab, col1d)
col1d_2 = colour.data.Data(colour.space.cielab, col1d + 1)
col2d_1 = colour.data.Data(colour.space.cielab, col2d)
col2d_2 = colour.data.Data(colour.space.cielab, col2d + 1)
col3d_1 = colour.data.Data(colour.space.cielab, col3d)
col3d_2 = colour.data.Data(colour.space.cielab, col3d + 1)
col4d_1 = colour.data.Data(colour.space.cielab, col4d)
col4d_2 = colour.data.Data(colour.space.cielab, col4d + 1)

diff1d = colour.metric.dE_ab(col1d_1, col1d_2)
diff2d = colour.metric.dE_ab(col2d_1, col2d_2)
diff3d = colour.metric.dE_ab(col3d_1, col3d_2)
diff4d = colour.metric.dE_ab(col4d_1, col4d_2)

print diff1d
print diff2d
print diff3d
print diff4d

print np.shape(col1d), np.shape(diff1d)
print np.shape(col2d), np.shape(diff2d)
print np.shape(col3d), np.shape(diff3d)
print np.shape(col4d), np.shape(diff4d)
