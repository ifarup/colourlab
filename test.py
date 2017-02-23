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

p = colour.data.Data(colour.space.srgb, np.array([.5, .5, .5]))
t = colour.data.TensorData(colour.space.srgb, p, np.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]]))
print(t.get(colour.space.cielab))