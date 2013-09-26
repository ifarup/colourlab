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

d1 = colour.data.build_d_regular(colour.space.cielab, [50], [-10, 0, 10], [-10, 0, 10])
d2 = colour.data.Data(colour.space.cielab, d1.get(colour.space.cielab) + 1 / np.sqrt(3))
print colour.metric.dE_ab(d1, d2)
print colour.metric.dE_uv(d1, d2)
print colour.metric.dE_00(d1, d2)