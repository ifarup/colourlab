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

dat1 = colour.data.d_regular(colour.space.cielab,
                             np.linspace(40, 70, 3),
                             np.linspace(-30, 30, 3),
                             np.linspace(-30, 30, 3))

dat2 = colour.data.d_regular(colour.space.cielab,
                             np.linspace(40, 70, 3),
                             np.linspace(-30, 30, 3) + 1,
                             np.linspace(-30, 30, 3))

tensor = colour.tensor.dE_00(dat1)

sp = colour.space.xyY
diff = dat2.get(sp) - dat1.get(sp)
t = tensor.get(sp)
for i in range(27):
    print(np.dot(np.dot(diff[i, :], t[i, ...]), diff[i, ...]))
print(colour.misc.norm_sq(diff, t))
