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

from colour import *

xyz = build_d_XYZ_31()
gMA = build_g_MacAdam()
p = gMA.points.get(spaceCIELAB)
plt.clf()
plt.plot(p[:,1], p[:,2], '.')
plot_ellipses(gMA.get_ellipses(spaceCIELAB, plane=gMA.plane_ab, scale=10))
plt.show()