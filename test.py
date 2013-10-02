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
import pylab as pl

step = 10
d = colour.data.build_d_regular(colour.space.cielab, [70], pl.arange(-50, 51, step), pl.arange(-50, 51, step))
g = colour.tensor.dE_DIN99d(d)
lab = d.get(colour.space.cielab)
pl.plot(lab[:,1], lab[:,2], '.')
colour.misc.plot_ellipses(g.get_ellipses(colour.space.cielab, g.plane_ab, scale=1))
pl.axis('equal')
pl.show()