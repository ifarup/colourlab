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

dinx = colour.space.TransformLogCompressL(colour.space.cielab, 30., .37)
cielab = colour.space.cielab
step = 10
d = colour.data.build_d_regular(colour.space.cielab, [50], pl.arange(-80, 81, step), pl.arange(-80, 81, step))
g = colour.tensor.dE_ab(d)
lab = d.get(cielab)
din = d.get(colour.space.lgj_e)
pl.plot(din[:,1], din[:,2], '.')
colour.misc.plot_ellipses(g.get_ellipses(colour.space.lgj_e, g.plane_ab, scale=step))
pl.axis('equal')
pl.show()