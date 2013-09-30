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

colour.metric.test()

# sc = 5
# d = colour.data.build_d_regular(colour.space.cielab, [50], pl.arange(-10, 11) * sc, pl.arange(-10,11) * sc)
# gab = colour.tensor.dE_ab(d)
# sp = colour.space.lgj_e
# col = d.get_linear(sp)
# pl.plot(col[:,1], col[:,2], '.')
# colour.misc.plot_ellipses(gab.get_ellipses(sp, gab.plane_ab, scale=sc))
# pl.axis('equal')
# pl.grid()
# pl.show()

bfd = colour.data.build_g_BFD()
locus = colour.data.build_d_XYZ_64().get_linear(colour.space.xyY)
gE = colour.tensor.dE_E(bfd.points)
pts = bfd.points.get_linear(colour.space.lgj_e)
pl.plot(pts[:,1], pts[:,2], '.')
colour.misc.plot_ellipses(bfd.get_ellipses(colour.space.lgj_e, bfd.plane_ab, scale=2.5))
# colour.misc.plot_ellipses(gE.get_ellipses(colour.space.lgj_e, gE.plane_ab, scale=2.5), edgecolor=[1,0,0])
pl.axis('equal')
pl.grid()
pl.show()
