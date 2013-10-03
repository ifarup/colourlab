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

d = colour.data.build_d_regular(colour.space.srgb, pl.linspace(.001, 1, 6), pl.linspace(.001, 1, 6), [.5])
g = colour.tensor.euclidean(colour.space.srgb, d)
rgb = d.get(colour.space._srgb_linear)
pl.plot(rgb[:,0], rgb[:,1] ,'x')
colour.misc.plot_ellipses(g.get_ellipses(colour.space._srgb_linear, g.plane_01, .2))
pl.grid()
pl.axis('equal')
pl.show()
