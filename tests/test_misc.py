#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_misc: Unittests for all functions in the misc module.

Copyright (C) 2017 Ivar Farup

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import unittest
import matplotlib
import matplotlib.pyplot as plt
from colourlab import misc, space, data

t = data.g_MacAdam()
ell = t.get_ellipses(space.xyY)
_, ax = plt.subplots()
misc.plot_ellipses(ell, ax)
misc.plot_ellipses(ell)

class TestPlot(unittest.TestCase):

    def test_plot(self):
        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))
