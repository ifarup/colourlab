#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_statistics: Unittests for all functions in the statistics module.

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
import numpy as np
from colourlab import data, space, metric, tensor, statistics

# Data

d1 = data.d_regular(space.cielab,
                    np.linspace(0, 100, 10),
                    np.linspace(-100, 100, 21),
                    np.linspace(-100, 100, 21))
d2 = data.Points(space.cielab,
                 d1.get(space.cielab) + 1)
diff = metric.dE_ab(d1, d2)

d3 = data.d_regular(space.cielab,
                    np.linspace(0, 100, 3),
                    np.linspace(-100, 100, 3),
                    np.linspace(-100, 100, 3))

t3 = tensor.dE_ab(d3)
t4 = data.Tensors(space.cielab,
                  t3.get(space.cielab) * 2,
                  t3.points)

R, scale = statistics.pant_R_values(space.cielab, t3, t4)
R_plane, scale = statistics.pant_R_values(space.cielab, t3, t4, plane=t3.plane_01)
R_nonopt, scale = statistics.pant_R_values(space.cielab, t3, t4, optimise=False)

dist = statistics.minimal_dataset_distance(d1.get_flattened(space.cielab), d1.get_flattened(space.cielab))[0]

# Tests

class TestStatistics(unittest.TestCase):

    def testStress(self):
        self.assertEqual(statistics.stress(diff, diff)[0], 0)
        self.assertTrue(statistics.stress(diff, diff + 1)[0] < 1e-11)

    def testPant(self):
        self.assertTrue(np.max(np.abs(1 - R)) < 1e-4)
        self.assertTrue(np.max(np.abs(1 - R_plane)) < 1e-4)
        self.assertTrue(np.max(np.abs(.5 - R_nonopt)) < 1e-4)
        

    def testMinimalDistance(self):
        self.assertTrue(np.max(dist) < 1e-4)
