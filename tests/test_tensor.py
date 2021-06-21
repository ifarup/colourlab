#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_tensor:  Unittests for all functions in the tensor module.

Copyright (C) 2017 Ivar Farup, Lars Niebuhr, Sahand Lahafdoozian,
Nawar Behenam, Jakob Voigt

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
from colourlab import space, data, tensor
import numpy as np

d = data.d_regular(space.cielab, np.linspace(1, 100, 10),
                   np.linspace(-100, 100, 21), np.linspace(-100, 100,
                                                           21))
ndat = np.shape(d.get_flattened(space.cielab))[0]
gab = tensor.dE_ab(d)
guv = tensor.dE_uv(d)
g00 = tensor.dE_00(d)
gE = tensor.dE_E(d)
gD = tensor.poincare_disk(space.TransformPoincareDisk(space.cielab, R=100), d)
gDIN99 = tensor.dE_DIN99(d)
gDIN99b = tensor.dE_DIN99b(d)
gDIN99c = tensor.dE_DIN99c(d)
gDIN99d = tensor.dE_DIN99d(d)
dat = data.Points(space.cielch, [50, 10, np.pi/4])
Gamma = tensor.christoffel(space.cielch,
                           lambda x : tensor.polar(space.cielch, x),
                           dat)
Gamma_expected = np.array([[[0, 0, 0],    
                            [0, 0, 0],
                            [0, 0, 0]],
                           [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, -10]],
                           [[0, 0, 0],
                            [0,  0, .1],
                            [0,  .1, 0]]])

# Tests

class TestTensor(unittest.TestCase):

    def testShapes(self):
        self.assertEqual(np.shape(gab.get(space.xyz)), (ndat, 3, 3))
        self.assertEqual(np.shape(guv.get(space.xyz)), (ndat, 3, 3))
        self.assertEqual(np.shape(gD.get(space.xyz)),  (ndat, 3, 3))
        self.assertEqual(np.shape(g00.get(space.xyz)), (ndat, 3, 3))
        self.assertEqual(np.shape(gE.get(space.xyz)),  (ndat, 3, 3))
        self.assertEqual(np.shape(gDIN99.get(space.xyz)), (ndat, 3, 3))
        self.assertEqual(np.shape(gDIN99b.get(space.xyz)), (ndat, 3, 3))
        self.assertEqual(np.shape(gDIN99c.get(space.xyz)), (ndat, 3, 3))
        self.assertEqual(np.shape(gDIN99d.get(space.xyz)), (ndat, 3, 3))
        self.assertAlmostEqual(np.linalg.norm(Gamma - Gamma_expected), 0, delta=1e-4)
