#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_data: Unittests for all functions in the data module.

Copyright (C) 2013-2017 Ivar Farup, Lars Niebuhr,
Sahand Lahafdoozian, Nawar Behenam, Jakob Voigt

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

import unittest
import numpy as np
from colour import data, space

# Global constants for use in the tests

col1 = np.array([.5, .5, .5])
col2 = np.array([[.5, .5, .5]])
col3 = np.array([[1e-10, 1e-10, 1e-10],
                 [.95, 1., 1.08],
                 [.5, .5, .5]])
col4 = np.array([[[1e-10, 1e-10, 1e-10],
                  [.95, 1., 1.08],
                  [.5, .5, .5]],
                 [[1e-10, 1e-10, 1e-10],
                  [.95, 1., 1.08],
                  [.5, .5, .5]]])

d1 = data.Data(space.xyz, col1)
d2 = data.Data(space.xyz, col2)
d3 = data.Data(space.xyz, col3)
d4 = data.Data(space.xyz, col4)


class TestData(unittest.TestCase):

    def test_get(self):
        self.assertEqual(d1.get(space.xyz).shape, (3, ))
        self.assertEqual(d2.get(space.xyz).shape, (1, 3))
        self.assertEqual(d3.get(space.xyz).shape, (3, 3))
        self.assertEqual(d4.get(space.xyz).shape, (2, 3, 3))

    def test_get_linear(self):
        self.assertEqual(d1.get_linear(space.xyz).shape, (1, 3))
        self.assertEqual(d2.get_linear(space.xyz).shape, (1, 3))
        self.assertEqual(d3.get_linear(space.xyz).shape, (3, 3))
        self.assertEqual(d4.get_linear(space.xyz).shape, (6, 3))

    def test_implicit_convert(self):
        lab1 = d1.get(space.cielab)
        lab2 = d2.get(space.cielab)
        lab3 = d3.get(space.cielab)
        lab4 = d4.get(space.cielab)
        dd1 = data.Data(space.cielab, lab1)
        dd2 = data.Data(space.cielab, lab2)
        dd3 = data.Data(space.cielab, lab3)
        dd4 = data.Data(space.cielab, lab4)
        self.assertTrue(np.max(np.abs(col1 - dd1.get(space.xyz))) < 1e-11)
        self.assertTrue(np.max(np.abs(col2 - dd2.get(space.xyz))) < 1e-11)
        self.assertTrue(np.max(np.abs(col3 - dd3.get(space.xyz))) < 1e-11)
        self.assertTrue(np.max(np.abs(col4 - dd4.get(space.xyz))) < 1e-11)

    def test_read_data_files(self):
        for func in [data.d_XYZ_31, data.d_XYZ_64, data.d_Melgosa]:
            d = func()
            self.assertIsInstance(d, data.Data)
        for arg in ['all', 'real', '1929']:
            d, n, l = data.d_Munsell(arg)
            self.assertIsInstance(n, list)
            self.assertIsInstance(l, np.ndarray)
            self.assertIsInstance(d, data.Data)

    def test_d_regular(self):
        d = data.d_regular(space.xyz, np.linspace(0, 1, 10),
                           np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        self.assertIsInstance(d, data.Data)
        dd = d.get(space.xyz)
        self.assertEqual(dd.shape, (1000, 3))


if __name__ == '__main__':
    unittest.main()
