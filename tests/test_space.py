#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_space: Unittests for all functions in the space module.

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
from colourlab import space, data

# Spaces and data for testing

_test_ui = space.TransformLinear(
    space.TransformGamma(
        space.TransformLinear(
            space.xyz,
            np.array([[0.1551646, 0.5430763, -0.0370161],
                      [-0.1551646, 0.4569237, 0.0296946],
                      [0, 0, 0.0073215]])), .43),
    np.array([[1.1032e+00, 5.0900e-01, 5.0840e-03],
              [2.2822e+00, -4.2580e+00, 6.2844e+00],
              [9.6110e+00, -1.2199e+01, -2.3843e+00]]))
_test_space_cartesian = space.TransformCartesian(space.cieluv)
_test_space_poincare_disk = space.TransformPoincareDisk(space.cielab)
_test_space_gamma = space.TransformGamma(space.xyz, .43)

col = np.array([[1e-10, 1e-10, 1e-10], [.95, 1., 1.08], [.5, .5, .5]])


class TestSpace(unittest.TestCase):

    def test_predefined_space_transforms(self):
        test_spaces = [space.xyz, space.xyY, space.cielab,
                       space.cieluv, space.cielch, space.ipt,
                       space.din99, space.din99b, space.din99c,
                       space.din99d,
                       space.srgb,
#                       space.lgj_osa,
#                       space.lgj_e,
                       _test_space_cartesian,
                       _test_space_poincare_disk, _test_space_gamma]
        for sp in test_spaces:
            c2 = sp.to_XYZ(sp.from_XYZ(col))
            err = np.max(np.abs(col - c2))
            self.assertTrue(err < .1)

    def test_predefined_space_jacobians(self):
        col_data = data.Points(space.xyz, col)
        test_spaces = [space.xyz, space.xyY, space.cielab,
                       space.cieluv, space.cielch, space.ipt,
                       space.ciede00lab, space.din99, space.din99b,
                       space.din99c, space.din99d, space.srgb,
                       space.lgj_osa, space.lgj_e,
                       _test_space_cartesian,
                       _test_space_poincare_disk, _test_space_gamma]
        for sp in test_spaces:
            jac1 = sp.jacobian_XYZ(col_data)
            jac2 = sp.inv_jacobian_XYZ(col_data)
            prod = np.zeros(np.shape(jac1))
            for i in range(np.shape(jac1)[0]):
                prod[i] = np.dot(jac1[i], jac2[i])
                prod[i] = np.abs(prod[i] - np.eye(3))
            err = np.max(prod)
            self.assertTrue(err < 1e-6)
