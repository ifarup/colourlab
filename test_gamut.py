#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_gamut: Unittests for all functions in the gamut module.

Copyright (C) 2017 Lars Niebuhr, Sahand Lahafdoozian,
Nawar Behenam, Jakob Voigt

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
from colour import data, gamut, space
import matplotlib.pyplot as plt

# Global variables.
n_data = np.array([[0, 0, 0],       # 0 vertecis
                   [10, 0, 0],      # 1 vertecis
                   [10, 10, 0],     # 2 vertecis
                   [0, 10, 0],      # 3 vertecis
                   [5, 5, 5],       # 4 non vertecis
                   [4, 6, 2],       # 5 non vertecis
                   [10, 10, 10],    # 6 vertecis
                   [1, 2, 3],       # 7 non vertecis
                   [10, 0, 10],     # 8 vertecis
                   [0, 0, 10],      # 9 vertecis
                   [0, 10, 10]])    # 10 vertecis

line = np.array([[0, 0, 0], [3, 3, 3]])             # Line used in testing.
point_on_line = np.array([1, 1, 1])                 # Point inside the line to be tested.
point_not_paralell_to_line = np.array([2, 3, 2])    # Point outside the line to be tested.
point_opposite_direction_than_line = np.array([-1, -1, -1])
point_further_away_than_line = np.array([4, 4, 4])

tetrahedron = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [0, 0, 10]])     # Tetrahedron used in testing.
tetra_p_inside = np.array([2, 3, 4])               # Point inside the tetrahedron to be tested.
tetra_p_not_inside = np.array([20, 1, 2])          # Point outside the tetrahedron to be tested.
tetra_p_on_surface = np.array([0, 5, 0])

triangle = np.array([[0, 0, 0], [4, 0, 0], [0, 0, 4]])
triangle_point_inside = np.array([2, 0, 2])
triangle_point_not_coplanar = np.array([2, 2, 2])
triangle_point_coplanar_but_outside = np.array([5, 0, 3])


class TestGamut(unittest.TestCase):

    def test_gamut_initialize(self):
        # Test for convex hull
        c_data = data.Data(space.srgb, n_data)          # Generating the colour Data object
        g = gamut.Gamut(space.srgb, c_data)
        vertices = np.array([0, 1, 2, 3, 6, 8, 9, 10])  # Known indices of vertecis for the test case

        self.assertEqual(vertices.tolist(), g.vertices.tolist())    # Checking that the vertecis match

    def test_is_inside(self):   # Test for gamut.Gamut.is_inside
        c_data = data.Data(space.srgb, n_data)
        g = gamut.Gamut(space.srgb, c_data)
        points = np.ones((2, 2, 2, 3))
        c_points = data.Data(space.srgb, points)
        g.is_inside(space.srgb, c_points)

    def test_get_vertices(self):
        # Test for gamut.Gamut.get_vertices
        c_data = data.Data(space.srgb, n_data)  # Generating the colour Data object
        g = gamut.Gamut(space.srgb, c_data)
        n1_data = np.array([[0, 0, 0],      # 0 vertecis    # Array with just the vercites used for comparison.
                           [10, 0, 0],      # 1 vertecis
                           [10, 10, 0],     # 2 vertecis
                           [0, 10, 0],      # 3 vertecis
                           [10, 10, 10],    # 6 vertecis
                           [10, 0, 10],     # 8 vertecis
                           [0, 0, 10],      # 9 vertecis
                           [0, 10, 10]])    # 10 vertecis

        vertices = g.get_vertices(n_data)
        self.assertTrue(np.array_equiv(n1_data, vertices))    # Compares return array with the known vertices array.

        vertices = g.get_vertices(n_data)                     # Calls the function and add the vertices to the array.
        self.assertTrue(np.array_equiv(n1_data, vertices))    # Compares returend array with the known vertices array.

    def test_plot_surface(self):         # Test for gamut.Gamut.plot_surface
        fig = plt.figure()                          # Creates a figure
        ax = fig.add_subplot(111, projection='3d')  # Creates a 3D plot ax

        c_data = data.Data(space.srgb, n_data)      # Generating the colour Data object
        g = gamut.Gamut(space.srgb, c_data)         # Creates a new gamut
        sp = g.space                                # specifies the color space
        g.plot_surface(ax, sp)                      # Calls the plot function

    def test_in_line(self):
        c_data = data.Data(space.srgb, n_data)
        g = gamut.Gamut(space.srgb, c_data)

        self.assertFalse(False, g.in_line(line, point_not_paralell_to_line))            # Point in NOT parallel to line
        self.assertFalse(False, g.in_line(line, point_opposite_direction_than_line))    # Point opposite dir then line
        self.assertFalse(False, g.in_line(line, point_further_away_than_line))          # Point is is further then line
        self.assertTrue(True, g.in_line(line, point_on_line))                           # Point is on line

    def test_in_tetrahedron(self):
        c_data = data.Data(space.srgb, n_data)
        g = gamut.Gamut(space.srgb, c_data)

        self.assertTrue(True, g.in_tetrahedron(tetrahedron, tetra_p_inside))        # Point is on the tetrahedron
        self.assertFalse(False, g.in_tetrahedron(tetrahedron, tetra_p_not_inside))  # Point is NOT on tetrahedron
        self.assertTrue(True, g.in_tetrahedron(tetrahedron, tetra_p_on_surface))

    def test_in_triangle(self):
        c_data = data.Data(space.srgb, n_data)
        g = gamut.Gamut(space.srgb, c_data)

        self.assertFalse(False, g.in_trinagle(triangle, triangle_point_not_coplanar))
        self.assertFalse(False, g.in_trinagle(triangle, triangle_point_coplanar_but_outside))
        self.assertTrue(True, g.in_trinagle(triangle, triangle_point_inside))

    def test_feito_torres(self):
        c_data = data.Data(space.srgb, n_data)
        g = gamut.Gamut(space.srgb, c_data)

        g.feito_torres(P=1)

if __name__ == '__main__':
    unittest.main(exit=False)
