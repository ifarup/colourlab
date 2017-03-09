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
cube = np.array([[0., 0., 0.],      # 0  vertices
                [10., 0., 0.],      # 1  vertices
                [10., 10., 0.],     # 2  vertices
                [0., 10., 0.],      # 3  vertices
                [5., 5., 5.],       # 4  non vertices
                [4., 6., 2.],       # 5  non vertices
                [10., 10., 10.],    # 6  vertices
                [1., 2., 3.],       # 7  non vertices
                [10., 0., 10.],     # 8  vertices
                [0., 0., 10.],      # 9  vertices
                [0., 10., 10.]])    # 10 vertices

line = np.array([[0, 0, 0], [3, 3, 3]])             # Line used in testing.
point_on_line = np.array([1, 1, 1])                 # Point inside the line to be tested.
point_not_paralell_to_line = np.array([2, 3, 2])    # Point outside the line to be tested.
point_opposite_direction_than_line = np.array([-1, -1, -1])
point_further_away_than_line = np.array([4, 4, 4])

tetrahedron = np.array([[10., 10., 10.], [0., 10., 0.], [0., 0., 0.], [0., 0., 10.]])  # Tetrahedron used in testing.
tetra_p_inside = np.array([2., 3., 4.])               # Point inside the tetrahedron to be tested.
tetra_p_not_inside = np.array([20., 1., 2.])          # Point outside the tetrahedron to be tested.
tetra_p_on_surface = np.array([0., 5., 0.])

tetrahedron_two = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, 0], [0, 0, 2]])     # Tetrahedron used in testing.

tetrahedron_three = np.array([[10, 10, 10], [10, 10, 0], [10, 0, 10], [0, 10, 10]])     # Tetrahedron used in testing.

# Used in test for is_inside
points_1d = np.array([5., 11., 3.])
bool_1d = np.array([False])
points_2d = np.array([[5., 11., 3.], [3., 2., 1.], [11., 3., 4.], [9., 2., 1.]])
bool_2d = np.array([False, True, False, True])
points_3d = np.array([[[3., 1., 2.], [3., 2., 4.], [10., 3., 11.], [14., 3., 2.]]])
bool_3d = np.array([[True, True, False, False]])

triangle = np.array([[0., 0., 0.], [4., 0., 0.], [0., 0., 4.]])
triangle_point_inside = np.array([2., 0., 2.])
triangle_point_not_coplanar = np.array([2., 2., 2.])
triangle_point_coplanar_but_outside = np.array([5., 0., 3.])

# Same triangle as above, move by vector (2,2,2)
triangle2 = np.array([[2., 2., 2.], [6., 2., 2.], [2., 2., 6.]])
triangle2_point_inside = np.array([4., 2., 4.])
triangle2_point_not_coplanar = np.array([4., 4., 4.])
triangle2_point_coplanar_but_outside = np.array([7., 2., 5.])

polyhedron = np.array([[38., 28., 30.], [31., 3., 43.],  [50., 12., 38.], [34., 45., 18.],
                       [22., 13., 29.], [22., 2., 31.],  [26., 44., 35.], [31., 43., 22.],
                       [22., 43., 13.], [13., 43., 11.], [50., 32., 29.], [26., 35., 18.],
                       [43., 3., 11.],  [26., 3., 44.],  [11., 3., 18.],  [18., 3., 26.],
                       [11., 45, 13.],  [13., 45., 29.], [18., 45., 11.], [2., 32., 31.],
                       [29., 2., 22.],  [35., 12., 18.], [18., 12., 34.], [34., 12., 50.],
                       [34., 50., 45.], [45., 50., 29.], [3., 30., 44.],  [29., 32., 2.],
                       [30., 28., 44.], [50., 30., 32.], [37., 12., 35.], [44., 28., 35.],
                       [35., 28., 37.], [32., 30., 31.], [31., 30., 3.],  [38., 30., 50.],
                       [37., 28., 38.], [38., 12., 37.]])


class TestGamut(unittest.TestCase):

    def test_gamut_initialize(self):
        # Test for convex hull
        c_data = data.Data(space.srgb, cube)          # Generating the colour Data object
        g = gamut.Gamut(space.srgb, c_data)
        vertices = np.array([0, 1, 2, 3, 6, 8, 9, 10])  # Known indices of vertices for the test case

        self.assertEqual(vertices.tolist(), g.vertices.tolist())    # Checking that the vertices match

    def test_is_inside(self):                               # Test for gamut.Gamut.is_inside
        c_data = data.Data(space.srgb, cube)
        g = gamut.Gamut(space.srgb, c_data)

        c_data = data.Data(space.srgb, points_3d)
        a = g.is_inside(space.srgb, c_data)
        self.assertEqual(a.shape, points_3d.shape[:-1])     # Asserts if shape is reduced by 1dim
        self.assertEqual(a.dtype, bool)                     # Asserts is data type in the array is boolean
        self.assertTrue(np.allclose(a, bool_3d))            # Asserts that the returned values are correct

        c_data = data.Data(space.srgb, points_2d)
        a = g.is_inside(space.srgb, c_data)
        self.assertEqual(a.shape, points_2d.shape[:-1])     # Asserts if shape is reduced by 1dim
        self.assertEqual(a.dtype, bool)                     # Asserts is data type in the array is boolean
        self.assertTrue(np.allclose(a, bool_2d))            # Asserts that the returned values are correct

        c_data = data.Data(space.srgb, points_1d)
        a = g.is_inside(space.srgb, c_data)
        self.assertEqual(1, a.size)                         # When only one point is sent, still returned a array
        self.assertEqual(a.dtype, bool)                     # Asserts is data type in the array is boolean
        self.assertTrue(np.allclose(a, bool_1d))            # Asserts that the returned values are correct

        c_data = data.Data(space.srgb, self.generate_sphere(15, 100))
        g = gamut.Gamut(space.srgb, c_data)

        c_data = data.Data(space.srgb, self.generate_sphere(10, 15))   # Points lie within the sphere(inclusion = true)
        a = g.is_inside(space.srgb, c_data)
        self.assertTrue(np.allclose(a, np.ones(a.shape)))              # Assert that all points lie within the gamut

        c_data = data.Data(space.srgb, self.generate_sphere(20, 15))   # Points lie outside the sphere(inclusion = true)
        a = g.is_inside(space.srgb, c_data)
        self.assertTrue(np.allclose(a, np.zeros(a.shape)))             # Assert that all points lie without the gamut

    def test_get_vertices(self):
        # Test for gamut.Gamut.get_vertices
        c_data = data.Data(space.srgb, cube)  # Generating the colour Data object
        g = gamut.Gamut(space.srgb, c_data)
        n1_data = np.array([[0, 0, 0],      # 0  vertices    # Array with just the vertices used for comparison.
                           [10, 0, 0],      # 1  vertices
                           [10, 10, 0],     # 2  vertices
                           [0, 10, 0],      # 3  vertices
                           [10, 10, 10],    # 6  vertices
                           [10, 0, 10],     # 8  vertices
                           [0, 0, 10],      # 9  vertices
                           [0, 10, 10]])    # 10 vertices

        vertices = g.get_vertices(cube)
        self.assertTrue(np.array_equiv(n1_data, vertices))    # Compares return array with the known vertices array.

        vertices = g.get_vertices(cube)                     # Calls the function and add the vertices to the array.
        self.assertTrue(np.array_equiv(n1_data, vertices))    # Compares returned array with the known vertices array.

    def test_plot_surface(self):         # Test for gamut.Gamut.plot_surface
        fig = plt.figure()                          # Creates a figure
        ax = fig.add_subplot(111, projection='3d')  # Creates a 3D plot ax

        # c = self.generate_circle(1000, 10)
        # c_data = data.Data(space.srgb, c)
        # g = gamut.Gamut(space.srgb, c_data)

        c_data = data.Data(space.srgb, polyhedron)  # Generating the colour Data object
        g = gamut.Gamut(space.srgb, c_data)         # Creates a new gamut

        sp = g.space                                 # specifies the color space
        g.plot_surface(ax, sp)                       # Calls the plot function

    def test_in_line(self):
        c_data = data.Data(space.srgb, cube)
        g = gamut.Gamut(space.srgb, c_data)

        self.assertTrue(g.in_line(np.array([[2, 2, 2],[2, 2, 2]]), np.array([2, 2, 2])))  # All points equal.
        self.assertFalse(g.in_line(line, point_not_paralell_to_line))            # Point in NOT parallel to line
        self.assertFalse(g.in_line(line, point_opposite_direction_than_line))    # Point opposite dir then line
        self.assertFalse(g.in_line(line, point_further_away_than_line))          # Point is is further then line
        self.assertTrue(g.in_line(line, point_on_line))                          # Point is on line
        self.assertFalse(g.in_line(np.array([[3, 3, 3], [4, 4, 4]]), np.array([5, 5, 5])))  # Point is on line

        self.assertFalse(g.interior(line, point_not_paralell_to_line))            # Point in NOT parallel to line
        self.assertFalse(g.interior(line, point_opposite_direction_than_line))    # Point opposite dir then line
        self.assertFalse(g.interior(line, point_further_away_than_line))          # Point is is further then line
        self.assertTrue(g.interior(line, point_on_line))                           # Point is on line
        self.assertFalse(g.interior(np.array([[3, 3, 3], [4, 4, 4]]), np.array([5, 5, 5])))  # Point is on line

    def test_in_tetrahedron(self):
        c_data = data.Data(space.srgb, tetrahedron)
        g = gamut.Gamut(space.srgb, c_data)

        self.assertTrue(g.in_tetrahedron(tetrahedron, tetra_p_inside))        # Point is on the tetrahedron
        self.assertFalse(g.in_tetrahedron(tetrahedron, tetra_p_not_inside))   # Point is NOT on tetrahedron
        self.assertTrue(g.in_tetrahedron(tetrahedron, tetra_p_on_surface))    # Point is on a simplex(counts as inside)

        self.assertTrue(g.interior(tetrahedron, tetra_p_inside))        # Point is on the tetrahedron
        self.assertFalse(g.interior(tetrahedron, tetra_p_not_inside))  # Point is NOT on tetrahedron
        self.assertTrue(g.interior(tetrahedron, tetra_p_on_surface))

    def test_in_triangle(self):
        c_data = data.Data(space.srgb, cube)
        g = gamut.Gamut(space.srgb, c_data)

        self.assertFalse(g.in_triangle(triangle, triangle_point_not_coplanar))
        self.assertFalse(g.in_triangle(triangle, triangle_point_coplanar_but_outside))
        self.assertTrue(g.in_triangle(triangle, triangle_point_inside))

        self.assertFalse(g.in_triangle(triangle2, triangle2_point_not_coplanar))
        self.assertFalse(g.in_triangle(triangle2, triangle2_point_coplanar_but_outside))
        self.assertTrue(g.in_triangle(triangle2, triangle2_point_inside))

        self.assertFalse(g.interior(triangle, triangle_point_not_coplanar))
        self.assertFalse(g.interior(triangle, triangle_point_coplanar_but_outside))
        self.assertTrue(g.interior(triangle, triangle_point_inside))

        self.assertFalse(g.interior(triangle2, triangle2_point_not_coplanar))
        self.assertFalse(g.interior(triangle2, triangle2_point_coplanar_but_outside))
        self.assertTrue(g.interior(triangle2, triangle2_point_inside))

    def test_sign(self):
        c_data = data.Data(space.srgb, cube)
        g = gamut.Gamut(space.srgb, c_data)

        print(g.sign(tetrahedron_two))

    def test_is_coplanar(self):
        c_data = data.Data(space.srgb, cube)
        g = gamut.Gamut(space.srgb, c_data)

        points = np.array([[0, 0, 0], [2, 2, 0], [3, 3, 0], [1, 1, 0]])  # coplanar points
        self.assertTrue(True, g.is_coplanar(points))

        points = np.array([[0, 0, 1], [2, 2, 0], [3, 3, 0], [1, 1, 0]])  # non-coplanar points
        self.assertFalse(False, g.is_coplanar(points))

    # def test_generate_sphere_points(self):
    #     r = 1
    #     phi = np.linspace(0, np.pi, 20)
    #     theta = np.linspace(0, 2 * np.pi, 40)
    #     x = r * np.cos(theta) * np.sin(phi)
    #     y = r * np.sin(theta) * np.sin(phi)
    #     z = r * np.cos(phi)
    #
    #     print(x)
    #
    #     np.reshape(a, (3,3), order='F')
    #
    #     print(y)
    #     print(z)
    #
    #
    #     coordinates = np.ndarray(shape=np.shape(num_of_points))
    #     for i in range(num_of_points):

    def test_center_of_mass(self):
        c_data = data.Data(space.srgb, cube)
        g = gamut.Gamut(space.srgb, c_data)
        cm = g.center_of_mass(g.get_vertices(g.hull.points))   # Get coordinate for center of the cube
        cp = np.array([5., 5., 5.])                            # Point in center of cube.
        self.assertEqual(cp.all(), cm.all())                   # Assert true that the points are the same.

    def test_fix_orientation(self):
        c_data = data.Data(space.srgb, cube)
        g = gamut.Gamut(space.srgb, c_data)
        g.fix_orientation()

    def test_feito_torres_with_sphere(self):
        gamut_sphere = self.generate_sphere(10, 100)
        outside = self.generate_sphere(12, 5)
        innside = self.generate_sphere(8, 5)
        c_idk = self.generate_sphere(9.9, 12)
        c_data = data.Data(space.srgb, gamut_sphere)
        g = gamut.Gamut(space.srgb, c_data)

        print("----------------")
        print("Should be inside")
        print("----------------")
        for i in range(0, innside.shape[0]):
            print(g.feito_torres(innside[i]))

        print("----------------")
        print("Should be outside")
        print("----------------")
        for i in range(0, innside.shape[0]):
            print(g.feito_torres(outside[i]))
        print("----------------")
        print("Uncertain")
        print("----------------")
        for i in range(0, innside.shape[0]):
            print(g.feito_torres(c_idk[i]))

    def test_generate_sphere(self):
        sphere = self.generate_sphere(5, 10)

        # plt.plot(sphere[:,0],sphere[:,1])
        # plt.show()
        #
        # plt.plot(sphere[:,0],sphere[:,2])
        # plt.show()

        fig = plt.figure()  # Creates a figure
        ax = fig.add_subplot(111, projection='3d')  # Creates a 3D plot ax

        c_data = data.Data(space.srgb, sphere)
        g = gamut.Gamut(space.srgb, c_data)

        sp = g.space  # specifies the color space
        g.plot_surface(ax, sp)  # Calls the plot function

    def test_true_shape(self):

        c_data = data.Data(space.srgb, cube)
        g = gamut.Gamut(space.srgb, c_data)

        a = np.array([[0, 0, 0], [2, 2, 2], [2, 2, 2]])
        a = g.true_shape(a)
        # Test remove duplicates
        a = np.array([[0, 0, 0], [2, 2, 2], [0, 0, 0], [2, 2, 2]])
        self.assertEqual(2, g.true_shape(a).shape[0])

        # Test 3 points on the same line should return outer points
        a = np.array([[0, 0, 0], [2, 2, 2], [3, 3, 3]])
        self.assertTrue(np.allclose(g.true_shape(a), np.array([[0, 0, 0], [3, 3, 3]])))

        # Test 4 points that are actually a triagle
        a = np.array([[0, 0, 0], [0, 3, 0], [3, 0, 0], [1, 1, 0]])
        self.assertTrue(np.allclose(g.true_shape(a), np.array([[0, 0, 0], [0, 3, 0], [3, 0, 0]])))

        # Test 4 points that are all outher vetecis in a convex polygon
        a = np.array([[0, 0, 0], [0, 3, 0], [3, 0, 0], [5, 5, 0]])
        self.assertTrue(np.allclose(g.true_shape(a), np.array([[0, 0, 0], [0, 3, 0], [3, 0, 0], [5, 5, 0]])))

    def test_linalg_det(self):
        matrix = np.array([[1, 1, 1], [3, 3, 3], [4, 4, 4]])
        a = np.linalg.det(matrix)
        print(a)

    @staticmethod
    def generate_sphere(r, n):
        """Generates a sphere or points. Used in tests to generate gamut, and inclusion points.

        :param r: int
            The radius to the points.
        :param n: int
            Number of points to be generated.
        :return: ndarray
            Numpy array dim(n,3) with the points of the sphere.
        """
        theta = np.random.uniform(0, 2*np.pi, n)
        phi = np.random.uniform(0, np.pi, n)

        x = r * (np.sin(phi) * np.cos(theta))
        y = r * (np.sin(phi) * np.sin(theta))
        z = r * (np.cos(phi))

        sphere = np.vstack((x, y, z)).T

        return sphere


if __name__ == '__main__':
    unittest.main(exit=False)
