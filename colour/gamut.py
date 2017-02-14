#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gamut: Colour metric functions. Part of the colour package.

Copyright (C) 2013-2016 Ivar Farup, Lars Niebuhr,
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

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d


class Gamut:
    """Class for representing colour gamuts computed in various colour spaces.
    """
    def __init__(self, sp, points):
        """Construct new gamut instance and compute the gamut.

            :param sp : Space
                The colour space for computing the gamut.
            :param points : Data
                The colour points for the gamut.
        """
        self.data = points      # The data points are stored in the original format.
        self.space = None
        self.hull = None
        self.vertices = None
        self.simplices = None
        self.neighbors = None
        self.initialize_convex_hull(sp, points)     # Initializes all of the above, using a sub-initialization method

    def initialize_convex_hull(self, sp, points):
        """Initializes the gamuts convex hull in the desired colour space

        :param sp : Space
            The colour space for computing the gamut.
        :param points : Data
            The colour points for the gamut.
        """
        self.space = sp
        self.hull = spatial.ConvexHull(points.get_linear(sp))
        self.vertices = self.hull.vertices
        self.simplices = self.hull.simplices
        self.neighbors = self.hull.neighbors

    def is_inside(self, sp, c_data):
        """For the given data points checks if points are inn the convex hull
            NB: this method cannot be used for modified convex hull.

        :param sp : Space
            The colour space for computing the gamut.
        :param c_data : Data
            Data object with the colour points for the gamut.
        """
        # Calculate a new convexhull given only the vertecis for further use to increase efficiency
        # hull = spatial.ConvexHull(g.vertices()).

        nd_data = c_data.get(sp)    # Convert to ndarray

        if nd_data.ndim == 1:   # Handle the special case of a only one point beeing evaluated.
            self.single_point_inside(self, c_data)
        else:
            indices = np.ones(nd_data.ndim - 1, int) * -1  # Important that indencis is initialized with negative numb.
            c_data = c_data.get(sp)
            bool_array = np.zeros(np.shape(nd_data)[:-1], bool)     # Create a bool array with the same shape as the
                                                                    # nd_data(minus the last dimension)
            self.traverse_ndarray(c_data, indices, bool_array)

    def traverse_ndarray(self, nda, indices, bool_array):
        """For the given data points checks if points are inn the convex hull
            NB: this method cannot be used for modified convex hull.

        :param nda : ndarray
            An n-dimensional array containing the remaining dimensions to iterate.
        :param indices : array
            The dimensional path to the coordinate.
            Needs to be as long as the (amount of dimensions)-1 in nda and filled with -1's
        :param bool_array : ndarray
            Array containing true/fals in last dimention.
            Shape is the same as nda(minus the last dim)
        :param hull: ConvexHull
            A ConvexHull generated from the gamuts vertices.
        """
        if np.ndim(nda) != 1:  # Not yet reached a leaf node
            curr_dim = 0
            for index in np.nditer(indices):              # calculate the dimension number witch we are currently in
                if index != -1:         # If a dimension is previously iterated the cell will have been changed to a
                                        # non-negative number.
                    curr_dim += 1

            numb_of_iterations = 0
            for nda_minus_one in nda:                   # Iterate over the length of the current dimension
                indices[curr_dim] = numb_of_iterations  # Update the path in indices before next recusrive call
                self.traverse_ndarray(nda_minus_one, indices, bool_array)
                numb_of_iterations += 1
            indices[curr_dim] = -1      # should reset the indences array when the call dies

        else:   # We have reached a leaf node
            # self.single_point_inside(nda) # nda is now reduced to a one dimensional list containing three numbers.
                                            # (a data point to be checked)
            print("Leaf node found:")
            bool_array[(tuple(indices))] = True
            print(bool_array)
            print("----------------")

    def single_point_inside(self, hull, point):
        """Checks if a single coordinate in 3d is inside the given hull.

        :param hull : ndarray
            Convex hull
        :param point: coordinate
            A single coordinate to be tested if it is inside the hull.
        :return True or False
        """
        new_hull = spatial.ConvexHull(np.concatenate((hull.points, [point])))
        if np.array_equal(new_hull.vertices, hull.vertices):
            return True
        return False

    def get_vertices(self, nd_data):
        """Get all hull vertices and save them in a array list.

        :param nd_data : ndarray
            The a copy of gamut.points pre-converted to a desierd colour space.
        :return: point_list
        """
        point_list = []                     # Array list with vertices points.

        for i in self.hull.vertices:        # For loop that goes through all the vertices
                                            # and for each goes to the points and adds the coordinents to the list.
            point_list.append(nd_data[i])
        point_array = np.array(point_list)

        return point_array                  # Returns ndarray.

    def plot_surface(self, ax, sp):
        """Plot all the vertices points on the recived axel

        :param ax: Axel
            The axel to draw the points on
        :param sp: Space
            The colour space for computing the gamut.
        """
        nd_data = self.data.get_linear(sp)              # Creates a new ndarray with points
        points = self.get_vertices(nd_data)             # ndarray with all the vertices
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        for i in range(self.hull.simplices.shape[0]):   # Itirates and draws all the vertices points
            tri = art3d.Poly3DCollection([self.hull.points[self.hull.simplices[i]]])
            ax.add_collection(tri)                      # Adds created points to the ax

        ax.set_xlim([0, 10])                            # Set the limits for the plot manually
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        plt.show()

    def feito_torres(self, P):
        """ Tests if a point P is inside a polyhedron.

        :param P: ndarray
            Point to be tested for inclusion.
        :return: bool
            True if P is included in the polyhedron.
        """
        inclusion = []
        v_pluss = []
        v_minus = []

        for el in self.simplices:
            facet = self.get_coordinates(el)    # Get the coordinates for the current facet

            if self.in_trinagle(facet, P):      # Check if P is on the current facet.
                return True

            o_v1 = np.array([0, 0, 0], facet[0])    # vector from origo to the first vertex in the facet


            # if( self.in_line(o_v1, P) ) & ((  & ... )  | ( ... & ... ))

    def sign(self):
        return True

    def get_coordinates(self, indices):
        """Return the coordinates of points correlating to the  the indices provided.

        :param indices: ndarray
            shape(N,), list of indices
        :return: ndarray
            shape(N, 3)
        """
        coordinates = np.array(indices.shape[0], 3)

        counter = 0
        for index in indices:
            coordinates[counter] = self.points(index)
            counter += 1

        return coordinates

    def in_tetrahedron(self, tetrahedron, p):
        """Checks if the point P, pointed to by vector p, is inside(incuding the surface) the tetrahedron

        :param tetrahedron: ndarray
            The four points of a tetrahedron
        :param p: ndarray
            The point to be tested for inclusion in the tetrahedron.

        :return: Bool
            True if q is inside or on the surface of the tetrahedron.
        """
        hull = spatial.Delaunay(tetrahedron)    # Generate a convex hull repesentaion of points
        return hull.find_simplex(p) >= 0        # and check if 'q' is inside.

    '''
            # If neccesary move the line so that a is the origin.
            if line[0] != np.array([0,0,0]):
                a = np.array([0,0,0])
                b = line[1] - line[0]
            else:
                a = line[0]
                b = line[1]
    '''

    def in_line(self, line, p):
        """Checks if a point P is on the line from  A to B

        :param line: ndarray
            line segment from point A to point B
        :param p: ndarray
            Vector from A to P
        :return: Bool
            True is P in in the line segment from A to P.
        """
        b = line[1] - line[0]   # Move the line so that A is (0,0,0). 'b' is the vector from A to B.

        # Check if the cross b x p is 0, if not the vectors are not collinear.
        matrix = np.array([[1, 1, 1], b, p, ])
        if np.linalg.det(matrix) != 0:
            return False

        # Check if b and p have opposite directions
        dot_b_p = np.dot(p, b)
        if dot_b_p < 0:
            return False

        # Finally check that q-vector is shorter b-vector
        dot_qq = np.dot(p, p)
        if dot_qq > dot_b_p:
            print("q-vector longer than b-vector")
            return False

        return True

    def in_trinagle(self, triangle, p):
        """Takes three points of a triangle in 3d, and determines if the point w is within that triangle.
            This function utilizes the baycentric technique explained here
            https://blogs.msdn.microsoft.com/rezanour/2011/08/07/barycentric-coordinates-and-point-in-triangle-tests/

        :param triangle: ndarray
            An ndarray with shape: (3,3), with points A, C and C beeing triangle[0]..[2]
        :param p: ndarray
            An ndarray with shape: (3,), the point to be tested for inclusion in the triangle.
        :return: Bool
            True if 'w' is within the triangle ABC.
        """
        # Make 'A' the local origo for the points.
        b = triangle[1] - triangle[0]  # 'b' is the vector from A to B
        c = triangle[2] - triangle[0]  # 'c' is the vector from A to C
        p -= triangle[0]               # 'p' is now the vector from A to the point being tested for inclusion

        b_X_c = np.cross(b, c)         # Calculating the vector of the cross product b x c
        if np.dot(b_X_c, p) != 0:      # If p-vector is not coplanar to b and c-vector, it is not in the triangle.
            print("not coplanar")
            return False

        c_X_p = np.cross(c, p)          # Calculating the vector of the cross product c x p
        c_X_b = np.cross(c, b)          # Calculating the vector of the cross product c x b

        if np.dot(c_X_p, c_X_b) < 0:    # If the two cross product vectors are not pointing in the same direction. exit
            print("c_X_p anc c_X_b has opposite directions")
            return False

        c_X_p = np.cross(b, p)          # Calculating the vector of the cross product b x p

        if np.dot(c_X_p, b_X_c) < 0:  # If the two cross product vectors are not pointing in the same direction. exit
            print("c_X_p and  ")
            return False
        denom = np.linalg.norm(b_X_c)
        r = c_X_p/denom
        t = np.linalg.norm(c_X_p)

        print("got to the end")
        return r + t <= 1
