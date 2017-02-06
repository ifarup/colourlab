#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gamut: Colour metric functions. Part of the colour package.

Copyright (C) 2013-2016 Ivar Farup, Lars Niebuhr,
Sahand Lahefdoozian, Nawar Behenam, Jakob Voigt

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

import os
import re
import numpy as np
import inspect
from colour import space, data
from scipy import spatial
import unittest.test
from ExFunction import retur, my_contains, my_first

class Gamut:
    """
    Class for representing colour gamuts computed in various colour spaces.
    """
    def __init__(self, sp, points):
        """
        Construct new gamut instance and compute the gamut.

        Parameters
        ----------
        sp : Space
            The colour space for computing the gamut.
        points : Data
            The colour points for the gamut.
        """
        self.data = points      # The data points are stored in the original format.
        self.space = None
        self.hull = None
        self.vertices = None
        self.simplices = None
        self.neighbors = None
        self.initialize_convex_hull(sp, points)   # Initializes all of the above, using a sub-initialization method


    def get_vertices(self):
        """ Get all convex hull vertices points and save it in a array list.

            Parameter
            ---------
            point_list : vertices points
                The colour vertices points.

        :return: point_list
        """

        point_list = []
        for i in range(self.hull.vertices):
            point_list.append(self.hull.points[i])
        return point_list


    def initialize_convex_hull(self, sp, points):
        """ Initializes the gamuts convex hull in the desired colour space

                Parameters
                ----------
                sp : Space
                    The colour space for computing the gamut.
                points : Data
                    The colour points for the gamut.
                """

       # print(points.get_linear(sp))
        self.space = sp

                # TODO: Change back to point.get_linear(sp)
       # self.hull = spatial.ConvexHull(points.get(colour.space.xyz))   # Creating the convex hull in the desired colour space

        self.hull = spatial.ConvexHull(points.get_linear(sp))
        self.vertices = self.hull.vertices
        self.simplices = self.hull.simplices
        self.neighbors = self.hull.neighbors

    def is_inside(self, sp, c_data):
        """ For the given data points checks if points are inn the convex hull
            NB: this method cannot be used for modified convex hull.

            Parameters
            ----------
            sp : Space
                The colour space for computing the gamut.
            c_data : Data
                Data object with the colour points for the gamut.
        """

        # Calculate a new convexhull given only the vertecis for further use to increase efficiency
        # hull = spatial.ConvexHull(g.vertices()).

        # Convert to nd_data

        if c_data.ndim() == 1:     # problem
            self.single_point_inside(self, c_data)
        else:
            indices = np.ones(c_data.ndim() - 1) * -1
            c_data = c_data.get(sp)
            bool_array = np.array(c_data.shape())  # ??
            np.squeeze(bool_array)
            self.find_coordinate(c_data, indices)

    def find_coordinate(self, nda, indices):
        """ For the given data points checks if points are inn the convex hull
            NB: this method cannot be used for modified convex hull.

                Parameters
                ----------
                nda : ndarray
                      An n-dimensional array containing the remaining dimensions to iterate.
                indices : array
                        The dimensional path to the coordinate.
                        Needs to be as long as the (amount of dimensions)-1 in nda and filled with -1's
                hull: ConvexHull
                    A ConvexHull generated from the gamuts vertices.
        """

        if np.ndim(nda) != 1:

            # calculate the dimension number witch we are currently in
            curr_dim = 0
            for index in np.nditer(indices):
                if index != -1:  # If a dimmension is previosly iterated the cell will have been changed to a
                    curr_dim += 1       # non-negative number.

            numb_of_iterations = 0
            for nda_minus_one in nda:              # Iterate over the length of the current dimension
                indices[curr_dim] = numb_of_iterations
                self.find_coordinate(nda_minus_one, indices)
                numb_of_iterations += 1
        else:
           # self.single_point_inside(nda)  # nda is now reduced to a one dimensional list containing three numbers.
                                            # (a data point to be checked)
            print(indices)
            print(nda)


    def single_point_inside(hull, point):
        """ Checks if a single coordinate in 3d is inside the given hull.

                Parameters
                ----------
                hull : array
                    Convex hull
                point: coordinate
                    A single coordinate to be tested if it is inside the hull.
        """

        new_hull = spatial.ConvexHull(np.concatenate((hull.points, [point])))
        if np.array_equal(new_hull.vertices, hull.vertices):
            return True
        return False

def gamut_test():
        n_data = np.array([[0, 0, 0],  # 0 vertecis
                           [10, 0, 0],  # 1 vertecis
                           [10, 10, 0],  # 2 vertecis
                           [0, 10, 0],  # 3 vertecis
                           [5, 5, 5],  # 4 non vertecis
                           [4, 6, 2],  # 5 non vertecis
                           [10, 10, 10],  # 6 vertecis
                           [1, 2, 3],  # 7 non vertecis
                           [10, 0, 10],  # 8 vertecis
                           [0, 0, 10],  # 9 vertecis
                           [0, 10, 10]])  # 10 vertecis
        c_data = data.Data(space.srgb, n_data)
        g = Gamut(space.srgb, c_data)

        points = np.array([[1, 1, 1],  # inside
                           [2, 2, 3],  # inside
                           [20, 2, 3],  # outside
                           [1, 2, 30]])  # outside
        c_points = data.Data(space.srgb, points)
        g.is_inside(space.srgb, c_points)