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

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d


class Gamut:
    """Class for representing colour gamuts computed in various colour spaces.
    """
    def __init__(self, sp, points):
        """Construct new gamut instance and compute the gamut.

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

    def initialize_convex_hull(self, sp, points):
        """Initializes the gamuts convex hull in the desired colour space

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
        """For the given data points checks if points are inn the convex hull
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

        nd_data = c_data.get(sp)    # Convert to ndarray

        # print("The ndarray send to is_inside:")
        # print(nd_data)
        # print("..And it's shape:")
        # print(np.shape(nd_data))

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

                Parameters
                ----------
                nda : ndarray
                      An n-dimensional array containing the remaining dimensions to iterate.
                indices : array
                        The dimensional path to the coordinate.
                        Needs to be as long as the (amount of dimensions)-1 in nda and filled with -1's
                bool_array : ndarray
                        Array containing true/fals in last dimention.
                        Shape is the same as nda(minus the last dim)
                hull: ConvexHull
                    A ConvexHull generated from the gamuts vertices.
        """
        if np.ndim(nda) != 1:  # Not yet reached a leaf node
            curr_dim = 0
            for index in np.nditer(indices):              # calculate the dimension number witch we are currently in
                if index != -1:         # If a dimension is previously iterated the cell will have been changed to a
                                        # non-negative number.
                    curr_dim += 1

            numb_of_iterations = 0
            for nda_minus_one in nda:              # Iterate over the length of the current dimension
                indices[curr_dim] = numb_of_iterations  # Update the path in indices before next recusrive call
                self.traverse_ndarray(nda_minus_one, indices, bool_array)
                numb_of_iterations += 1
            indices[curr_dim] = -1  # should reset the indences array when the call dies

        else:   # We have reached a leaf node
            # self.single_point_inside(nda)  # nda is now reduced to a one dimensional list containing three numbers.
                                            # (a data point to be checked)
            print("Leaf node found:")
            bool_array[(tuple(indices))] = True
            print(bool_array)
            print("----------------")

            # print(indices)
            # print(nda)

    def single_point_inside(hull, point):
        """Checks if a single coordinate in 3d is inside the given hull.

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

    def get_vertices(self, nd_data):
        """Get all convex hull vertices points and save it in a array list.

            Parameter
            ---------

        :return: point_list
        """

        point_list = []  # Array list with vertices points.
        for i in self.hull.vertices:
            point_list.append(nd_data[i])
        point_array = np.array(point_list)
        return point_array

    """def get_surface(self, sp):


            Parameters
            ----------
            :param sp: Space
                The colour space for computing the gamut.
            :return:

        nd_data = self.data.get_linear(sp)

        points = self.get_vertices(nd_data)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, z, cmap=plt.cm.jet)
        plt.show()"""

    def plot_surface(self, ax, sp):
        """Plot all the vertices points on the recived axel

            Parameters
            ----------
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

        ax.set_xlim([0, 20])                            # Set the limits for the plot manually
        ax.set_ylim([-20, 20])
        ax.set_zlim([-20, 20])
        plt.show()
