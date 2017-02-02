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
import colour

from scipy import spatial


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
        self.data = points # orginlt format eller endre til sp fargerom?
        self.space = None
        self.hull = None
        self.vertices = None
        self.simplices = None
        self.neighbors = None
        self.initialize_convex_hull(sp, points)   # Initializes all of the above, using a sub-initialization method

    def initialize_convex_hull(self, sp, points):
        """ Initializes the gamuts convex hull in the desired colour space

                Parameters
                ----------
                sp : Space
                    The colour space for computing the gamut.
                points : Data
                    The colour points for the gamut.
                """

        self.space = sp
        self.hull = spatial.ConvexHull(points.get_linear(sp))   # Creating the convex hull in
                                                                # the desired colour space
        self.vertices = self.hull.vertices
        self.simplices = self.hull.simplices
        self.neighbors = self.hull.neighbors


# Add test function, see one of the other modules.
def test():
    # Test for convex hull
    data = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 5, 5], [4, 6, 2],  # Test data
                     [10, 10, 10], [1, 2, 3], [10, 0, 10], [0, 0, 10], [0, 10, 10]])

    points = colour.data.Data(colour.space.srgb, data)   # Generating the Data object
    vertices = np.array([0, 1, 2, 3, 6, 8, 9, 10])
    g = Gamut(colour.space.srgb, points)
    print("\tTesting vertices should be true: ")
    if g.vertices.all() == vertices.all():
        print("\tTrue")
    else:
        print("\tFalse")
