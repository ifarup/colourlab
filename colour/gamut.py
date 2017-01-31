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
        self.space = sp
        self.data = points
        hull = spatial.ConvexHull(points.get_linear(sp))
        self.vertices = hull.vertices
        self.simplices = hull.simplices
        self.neighbors = hull.neighbors


# Add test function, see one of the other modules.
def test():
    print("ok")

