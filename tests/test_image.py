#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_image: Unittests for all functions in the image module.

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
from colourlab import image, space

im = image.Image(space.srgb, np.random.rand(5, 5, 3))

class TestImage(unittest.TestCase):

    def test_stress(self):
        im_stress = im.stress(space.srgb)
        self.assertTrue(isinstance(im_stress, image.Image))

    def test_c2g(self):
        g = im.c2g_diffusion(space.srgb, 5)
        self.assertTrue(isinstance(g, np.ndarray))
