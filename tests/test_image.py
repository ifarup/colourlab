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
from colourlab import image, space, data

im = image.Image(space.srgb, np.random.rand(5, 5, 3))
im2 = image.Image(space.srgb, np.random.rand(5, 5, 3))

class TestImage(unittest.TestCase):

    def test_stress(self):
        im_stress = im.stress(space.srgb)
        self.assertTrue(isinstance(im_stress, image.Image))

    def test_c2g(self):
        g1 = im.c2g_diffusion(space.srgb, 5)
        self.assertTrue(isinstance(g1, np.ndarray))
        g2 = im.c2g_diffusion(space.srgb, 5, l_minus=False)
        self.assertTrue(isinstance(g2, np.ndarray))
        g3 = im.c2g_diffusion(space.srgb, 5, aniso=False)
        self.assertTrue(isinstance(g3, np.ndarray))

    def test_diffusion_tensor(self):
        d1 = im.diffusion_tensor(space.srgb, type='exp', dir='m')
        self.assertTrue(isinstance(d1[0], np.ndarray))
        self.assertEqual(len(d1), 3)
        d2 = im.diffusion_tensor(space.srgb, type='exp', dir='c')
        self.assertTrue(isinstance(d1[0], np.ndarray))
        self.assertEqual(len(d1), 3)

    def test_diff(self):
        diff = im.diff(space.srgb, im2)
        self.assertTrue(isinstance(diff, data.Vectors))
