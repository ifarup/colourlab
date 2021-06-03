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
from colourlab import image, image_core, space, data

im = image.Image(space.srgb, np.random.rand(5, 5, 3))
im2 = image.Image(space.srgb, np.random.rand(5, 5, 3))

class TestImage(unittest.TestCase):

    def test_gradient(self):
        imi, imj = im.gradient(space.srgb)
        self.assertTrue(isinstance(imi, data.Vectors))
        self.assertTrue(isinstance(imj, data.Vectors))

    def test_stress(self):
        im_stress = im.stress(space.srgb)
        self.assertTrue(isinstance(im_stress, image.Image))
    
    def test_c2g_diffusion(self):
        g1 = im.c2g_diffusion(space.srgb, 5)
        self.assertTrue(isinstance(g1, np.ndarray))
        g2 = im.c2g_diffusion(space.srgb, 5, l_minus=False)
        self.assertTrue(isinstance(g2, np.ndarray))

    def test_structure_tensor(self):
        s11, s12, s22, lambda1, lambda2, e1i, e1j, e2i, e2j = im.structure_tensor(space.srgb)
        self.assertTrue(isinstance(s11, np.ndarray))
        self.assertTrue(isinstance(s12, np.ndarray))
        self.assertTrue(isinstance(s22, np.ndarray))
        self.assertTrue(isinstance(lambda1, np.ndarray))
        self.assertTrue(isinstance(lambda2, np.ndarray))
        self.assertTrue(isinstance(e1i, np.ndarray))
        self.assertTrue(isinstance(e1j, np.ndarray))
        self.assertTrue(isinstance(e2i, np.ndarray))
        self.assertTrue(isinstance(e2j, np.ndarray))

    def test_diffusion_tensor(self):
        d1 = im.diffusion_tensor(space.srgb, image_core.dpdl_perona_invsq)
        self.assertTrue(isinstance(d1[0], np.ndarray))
        self.assertEqual(len(d1), 3)
        d2 = im.diffusion_tensor(space.srgb, image_core.dpdl_perona_exp)
        self.assertTrue(isinstance(d2[0], np.ndarray))
        self.assertEqual(len(d2), 3)
        d3 = im.diffusion_tensor(space.srgb, image_core.dpdl_tv)
        self.assertTrue(isinstance(d3[0], np.ndarray))
        self.assertEqual(len(d3), 3)

    def test_anisotropic_diffusion(self):
        im_diff1 = im.anisotropic_diffusion(space.srgb, 5)
        self.assertTrue(isinstance(im_diff1, image.Image))
        im_diff2 = im.anisotropic_diffusion(space.srgb, 5, linear=False)
        self.assertTrue(isinstance(im_diff2, image.Image))