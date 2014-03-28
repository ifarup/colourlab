#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test: Test various features of the colour package (continuously updated)

Copyright (C) 2013 Ivar Farup

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
import colour
import matplotlib.pyplot as plt
import numpy as np

m_data, m_names, m_lab = colour.data.d_Munsell('real')
lab = m_data.get(colour.space.cielab)
diffs, opt_lab, scale, angle = colour.statistics.minimal_dataset_distance(lab, m_lab)
print scale, angle, angle * 180 / np.pi
plt.plot(m_lab[:,1], m_lab[:,0], '.')
plt.plot(opt_lab[:,1], opt_lab[:,0], 'r.')
plt.axis('equal')
plt.grid()
plt.show()