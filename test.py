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

m_data, m_names, m_lab = colour.data.d_Munsell()
mn = m_data.new_white_point(colour.space.ciecat02, colour.space.white_D65, colour.space.white_E)
lab = m_data.get(colour.space.cielab)
labn = mn.get(colour.space.cielab)
plt.figure(1)
plt.plot(lab[:,1], lab[:,0], '.')
plt.figure(2)
plt.plot(labn[:,1], labn[:,0], '.')
plt.show()
