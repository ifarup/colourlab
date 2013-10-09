#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test: Test various features of the colour package (continuosly updated)

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

m = colour.data.build_m_rit_dupont()
dE_00 = colour.metric.dE_00(m['data1'], m['data2'])
dE_00_alt = colour.metric.linear(colour.space.cielab, m['data1'], m['data2'], colour.tensor.dE_00)
print np.max(np.abs(dE_00_alt - dE_00)) 
plt.plot(dE_00_alt, dE_00, '.')
plt.show()
