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

d = colour.data.build_m_rit_dupont()

dE = colour.metric.euclidean(colour.space.ipt, d['data1'], d['data2'])
s, i = colour.statistics.stress(dE, d['dV'], d['weights'])

sp = []
r = np.linspace(1000, 10000)
for i in r:
    space_p = colour.space.TransformPoincareDisk(colour.space.ipt, i)
    dEp = colour.metric.poincare_disk(space_p, d['data1'], d['data2'])
    s_p, interval = colour.statistics.stress(dEp, d['dV'], d['weights'])
    sp.append(s_p)

print s, interval[0] * s, interval[1] * s

plt.plot(r, sp)
plt.grid()
plt.show()
