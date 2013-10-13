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

dEab = colour.metric.dE_ab(d['data1'], d['data2'])
dEuv = colour.metric.dE_uv(d['data1'], d['data2'])
dE00 = colour.metric.dE_00(d['data1'], d['data2'])
dE99 = colour.metric.dE_DIN99(d['data1'], d['data2'])
dE99b = colour.metric.dE_DIN99b(d['data1'], d['data2'])
dE99c = colour.metric.dE_DIN99c(d['data1'], d['data2'])
dE99d = colour.metric.dE_DIN99d(d['data1'], d['data2'])

s_ab, i = colour.statistics.stress(dEab, d['dV'], d['weights'])
s_uv, i = colour.statistics.stress(dEuv, d['dV'], d['weights'])
s_00, i = colour.statistics.stress(dE00, d['dV'], d['weights'])
s_99, i = colour.statistics.stress(dE99, d['dV'], d['weights'])
s_99b, i = colour.statistics.stress(dE99b, d['dV'], d['weights'])
s_99c, i = colour.statistics.stress(dE99c, d['dV'], d['weights'])
s_99d, i = colour.statistics.stress(dE99d, d['dV'], d['weights'])

print s_00, s_99d, s_00 / s_99d, i
