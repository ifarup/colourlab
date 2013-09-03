#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
<name>: <description>

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

from colour import *

ones = 1 * np.ones(30)
diff1 = ones + np.random.random_sample(np.shape(ones)[0])
diff2 = ones + np.random.random_sample(np.shape(ones)[0])
print colour.stress(diff1, diff2)
