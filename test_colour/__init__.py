#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_colour: Colour spaces, colour metrics and colour data tests

Copyright (C) 2013-2016 Ivar Farup, Lars Niebuhr,
Sahand Lahafdoozian, Nawar Behenam, Jakob Voigt

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

# Main file. Just import the other files.
import os
from test_colour import test_space, test_data, test_tensor, \
    test_metric, test_statistics, test_misc, test_image, test_gamut


if __name__ == "__main__":
    print("Running test_gamut")
    os.system('python test_gamut.py')

    print("Running test_space")
    os.system('python test_space.py')

    print("Running test_data")
    os.system('python test_data.py')

    print("Running test_tensor")
    os.system('python test_tensor.py')

    print("Running test_metric")
    os.system('python test_metric.py')

    print("Running test_statistics")
    os.system('python test_statistics.py')

    print("Running test_misc")
    os.system('python test_misc.py')

    print("Running test_image")
    os.system('python test_image.py')
