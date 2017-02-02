#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colour: Colour spaces, colour metrics and colour data

Copyright (C) 2011-2016 Ivar Farup

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

from colour import space, data, tensor, metric, statistics, misc, image, gamut


# Test package

def test():
    """
    Tests only gamut module, because other module test are out of date.
    """
    print('\nTesting module gamut:\n')
    gamut.test()

test()
