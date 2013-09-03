#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
misc: Various auxilliary functions, part of the colour package

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

import matplotlib.pyplot as plt

#==============================================================================
# Auxiliary functions
#==============================================================================

def plot_ellipses(ellipses, axis=None, alpha=1,
                  facecolor=[.5, .5, .5], edgecolor=[0, 0, 0], fill=False):
    """
    Plot the list of ellipses on the given axis.
    
    Parameters
    ----------
    ellipses : list
        List of Ellipse objects.
    axis : AxesSubplot
        Axis on which to plot the ellipses.
    alpha : float
        Alpha for colour blending.
    facecolor : ndarray
        RGB array of the ellipse faces.
    edgecolor : ndarray
        RGB array of the ellipse edges.
    fill : bool
        Fill the ellipses or not.
    """
    if axis == None:
        axis = plt.gca()
    for e in ellipses:
        axis.add_artist(e)
        e.set_clip_box(axis.bbox)
        e.set_alpha(alpha)
        e.set_facecolor(facecolor)
        e.set_edgecolor(edgecolor)
        e.set_fill(fill)
