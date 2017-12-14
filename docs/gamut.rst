colourlab.gamut
===============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Constructing Gamut
~~~~~~~~~~~~~~~~~~

To construct a new Gamut we need to provide a colour space in the format
provided by colourlab.space, and data/colour points in the format given
the colourlab.data.Data class. If we want to construct the new Gamut in
the colourlab RGB and the fictive points my\_points, we would do the
following:

For convex hull
^^^^^^^^^^^^^^^

.. code:: python

    c_data = data.Data(space.srgb, my_points)    # First generate the Data object to use
    g = gamut.Gamut(space.srgb, c_data)          # Pass along the colourlab and c_data

For modified-convex hull
^^^^^^^^^^^^^^^^^^^^^^^^

When using the modified constructor, we have to choose an exponent for
modifying the gamut radius(gamma), and define the center for expansion.

.. code:: python

    c_data = data.Data(space.srgb, my_points)                        # First generate the Data object to use
    g = gamut.Gamut(space.srgb, c_data, gamma=0.2, center=my_center) # Pass along the colourlab, c_data, gamma and center 

Examples
--------

For all examples: \* **space:** a colourlab.space.Space object \*
**c\_data:** a colourlab.data.Data object \* **p\_in/p\_out:** a point
inside/outside the gamut

All examples presupposes that you have created a colour Data
object(c\_data) and a gamut(g) object.

::

    c_data = data.Data(space, gamut_points)  # Generating the colour Data object
    g = gamut.Gamut(space, c_data)           # Creates a new gamut

is\_inside()
^^^^^^^^^^^^

The function receives two parameters, colourspace and a colour data
object(c\_data). The function checks if points are inn the gamout
boundry and returns an boolean-array containing true/false in the last
dimension.

::

    a = g.is_inside(space, c_data)                # Call the method

plot\_surface()
^^^^^^^^^^^^^^^

The function receives two parameters axis and space. The function
visualizes a gamut figure in 3D.

::

    fig = plt.figure()                            # Creates a figure
    axis = fig.add_subplot(111, projection='3d')  # Creates a 3D plot ax
    space = g.space                               # Specifies the color space
    g.plot_surface(axis, space)                   # Call the method

intersection\_on\_line():
^^^^^^^^^^^^^^^^^^^^^^^^^

The function receives three parameters. The colourspace, the points in
the c\_data format, and center(if no center is defined, it will use the
default gamut center). The function will return nearest point along a
line between the point and the given center.

::

    points = np.array([[15, 5, 5], [5, 15, 5], [5, 5, 15]])             # Points outside the gamut object
    c_data = data.Data(space.srgb, points)                              # data.Data object
    re_data = g.intersection_on_line(space.srgb, c_data)                # Call the method

clip\_nearest()
^^^^^^^^^^^^^^^

The function receives two parameters, colourspace and colour
data.outside are colour data object and are represented as numpy arrays
of dimensions Nx...xMx3. The function will return nearest point in 3D.

::

    points = np.array([[5, 5, 15], [5, 5, 15], [5, 5, 15]])                   # Points outside the gamut object
    c_data = data.Data(space.srgb, points)                                    # data.Data object
    re_data = g.clip_nearest(space.srgb, c_data)                              # Call the method

compress\_axis()
^^^^^^^^^^^^^^^^

The function receives three parameters. The color space, pints in the
c\_data format, and the axis to compress as integer. The axis range is
[0,1,2] where 0==X, 1==Y and 2==Z.

::

    c = g.compress_axis(space, c_data, axis)    # Call the method

HPminDE()
^^^^^^^^^

The function receives one parameter. The points in the c\_data format,
Maps all points that lie outside of the gamut to the nearest point on
the plane formed by the point and the L axe in the CIELAB colour space.
Returns coordinate for the closest point on plane in the
colourlab.data.Data format.

::

    points = np.array([[0, 8, 8], [4, 0, 9], [4, 4, 3], [0, 10, 0], [15, 0, 0]])    # Points outside the gamut object
    c_data = data.Data(space.cielab, points)                                        # data.Data object
    re_data = g.HPminDE(c_data)                                                     # Call the method

minDE()
^^^^^^^

The function receives one parameter. The points in the c\_data format,
and maps all points that lie outside of the gamut to the nearest point
on the gamut in CIELAB colour space. Returns the nearest point in the
colourlab.data.Data format.

::

    mapped_im = g.minDE(c_data)               # Call the method

Attributes
----------

+------------+---------------------------------------------------------------+
| Attribute  | Description                                                   |
+============+===============================================================+
| data       | The data.Data object used when constructed.                   |
+------------+---------------------------------------------------------------+
| space      | The original colour space used when constructed.              |
+------------+---------------------------------------------------------------+
| hull\*     | The gamuts convex hull in the desired colour space.           |
+------------+---------------------------------------------------------------+
| vertices   | Indices of points forming the vertices of the convex hull.    |
+------------+---------------------------------------------------------------+
| simplices  | Indices of points forming the simplices facets of the convex  |
|            | Hull.                                                         |
+------------+---------------------------------------------------------------+
| neighbors  | Indices of neighbor facets for each facet.                    |
+------------+---------------------------------------------------------------+
| center     | The Gamuts geometric center.                                  |
+------------+---------------------------------------------------------------+

\*see documentation on convex hull for a list of attributes.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html

Methods
-------

-  **is\_inside()**
-  **plot\_surface()**
-  **intersection\_on\_line():**
-  **clip\_nearest()**
-  **compress\_axis()**
-  **HPminDE()**
-  **minDE()**

+-----------------------+--------------------------------------+----------------+
| Method                | Description                          | Return         |
+=======================+======================================+================+
| ``is_inside(sp, c_dat | Returns a boolean array containing   | boolean array  |
| a, t=false)``         | T/F for all points in the array.     |                |
+-----------------------+--------------------------------------+----------------+
| ``plot_surface(ax, sp | Plot the gamut's simplices.          | -              |
| )``                   |                                      |                |
+-----------------------+--------------------------------------+----------------+
| ``intersection_on_lin | Returns the nearest point in a line  | np.array       |
| e(sp, c_data, center= | on a gamut surface from the given    |                |
| None):``              | point to the given center.           |                |
+-----------------------+--------------------------------------+----------------+
| ``_clip_nearest(sp, p | Returns the nearest point on a gamut | np.array       |
| _out, side)``         | in 3D.                               |                |
+-----------------------+--------------------------------------+----------------+
| ``compress_axis(sp, c | Compresses the points linearly in    | colourlab.data |
| _data, ax):``         | the desired axel and colour space.   | .Data          |
|                       |                                      | object         |
+-----------------------+--------------------------------------+----------------+
| ``HPminDE(c_data):``  | Get coordinate for the closest point | colourlab.data |
|                       | on plane and return mapped points    | .Data          |
|                       |                                      | object         |
+-----------------------+--------------------------------------+----------------+
| ``minDE(c_data):``    | Get nearest point and return mapped  | colourlab.data |
|                       | points.                              | .Data          |
|                       |                                      | object         |
+-----------------------+--------------------------------------+----------------+

.. |Build Status| image:: https://travis-ci.org/ifarup/colourlab.svg?branch=master
   :target: https://travis-ci.org/ifarup/colourlab
.. |Code Coverage| image:: https://codecov.io/gh/ifarup/colourlab/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/ifarup/colourlab
.. |Documentation Status| image:: https://readthedocs.org/projects/colourlab/badge/?version=latest
   :target: http://colourlab.readthedocs.io/en/latest/?badge=latest
.. |PyPI version| image:: https://badge.fury.io/py/colourlab.svg
   :target: https://badge.fury.io/py/colourlab
.. |License: GPL v3| image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: http://www.gnu.org/licenses/gpl-3.0
