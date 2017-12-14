.. colourlab documentation master file, created by
   sphinx-quickstart on Tue May 23 21:39:55 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

colourlab
=========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: colourlab

**This documentation is currently under construction**

Overview
========

colourlab is a Python package for colour metrics and colour space
transforms. Many common colour spaces and colour metrics are
available, and more are continuously added. A description of the
design of the original design of the computational framework is
available as `Farup I. (2016) A computational framework for colour
metrics and colour space
transforms. <https://doi.org/10.7717/peerj-cs.48>`_ *PeerJ Computer
Science* 2:e48

The package consists of eight modules:

* :doc:`space`
* :doc:`data`
* :doc:`tensor`
* :doc:`metric`
* :doc:`image`
* :doc:`gamut`
* :doc:`statistics`
* :doc:`misc`

All the modules are imported when importing the package.

Basic numerical colour data are represented as numpy arrays of
dimensions Nx...xMx3. In other words, colour data can be of any
dimension, as long as the last dimension is the colour dimension. In
particular, single points in the colour space will be ndarrays of shape
(3,), lists of colour data will have dimension (N,3), and colour images
will have dimension (M,N,3).

Colour data is scaled such that the natural maximum value is unity for
most colour spaces, including XYZ having Y=1 for the whichever white
point, and the RGB spaces having (1,1,1) as white points. For colour
spaces with explicitly defined scaling like CIELAB and CIELAB (where the
white point is defined as L\*=100), the original scaling is used.

Colour data, i.e., colour points, lists of colours, colour images etc.,
are most conveniently represented by objects of the colourlab.data.Data
class. In order to construct such an object, the colour data and the
colour space has to be specified. Typically, if ``col_xyz`` is an
ndarray with colour data in XYZ, a colour data object can be constructed
as

.. code:: python

    col_data = colourlab.data.Data(colourlab.space.xyz, col_xyz)

Then the corresponding ndarray data in other colour spaces can be
acquired by the get method:

.. code:: python

    col_srgb = col_data.get(colourlab.space.srgb)
    col_lab  = col_data.get(colourlab.space.cielab)
    col_xyY  = col_data.get(colourlab.space.xyY)

and so on. The colour conversions are computed only once and buffered
within the Data object, so no extra overhead (besides the function call)
is caused by sequential calls to the get method with the same colour
space as the argument. Currently, the following colour spaces are
available:

-  **colourlab.space.xyz**: The CIE XYZ colour space.
-  **colourlab.space.xyY**: The CIE xyY colour space.
-  **colourlab.space.cielab**: The CIELAB colour space with D65 white
   point.
-  **colourlab.space.cielch**: Polar coordinates in the CIELAB colour
   space.
-  **colourlab.space.cieluv**: The CIELUV colour space with D65 white
   point.
-  **colourlab.space.ciecat02**: The colour space of the CIECAT02 colour
   adaptation transform.
-  **colourlab.space.ciede00lab**: The underlying colourlab of the
   CIEDE2000 colour metric.
-  **colourlab.space.srgb**: The sRGB colour space.
-  **colourlab.space.rgb\_adobe**: The Adobe RGB colour space.
-  **colourlab.space.ipt**: The IPT colour space.
-  **colourlab.space.lgj\_osa**: The OSA-UCS colour space.
-  **colourlab.space.lgj\_e**: The Euclidised OSA-UCS colour space used
   in the ΔEE metric.
-  **colourlab.space.din99x**: The various DIN99x colour spaces for the
   corresponding metrics (x is empty, b, c, or d).

There are also built-in colour data sets available. They are all
represented by Data objects that can be constructed upon need by
functions in the colourlab.data module. These functions have names
starting with ``d_``. Most of these data sets are mainly of interest for
colour metrics researcher, but some of them will have broader interest,
such as the various CIE colour matching functions, and the data of the
Munsell patches.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
