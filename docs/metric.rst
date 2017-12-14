colourlab.metric
================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

The most common colour metrics are available as functions in the
colourlab.metrics module. All the metric funtions take two colour Data
objects as parameters. The two objects must be of the same dimension. If
the colour data in the Data objects are of the dimension Nx...xMx3, the
return value of the metric functions are ndarrays of dimension Nx...XM.
For example, the ΔEab colour difference between the two datasets
``dataset1`` and ``dataset2`` is computed as

.. code:: python

    diff = colourlab.metric.dE_ab(dataset1, dataset2)

The following metrics are available:

-  **colourlab.metric.dE\_ab**: The CIE76 standard ΔEab – the Euclidean
   distance in CIELAB.
-  **colourlab.metric.dE\_uv**: The Euclidean metric in CIELUV, ΔEuv.
-  **colourlab.metric.dE\_00**: The CIEDE2000 non-Euclidean colour
   metric.
-  **colourlab.metric.dE\_E**: The Euclidean colour metric ΔEE.
-  **colourlab.metric.dE\_DIN99x**: The DIN99x colour metrics (where x
   is empty, b, c or d).

Additionally, a general Euclidean colour metric in a given colour space
can be computed as

.. code:: python

    my_diff = colourlab.metric.euclidean(my_colour_space, dataset1, dataset2)

