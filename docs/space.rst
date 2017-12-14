colourlab.space
===============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

New colour space objects are constructed by starting with an exisiting
base colour space, and applying colour space transformations. For
example, the fictive colour space ``my_rgb`` can be defined by a linear
transformation from XYZ using the matrix ``my_M`` followed by a gamma
correction with ``my_gamma`` as follows

.. code:: python

    my_rgb_linear = colourlab.space.TransformLinear(colourlab.space.xyz, my_M)
    my_rgb = colourlab.space.TransformGamma(my_rgb_linear, my_gamma)

Any existing colour space can be used as the base for the
transformation. Currently, the following common colour space
transformations have been defined:

-  **colourlab.space.TransformLinear**: A linear transformation defined
   by a 3x3 matrix (represented as a ndarray of shape (3,3)).
-  **colourlab.space.TransformGamma**: A gamma correction applied
   individually to all channels. If the channel values are negative
   (should not be the case for RGB type spaces, but occurs, e.g., in
   IPT), the gamma transform is applied to the absolute value of the
   channel, and the sign is added in the end.
-  **colourlab.space.TransformPolar**: Transform the two last colour
   coordinates from cartesian to polar. This can be used, e.g., to
   transform from CIELAB to CIELCH. Keep in mind, though, that this
   transformation is singular at the origin of the chromatic plane, and
   that this can confuse some colour metrics. The angle is represented
   in radians.
-  **colourlab.space.TransformCartesian**: The oposite of the above.
-  **colourlab.space.TransformxyY**: The projective transform from,
   e.g., XYZ to xyY, with that order of the coordinates.
-  **colourlab.space.TransformCIELAB**: The non-linear transformation
   from XYZ to CIELAB, including the linear part for small XYZ values.
   The white point is a parameter, so this transformation can be used
   also to create CIELAB D50, etc. The base space does not have to be
   XYZ, so this transform can also be used to create, e.g., the DIN99
   colour spaces.
-  **colourlab.space.TransformCIELUV**: The non-linear transformation
   from XYZ to CIELUV, including the linear part for small XYZ
   values.The white point is a parameter, so this transformation can be
   used also to create CIELUV D50, etc.
-  **colourlab.space.TransformSRGB**: The non-linear gamma-like
   correction from linear RGB with sRGB primaries to the non-linear sRGB
   colour space.

In addition, the following more special colour space transforms are
available:

-  colourlab.space.TransformLogCompressL
-  colourlab.space.TransformLogCompressC
-  colourlab.space.TransformPoincareDisk
-  colourlab.space.TransformCIEDE00
-  colourlab.space.TransformLGJOSA
-  colourlab.space.TransformLGJE

Adding to this, new colour space transforms can be defined as classes
inheriting from the colourlab.space.Transform base class.

Common white points are available as the following Data objects:

-  colourlab.space.white\_A
-  colourlab.space.white\_B
-  colourlab.space.white\_C
-  colourlab.space.white\_D50
-  colourlab.space.white\_D55
-  colourlab.space.white\_D65
-  colourlab.space.white\_D75
-  colourlab.space.white\_E
-  colourlab.space.white\_F2
-  colourlab.space.white\_F7
-  colourlab.space.white\_F11
