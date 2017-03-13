colour
======

colour is a Python package for colour metrics and colour space transforms. Many common colour spaces and colour metrics are available, and more are continuously added. A description of the design of the computational framework is available as Farup I. (2016) A computational framework for colour metrics and colour space transforms. _PeerJ Computer Science_ 2:e48 https://doi.org/10.7717/peerj-cs.48

Modules
-------

The package consists of eight modules:

* colour.data
* colour.space
* colour.metric
* colour.tensor
* colour.statistics
* colour.misc
* colour.image
* colour.gamut

All the modules are imported when importing the package. The basic functionality of supposedly general interest is found in the three first modules. Only the really basic functionality is documented here. For more advanced features, please refer to the code (which is documented with standard pydoc docstrings), or contact the author.

Representing and Converting Colour Data
---------------------------------------

Basic numerical colour data are represented as numpy arrays of dimensions Nx...xMx3. In other words, colour data can be of any dimension, as long as the last dimension is the colour dimension. In particular, single points in the colour space will be ndarrays of shape (3,), lists of colour data will have dimension (N,3), and colour images will have dimension (M,N,3).

Colour data is scaled such that the natural maximum value is unity for most colour spaces, including XYZ having Y=1 for the whichever white point, and the RGB spaces having (1,1,1) as white points. For colour spaces with explicitly defined scaling like CIELAB and CIELAB (where the white point is defined as L*=100), the original scaling is used.

Colour data, i.e., colour points, lists of colours, colour images etc., are most conveniently represented by objects of the colour.data.Data class. In order to construct such an object, the colour data and the colour space has to be specified. Typically, if `col_xyz` is an ndarray with colour data in XYZ, a colour data object can be constructed as

```python
col_data = colour.data.Data(colour.space.xyz, col_xyz)
```

Then the corresponding ndarray data in other colour spaces can be acquired by the get method:

```python
col_srgb = col_data.get(colour.space.srgb)
col_lab  = col_data.get(colour.space.cielab)
col_xyY  = col_data.get(colour.space.xyY)
```

and so on. The colour conversions are computed only once and buffered within the Data object, so no extra overhead (besides the function call) is caused by sequential calls to the get method with the same colour space as the argument. Currently, the following colour spaces are available:

* **colour.space.xyz**: The CIE XYZ colour space.
* **colour.space.xyY**: The CIE xyY colour space.
* **colour.space.cielab**: The CIELAB colour space with D65 white point.
* **colour.space.cielch**: Polar coordinates in the CIELAB colour space.
* **colour.space.cieluv**: The CIELUV colour space with D65 white point.
* **colour.space.ciecat02**: The colour space of the CIECAT02 colour adaptation transform.
* **colour.space.ciede00lab**: The underlying colourspace of the CIEDE2000 colour metric.
* **colour.space.srgb**: The sRGB colour space.
* **colour.space.rgb_adobe**: The Adobe RGB colour space.
* **colour.space.ipt**: The IPT colour space.
* **colour.space.lgj_osa**: The OSA-UCS colour space.
* **colour.space.lgj_e**: The Euclidised OSA-UCS colour space used in the &Delta;E<sub>E</sub> metric.
* **colour.space.din99x**: The various DIN99x colour spaces for the corresponding metrics (x is empty, b, c, or d).

There are also built-in colour data sets available. They are all represented by Data objects that can be constructed upon need by functions in the colour.data module. These functions have names starting with `d_`. Most of these data sets are mainly of interest for colour metrics researcher, but some of them will have broader interest, such as the various CIE colour matching functions, and the data of the Munsell patches.

Computing Colour Metrics
------------------------

The most common colour metrics are available as functions in the colour.metrics module. All the metric funtions take two colour Data objects as parameters. The two objects must be of the same dimension. If the colour data in the Data objects are of the dimension Nx...xMx3, the return value of the metric functions are ndarrays of dimension Nx...XM. For example, the &Delta;E<sub>ab</sub> colour difference between the two datasets `dataset1` and `dataset2` is computed as

```python
diff = colour.metric.dE_ab(dataset1, dataset2)
```

The following metrics are available:

* **colour.metric.dE_ab**: The CIE76 standard &Delta;E<sub>ab</sub> &ndash; the Euclidean distance in CIELAB.
* **colour.metric.dE_uv**: The Euclidean metric in CIELUV, &Delta;E<sub>uv</sub>.
* **colour.metric.dE_00**: The CIEDE2000 non-Euclidean colour metric.
* **colour.metric.dE_E**: The Euclidean colour metric &Delta;E<sub>E</sub>.
* **colour.metric.dE_DIN99x**: The DIN99x colour metrics (where x is empty, b, c or d).

Additionally, a general Euclidean colour metric in a given colour space can be computed as

```python
my_diff = colour.metric.euclidean(my_colour_space, dataset1, dataset2)
```

Constructing Colour Spaces
--------------------------

New colour space objects are constructed by starting with an exisiting base colour space, and applying colour space transformations. For example, the fictive colour space `my_rgb` can be defined by a linear transformation from XYZ using the matrix `my_M` followed by a gamma correction with `my_gamma` as follows

```python
my_rgb_linear = colour.space.TransformLinear(colour.space.xyz, my_M)
my_rgb = colour.space.TransformGamma(my_rgb_linear, my_gamma)
```

Any existing colour space can be used as the base for the transformation. Currently, the following common colour space transformations have been defined:

* **colour.space.TransformLinear**: A linear transformation defined by a 3x3 matrix (represented as a ndarray of shape (3,3)).
* **colour.space.TransformGamma**: A gamma correction applied individually to all channels. If the channel values are negative (should not be the case for RGB type spaces, but occurs, e.g., in IPT), the gamma transform is applied to the absolute value of the channel, and the sign is added in the end.
* **colour.space.TransformPolar**: Transform the two last colour coordinates from cartesian to polar. This can be used, e.g., to transform from CIELAB to CIELCH. Keep in mind, though, that this transformation is singular at the origin of the chromatic plane, and that this can confuse some colour metrics. The angle is represented in radians.
* **colour.space.TransformCartesian**: The oposite of the above.
* **colour.space.TransformxyY**: The projective transform from, e.g., XYZ to xyY, with that order of the coordinates.
* **colour.space.TransformCIELAB**: The non-linear transformation from XYZ to CIELAB, including the linear part for small XYZ values. The white point is a parameter, so this transformation can be used also to create CIELAB D50, etc. The base space does not have to be XYZ, so this transform can also be used to create, e.g., the DIN99 colour spaces.
* **colour.space.TransformCIELUV**: The non-linear transformation from XYZ to CIELUV, including the linear part for small XYZ values.The white point is a parameter, so this transformation can be used also to create CIELUV D50, etc.
* **colour.space.TransformSRGB**: The non-linear gamma-like correction from linear RGB with sRGB primaries to the non-linear sRGB colour space.

In addition, the following more special colour space transforms are available:

* colour.space.TransformLogCompressL
* colour.space.TransformLogCompressC
* colour.space.TransformPoincareDisk
* colour.space.TransformCIEDE00
* colour.space.TransformLGJOSA
* colour.space.TransformLGJE

Adding to this, new colour space transforms can be defined as classes inheriting from the colour.space.Transform base class.

Common white points are available as the following Data objects:

* colour.space.white_A
* colour.space.white_B
* colour.space.white_C
* colour.space.white_D50
* colour.space.white_D55
* colour.space.white_D65
* colour.space.white_D75
* colour.space.white_E
* colour.space.white_F2
* colour.space.white_F7
* colour.space.white_F11

# Gamut
***

## Attributes

| Attribute      | Description                    
| -------------  | ------------------------------------------------------------------------------
data             | The data.Data object used when constructed.
space            | The original colour space used when constructed.
hull*            | The gamuts convex hull in the desired colour space.
vertices         | Indices of points forming the vertices of the convex hull.
simplices        | Indices of points forming the simplices facets of the convex Hull.
neighbors        | Indices of neighbor facets for each facet.
center           | The Gamuts geometric center.

 *see documentation on convex hull for a list of attributes.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html

## Constructing Gamut 
To construct a new Gamut we need to provide a colour space in the format provided by colour.space, 
and data/colour points in the format given the colour.data.Data class. If we want to construct the new Gamut in the colourspace RGB and the fictive points my_points, we would do it as follows

#### For convex hull
```python
c_data = data.Data(space.srgb, my_points)    # First generate the Data objekt to use
g = gamut.Gamut(space.srgb, c_data)          # Pass along the colourspace and c_data
```
#### For modified-convex hull
When using the modified constructor, we have to choose an exponent for modifying the gamut radius(gamma), and define a center for expansion.
```python
c_data = data.Data(space.srgb, my_points)                        # First generate the Data objekt to use
g = gamut.Gamut(space.srgb, c_data, gamma=0.2, center=my_center) # Pass along the colourspace, c_data, gamma and center 
```

## Methods

* **is_inside()**
* **plot_surface()**
* **intersectionpoint_on_line()**

Method      	                              | Description
--------------------------------------------  | -----------------------------------------------------------
`is_inside(sp, c_data)`                       | Returns a boolean array containing T/F for all points in the array.        
`plot_surface(ax, sp)`                        | Plot the gamut's simplices.
`intersectionpoint_on_line(d, center, sp)`    | Return the nearest point on a gamut serface for the given point.