colour
======

colour is a Python package for colour metrics and colour space transforms. The package consists of six modules:

* colour.data
* colour.space
* colour.metric
* colour.tensor
* colour.statistics
* colour.misc

All the modules are imported when importing the package.

The main functionality is found in the two modules colour.data and colour.space. Many common colour spaces are available as predefined objects in colour.space (as, e.g., colour.space.cielab, colour.space.xyz, colour.space.srgb etc.), and more are continuously added.

Representing and Converting Colour Data
---------------------------------------

Basic numeric colour data are represented as numpy arrays of dimensions NxMx...Px3. In other words, colour data can be of any dimension, as long as the last dimension is the colour dimension. In particular, single points in the colour space will be ndarrays of shape (3,), lists of colour data will have dimension (N,3), and colour images will have dimension (M,N,3).

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

and so on. The colour conversions are computed only once and buffered within the Data object, so there is no extra overhead (besides the function call) by sequential calls to the get method with the same colour space as the argument.

There are also built-in colour data sets available. They are all represented by Data objects that can be constructed upon need by functions in the colour.data module. These functions have names starting with `d_`. Most of these data sets are mainly of interest for colour metrics researcher, but some of them will have broader interest, such as the various CIE colour matching functions, and the data of the Munsell patches.
