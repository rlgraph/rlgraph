.. Copyright 2018 The RLgraph authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   ============================================================================

.. image:: images/rlcore-logo-full.png
   :scale: 25%
   :alt:

The Space Classes
=================

What is a Space?
----------------

Spaces in RLgraph are used to define the types and shapes of data flowing into, between, and from the different
machine learning components. For example, an environment may output an RGB color image (e.g. 80x80 pixels) at any
time step, which our RL algorithm will use as its only input to determine the next action.
An RGB color image is usually represented by an IntBox Space with the exact data type being uint8
(value can range from 0 to 255) and with a shape of [80 x 80 x 3], where the 80 represent the width and height
of the image and the 3 represents the three color channels: red, green, and blue.
Values of 0 mean no intensity in a color channel, 255 means full intensity.

*Difference between Space and Tensor*
Tensors are the members or instances of a Space. Tensors hold actual numeric values, which adhere to the rules and
bounds of the given Space in terms of data type and dimensionality. For example, a specific RGB image coming from
our environment above is a tensor (or an instance) of the IntBox-[80 x 80 x 3] space.

*Rank, dimension and shape*
The rank of our example image space is 3, since we have - for each image - a 3D cube of numbers, which represents that
image. The 3 ranks stand for the width, height and color depth of the image.
A Space's rank is often confused with it's dimensions. In RLgraph (as well as e.g. in tensorflow), we speak of
dimensions only as the size of each rank: In our example, the dimensions are 80 (1st rank), 80 (2nd rank),
and 3 (3rd rank).
Another often used term for the set of all
dimension numbers is "shape" and it's often provided as a tuple of numbers. The shape of our image is (80, 80, 3) and
this shape tuple is sufficient to determine both a space's rank (len(shape)) and its dimensions (shape[0], shape[1],
and shape[2]).


There are two major types of Spaces: BoxSpaces and ContainerSpaces.
-------------------------------------------------------------------

Box Spaces
++++++++++

A BoxSpace is simply an n-dimensional cube of numbers (or strings), where the numbers must all be of the same data type
("dtype" from here on). RLgraph's dtypes are based on the numpy type system and supported types are np.ints (such as
np.int32 or np.uint8), np.floats (e.g. np.float32), np.bool\_, as well as a box type for String data, which has the
dtype np.string\_. The dimensionality of a box can be anything from 0D (a single scalar value), 1D (a vector of values),
2D (a matrix of values), 3D (a cube), to any higher dimensional box.


Container Spaces
++++++++++++++++

Container Spaces can contain Box Spaces as well as other Container Spaces in an arbitrarily nested fashion. The two
supported container types are Tuple and Dict.

A Tuple is an ordered sequence of other Spaces (similar to a python tuple). For example:
An environment that produces an RGB image and a
text string at each time step could have the space: Tuple(IntBox(80,80,3, np.uint8), TextBox()).
Another way to describe this space is through a keyed Dict space (similar to python dicts), with the keys
"image" and "text". For example: Dict({"image": IntBox(80,80,3, np.uint8), "text": TextBox()}).

Containers are fully equivalent to Box Space classes in that they also have shapes and dtypes. However, these are
represented by heterogeneous tuples. Our Image-and-Text Dict space from above, for example, would have a shape of
((80,80,3),()), a rank of (3, 0) and a dtype of (np.uint8, np.string\_).
Note here that for Dict spaces, the order of the keys are sorted alphabetically before generating shape, rank and
dtype tuples. In this case: The "image" key comes before the "text" key. For Tuple spaces this order is given by
the sequence of sub-Spaces inside the Tuple. Nested Container Spaces (e.g. a Dict inside another Dict) generate
equally nested shape, rank and dtype tuples.


Special Ranks of BoxSpaces.
---------------------------

TODO: Describe the generic special ranks once we have them implemented (will replace the current special ranks: batch
and time).

