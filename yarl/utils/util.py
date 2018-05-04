# Copyright 2018 The YARL-Project, All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import itertools


from .yarl_error import YARLError


def dtype(dtype_, to="tf"):
    """
    Translates any type (tf, numpy, python, etc..) into the respective tensorflow/numpy data type.

    Args:
        dtype_ (any): String describing a numerical type (e.g. 'float'), numpy data type, tf dtype,
            or python numerical type.
        to (str): Either one of 'tf' or 'np' (default="tf").

    Returns: TensorFlow or Numpy data type (depending on `to` parameter).
    """

    if dtype_ in ["float", float, np.float32, tf.float32]:
        return tf.float32 if to == "tf" else np.float32
    elif dtype_ in ["int", int, np.int32, tf.int32]:
        return tf.int32 if to == "tf" else np.int32
    elif dtype_ in ["bool", bool, np.bool_, tf.bool]:
        return tf.bool if to == "tf" else np.bool
    else:
        raise YARLError("Error: Type conversion to '{}' for type '{}' not supported.".format(to, str(dtype_)))


def force_list(elements):
    """
    Makes sure elements is returned as a list, whether elements is a single item, already a list, or a tuple.

    Args:
        elements (any): The input single item, list, or tuple to be converted into a list.

    Returns:
        All given elements in a list.
    """
    return list(elements) if isinstance(elements, (list, tuple)) else [elements]


def all_combinations(input_list, descending=False):
    """
    Returns a list containing sorted tuples of all possible combinations of all possible length of the elements
    in the input_list (without repeating elements inside each tuple).
    The returned list is sorted by the length of the combination tuples (in descending (longest first) or ascending
    (shortest first) order).
    Example:
         input_list=[1, 2, 3] returns: [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
         input_list=[5, 1]    returns: [(5,), (1,), (1,5)]

    Args:
        input_list (list): The list to get all combinations for.
        descending (bool): Whether to sort the tuples longest first (default: shortest first).

    Returns:
        The list of tuples of possible combinations.
    """
    return sorted(list(itertools.chain.from_iterable(
        itertools.combinations(input_list, i+1) for i in range(len(input_list)))),
        key=len, reverse=descending)
