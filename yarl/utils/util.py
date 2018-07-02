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

from functools import partial
import logging
import numpy as np
import inspect
import itertools
import re
from six.moves import xrange
import sys
import tensorflow as tf

from yarl.backend_system import get_backend
from yarl.utils.yarl_error import YARLError

if get_backend() == "tf":
    import tensorflow as be
else:
    import pytorch as be

SMALL_NUMBER = 1e-6
LARGE_INTEGER = 100000000

# Logging config for testing.
logging_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%y-%m-%d %H:%M:%S')
root_logger = logging.getLogger('')
root_logger.setLevel(level=logging.DEBUG)
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(level=logging.DEBUG)

print_logging_handler = logging.StreamHandler(stream=sys.stdout)
print_logging_handler.setFormatter(logging_formatter)
print_logging_handler.setLevel(level=logging.DEBUG)
root_logger.addHandler(print_logging_handler)


def dtype(dtype_, to="tf"):
    """
    Translates any type (tf, numpy, python, etc..) into the respective tensorflow/numpy data type.

    Args:
        dtype_ (any): String describing a numerical type (e.g. 'float'), numpy data type, tf dtype,
            pytorch data-type, or python numerical type.
        to (str): Either one of 'tf' (tensorflow), 'pt' (pytorch), or 'np' (numpy). Default="tf".

    Returns:
        TensorFlow or Numpy data type (depending on `to` parameter).
    """
    # bool: tensorflow
    if get_backend() == "tf":
        if dtype_ in ["bool", bool, np.bool_, be.bool]:
            return np.bool_ if to == "np" else be.bool
    # bool: pytorch backend (not supported)
    elif dtype_ in ["bool", bool, np.bool_]:
        return np.bool_ if to == "np" else tf.bool

    # generic backend
    if dtype_ in ["float", "float32", float, np.float32, be.float32]:
        return np.float32 if to == "np" else tf.float32
    if dtype_ in ["float64", np.float64, be.float64]:
        return np.float64 if to == "np" else tf.float64
    elif dtype_ in ["int", "int32", int, np.int32, be.int32]:
        return np.int32 if to == "np" else tf.int32
    elif dtype_ in ["int64", np.int64]:
        return np.int64 if to == "np" else tf.int64

    raise YARLError("Error: Type conversion to '{}' for type '{}' not supported.".format(to, str(dtype_)))


def get_rank(tensor):
    """
    Returns the rank (as a single int) of an input tensor.

    Args:
        tensor (any): The input tensor.

    Returns:
        int: The rank of the given tensor.
    """
    if get_backend() == "tf":
        return tensor.get_shape().ndims


def get_shape(op, flat=False, no_batch=False):
    """
    Returns the (static) shape of the tensor as a tuple.

    Args:
        op (DataOp): The input op.
        flat (bool): Whether to return the flattened shape (the product of all ints in the shape tuple).
            Default: False.
        no_batch (bool): Whether to exclude a possible 0th batch rank (None) from the returned shape.
            Default: False.

    Returns:
        tuple: The shape of the given op.
        int: The flattened dim of the given op (if flat=True).
    """
    # dict
    if isinstance(op, dict):
        shape = tuple([get_shape(op[key]) for key in sorted(op.keys())])
    # tuple-op
    elif isinstance(op, tuple):
        shape = tuple([get_shape(i) for i in op])
    # primitive op (e.g. tensorflow)
    else:
        op_shape = op.get_shape()
        # Unknown shape (e.g. a cond op).
        if op_shape.ndims is None:
            return None
        shape = tuple(op_shape.as_list())

    # Remove batch rank?
    if no_batch is True and shape[0] is None:
        shape = shape[1:]

    if flat is False:
        return shape
    else:
        return np.prod(shape)


def get_batch_size(tensor):
    """
    Returns the (dynamic) batch size (dim of 0th rank) of an input tensor.

    Args:
        tensor (SingleDataOp): The input tensor.

    Returns:
        SingleDataOp: The op holding the batch size information of the given tensor.
    """
    if get_backend() == "tf":
        return tf.shape(tensor)[0]


def force_list(elements, to_tuple=False):
    """
    Makes sure `elements` is returned as a list, whether `elements` is a single item, already a list, or a tuple.

    Args:
        elements (Optional[any]): The input single item, list, or tuple to be converted into a list/tuple.
            If None, returns empty list/tuple.
        to_tuple (bool): Whether to use tuple (instead of list).

    Returns:
        Union[list,tuple]: All given elements in a list/tuple depending on `to_tuple`'s value.
    """
    ctor = list
    if to_tuple is True:
        ctor = tuple
    return ctor() if elements is None else ctor(elements) \
        if type(elements) in [list, tuple] else ctor([elements])


force_tuple = partial(force_list, to_tuple=True)


def default_dict(original, defaults):
    """
    Updates the original dict with values from `defaults`, but only for those keys that
    do not exist yet in `original`.
    Changes `original` in place, but leaves `defaults` as is.

    Args:
        original (Optional[dict]): The dict to (soft)-update. If None, return `defaults`.
        defaults (dict): The dict to update from.
    """
    if original is None:
        return defaults

    for key in defaults:
        if key not in original:
            original[key] = defaults[key]
    return original


def clamp(x, min_, max_):
    """
    Clamps x between min_ and max_.

    Args:
        x (float): The input to be clamped.
        min_ (float): The min value for x.
        max_ (float): The max value for x.

    Returns:
        float: The clamped value.
    """
    return max(min_, min(x, max_))


def softmax(x, axis=-1):
    """
    Returns the softmax values for x as:
    S(xi) = e^xi / SUMj(e^xj), where j goes over all elements in x.

    Thanks to alvas for the trick with the max for numerical stability:
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

    Args:
        x (np.ndarray): The input to the softmax function.

    Returns:
        np.ndarray: The softmax over x.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def relu(x, alpha=0.0):
    """
    Implementation of the leaky ReLU function:
    y = x * alpha if x < 0 else x

    Args:
        x (np.ndarray): The input values.
        alpha (float): A scaling ("leak") factor to use for negative x.

    Returns:
        np.ndarray: The leaky ReLU output for x.
    """
    return np.maximum(x, x*alpha, x)


def get_method_type(method):
    """
    Returns either "graph_fn" OR "api" OR "other" depending on which method (and method
    name) is passed in.

    Args:
        method (callable): The actual method to analyze.

    Returns:
        Union[str,None]: "graph_fn", "api" or "other". None if method is not a callable.
    """
    # Not a callable: Return None.
    if not callable(method):
        return None
    # Simply recognize graph_fn by their name.
    elif method.__name__[:9] == "_graph_fn":
        return "graph_fn"
    # Figure out API-methods by their source code: Have calls to `Component.call`.
    elif method.__name__[0] != "_" and method.__name__ != "define_api_method" and \
        re.search(r'\.call\(', inspect.getsource(method)):
        return "api"
    else:
        return "unknown"


def get_num_return_values(method):
    """
    Does a regexp-based source code inspection and tries to figure out the number of values that the method
    will return (assuming, that the method always returns the same number of values, regardless of its inputs).

    Args:
        method (callable): The method to get the number of returns  values for.

    Returns:
        int: The number of return values of `method`.
    """
    src = inspect.getsource(method)
    # Resolve '\' at end of lines.
    src = re.sub(r'\\\s*\n', "", src)

    mo = re.search(r'.*\breturn (.+)', src, flags=re.DOTALL)
    if mo:
        return_code = mo.group(1)
        # TODO: raise error if "tuple()" -> means we don't know how many return value we will have
        # TODO: in this case, the Component needs to specify how many output records it produces.
        # Resolve tricky things (parentheses, etc..).
        while re.search(r'[\(\)"]', return_code):
            return_code = re.sub(r'\([^\(\)]*\)', '', return_code)
            return_code = re.sub(r'"[^"]*"', '', return_code)
            # Probably have to add more resolution code here.
            # ...

        # Count the commas and return.
        num_return_values = len(return_code.split(","))
        return num_return_values
    else:
        return 0
