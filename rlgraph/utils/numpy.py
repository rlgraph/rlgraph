# Copyright 2018 The RLgraph authors. All Rights Reserved.
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

import numpy as np


def sigmoid(x, derivative=False):
    """
    Returns the sigmoid function applied to x.
    Alternatively, can return the derivative or the sigmoid function.

    Args:
        x (np.ndarray): The input to the sigmoid function.
        derivative (bool): Whether to return the derivative or not. Default: False.

    Returns:
        np.ndarray: The sigmoid function (or its derivative) applied to x.
    """
    if derivative:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """
    Returns the softmax values for x as:
    S(xi) = e^xi / SUMj(e^xj), where j goes over all elements in x.

    Thanks to alvas for the trick with the max for numerical stability:
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

    Args:
        x (np.ndarray): The input to the softmax function.
        axis (int): The axis along which to softmax.

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


def one_hot(x, depth=0, on_value=1, off_value=0):
    """
    One-hot utility function for numpy.
    Thanks to qianyizhang:
    https://gist.github.com/qianyizhang/07ee1c15cad08afb03f5de69349efc30.

    Args:
        x (np.ndarray): The input to be one-hot.
        depth (int): The max. number to be one-hot encoded (size of last rank).
        on_value (float): The value to use for on. Default: 1.0.
        off_value (float): The value to use for off. Default: 0.0.

    Returns:
        np.ndarray: The one-hot encoded equivalent of the input array.
    """
    if depth == 0:
        depth = np.max(x) + 1
    assert np.max(x) < depth, "ERROR: The max. index of `x` ({}) is larger than depth ({})!".format(np.max(x), depth)
    shape = x.shape

    # Python 2.7 compatibility, (*shape, depth) is not allowed.
    shape_list = [dim for dim in shape]
    shape_list.append(depth)
    out = np.ones(shape_list) * off_value
    indices = []
    for i in range(x.ndim):
        tiles = [1] * x.ndim
        s = [1] * x.ndim
        s[i] = -1
        r = np.arange(shape[i]).reshape(s)
        if i > 0:
            tiles[i-1] = shape[i-1]
            r = np.tile(r, tiles)
        indices.append(r)
    indices.append(x)
    out[tuple(indices)] = on_value
    return out
