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

from functools import partial

from rlgraph import get_backend
from rlgraph.utils.rlgraph_errors import RLGraphError


if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch.nn as nn


def get_activation_function(activation_function=None, *other_parameters):
    """
    Returns an activation function (callable) to use in a NN layer.

    Args:
        activation_function (Optional[callable,str]): The activation function to lookup. Could be given as:
            - already a callable (return just that)
            - a lookup key (str)
            - None: Use linear activation.

        other_parameters (any): Possible extra parameter(s) used for some of the activation functions.

    Returns:
        callable: The backend-dependent activation function.
    """
    if get_backend() == "tf":
        if activation_function is None or callable(activation_function):
            return activation_function
        elif activation_function == "linear":
            return tf.identity
        # Rectifier linear unit (ReLU) : 0 if x < 0 else x
        elif activation_function == "relu":
            return tf.nn.relu
        # Exponential linear: exp(x) - 1 if x < 0 else x
        elif activation_function == "elu":
            return tf.nn.elu
        # Sigmoid: 1 / (1 + exp(-x))
        elif activation_function == "sigmoid":
            return tf.sigmoid
        # Scaled exponential linear unit: scale * [alpha * (exp(x) - 1) if < 0 else x]
        # https://arxiv.org/pdf/1706.02515.pdf
        elif activation_function == "selu":
            return tf.nn.selu
        # Swish function: x * sigmoid(x)
        # https://arxiv.org/abs/1710.05941
        elif activation_function == "swish":
            return lambda x: x * tf.sigmoid(x=x)
        # Leaky ReLU: x * [alpha if x < 0 else 1.0]
        elif activation_function in ["lrelu", "leaky_relu"]:
            alpha = other_parameters[0] if len(other_parameters) > 0 else 0.2
            return partial(tf.nn.leaky_relu, alpha=alpha)
        # Concatenated ReLU:
        elif activation_function == "crelu":
            return tf.nn.crelu
        # Softmax function:
        elif activation_function == "softmax":
            return tf.nn.softmax
        # Softplus function:
        elif activation_function == 'softplus':
            return tf.nn.softplus
        # Softsign function:
        elif activation_function == "softsign":
            return tf.nn.softsign
        # tanh activation function:
        elif activation_function == "tanh":
            return tf.nn.tanh
        else:
            raise RLGraphError("ERROR: Unknown activation_function '{}' for TensorFlow backend!".
                               format(activation_function))
    elif get_backend() == "pytorch":
        # Have to instantiate objects here.
        if activation_function is None or callable(activation_function):
            return activation_function
        elif activation_function == "linear":
            # Do nothing.
            return None
        # Rectifier linear unit (ReLU) : 0 if x < 0 else x
        elif activation_function == "relu":
            return nn.ReLU()
        # Exponential linear: exp(x) - 1 if x < 0 else x
        elif activation_function == "elu":
            return nn.ELU()
        # Sigmoid: 1 / (1 + exp(-x))
        elif activation_function == "sigmoid":
            return nn.Sigmoid()
        # Scaled exponential linear unit: scale * [alpha * (exp(x) - 1) if < 0 else x]
        # https://arxiv.org/pdf/1706.02515.pdf
        elif activation_function == "selu":
            return nn.SELU()
        # Leaky ReLU: x * [alpha if x < 0 else 1.0]
        elif activation_function in ["lrelu", "leaky_relu"]:
            alpha = other_parameters[0] if len(other_parameters) > 0 else 0.2
            return partial(nn.LeakyReLU(), alpha=alpha)
        # Softmax function:
        elif activation_function == "softmax":
            return nn.Softmax()
        # Softplus function:
        elif activation_function == 'softplus':
            return nn.Softplus()
        # Softsign function:
        elif activation_function == "softsign":
            return nn.Softsign()
        # tanh activation function:
        elif activation_function == "tanh":
            return nn.Tanh()
        else:
            raise RLGraphError("ERROR: Unknown activation_function '{}' for PyTorch backend!".
                               format(activation_function))
