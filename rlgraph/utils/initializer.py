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

import math
import numpy as np

from rlgraph import get_backend
from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.utils.specifiable import Specifiable
from rlgraph.utils.util import dtype

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class Initializer(Specifiable):
    def __init__(self, shape, specification=None, **kwargs):
        """
        Args:
            shape (tuple): The shape of the Variables to initialize.
            specification (any): A spec that determines the nature of this initializer.

        Raises:
            RLGraphError: If a fixed shape in `specification` does not match `shape`.
        """
        super(Initializer, self).__init__()

        # The shape of the variable to be initialized.
        self.shape = shape
        # The actual underlying initializer object.
        self.initializer = None

        # Truncated Normal.
        if specification == "truncated_normal":
            if get_backend() == "tf":
                # Use the first dimension (num_rows or batch rank) to figure out the stddev.
                stddev = 1 / math.sqrt(shape[0] if isinstance(shape, (tuple, list)) and len(shape) > 0 else 1.0)
                self.initializer = tf.truncated_normal_initializer(stddev=stddev)
            elif get_backend() == "pytorch":
                stddev = 1 / math.sqrt(shape[0] if isinstance(shape, (tuple, list)) and len(shape) > 0 else 1.0)
                self.initializer = lambda t: torch.nn.init.normal_(tensor=t, std=stddev)

        # No spec -> Leave initializer as None for TF (will then use default;
        #  e.g. for tf weights: Xavier uniform). For PyTorch, still have to set Xavier.
        # TODO this is None or is False is very unclean because TF and PT have different defaults ->
        # change to clean default values for weights and biases.
        elif specification is None or specification is False:
            if get_backend() == "tf":
                pass
            elif get_backend() == "pytorch":
                self.initializer = torch.nn.init.xavier_uniform_

        # Fixed values spec -> Use them, just do sanity checking.
        else:
            # Constant value across the variable.
            if isinstance(specification, (float, int)):
                pass
            # A 1D initializer (e.g. for biases).
            elif isinstance(specification, list):
                array = np.asarray(specification, dtype=dtype("float32", "np"))
                if array.shape != self.shape:
                    raise RLGraphError("ERROR: Number/shape of given items ({}) not identical with shape ({})!".
                                       format(array.shape, self.shape))
            # A nD initializer (numpy-array).
            elif isinstance(specification, np.ndarray):
                if specification.shape != self.shape:
                    raise RLGraphError("ERROR: Shape of given items ({}) not identical with shape ({})!".
                                       format(specification.shape, self.shape))
            # Unknown type.
            else:
                raise RLGraphError("ERROR: Bad specification given ({}) for Initializer object!".format(specification))

            # Create the backend initializer object.
            if get_backend() == "tf":
                self.initializer = tf.constant_initializer(value=specification, dtype=dtype("float32"))
            elif get_backend() == "pytorch":
                self.initializer = lambda t: torch.nn.init.constant_(tensor=t, val=specification)