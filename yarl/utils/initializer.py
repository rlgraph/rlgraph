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

import numpy as np

from yarl.backend_system import backend
from .yarl_error import YARLError
from .specifiable import Specifiable
from .util import dtype


class Initializer(Specifiable):
    def __init__(self, shape, specification=None):
        """
        Args:
            shape (tuple): The shape of the Variables to initialize.
            specification (any): Some spec that determines the nature of this initializer.

        Raises:
            YARLError: If fixed shape in specification does not match `self.shape`.
        """
        # The shape of the variable to be initialized.
        self.shape = shape
        # The actual underlying initializer object.
        self.initializer = None

        if backend == "tf":
            import tensorflow as tf
        else:
            tf = None

        # No spec -> Leave initializer as None (will then use default; for tf: Xavier uniform).
        if specification is None or specification is False:
            pass
        # Fixed values spec -> Use them, just do sanity checking.
        else:
            # Constant value across the variable.
            if isinstance(specification, float):
                pass
            # A 1D initializer (e.g. for biases).
            elif isinstance(specification, list):
                array = np.asarray(specification, dtype=dtype("float32", "np"))
                if array.shape != self.shape:
                    raise YARLError("ERROR: Number/shape of given items ({}) not identical with shape ({})!".
                                    format(array.shape, self.shape))
            # A nD initializer (numpy-array).
            elif isinstance(specification, np.ndarray):
                if specification.shape != self.shape:
                    raise YARLError("ERROR: Shape of given items ({}) not identical with shape ({})!".
                                    format(specification.shape, self.shape))
            # Unknown type.
            else:
                raise YARLError("ERROR: Bad specification given ({}) for Initializer object!".format(specification))

            self.initializer = tf.constant_initializer(value=specification, dtype=dtype("float32"))
