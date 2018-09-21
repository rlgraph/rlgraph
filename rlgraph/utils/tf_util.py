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

import contextlib

from rlgraph import get_backend

# TF specific scope/device utilities.
if get_backend() == "tf":
    import tensorflow as tf


    @contextlib.contextmanager
    def pin_global_variables(device):
        """Pins global variables to the specified device."""

        def getter(getter, *args, **kwargs):
            var_collections = kwargs.get('collections', None)
            if var_collections is None:
                var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
            if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
                with tf.device(device):
                    return getter(*args, **kwargs)
            else:
                return getter(*args, **kwargs)

        with tf.variable_scope('', custom_getter=getter) as vs:
            yield vs