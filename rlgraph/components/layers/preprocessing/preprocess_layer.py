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
from __future__ import print_function
from __future__ import division

from rlgraph import get_backend
from rlgraph.components.layers import Layer
from rlgraph.utils.util import default_dict
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf


class PreprocessLayer(Layer):
    """
    A Layer that - additionally to `apply` - implements the `reset` API-method.
    `apply` is usually used for preprocessing inputs. `reset` is used to reset some state information of this
    preprocessor (e.g reset/reinitialize a variable).
    """
    def __init__(self, scope="pre-process", **kwargs):
        super(PreprocessLayer, self).__init__(scope=scope, **kwargs)

    @rlgraph_api
    def _graph_fn_reset(self):
        """
        Does some reset operations e.g. in case this PreprocessLayer contains variables and state.

        Returns:
            SingleDataOp: The op that resets this processor to some initial state.
        """
        if get_backend() == "tf":
            return tf.no_op(name="reset-op")  # Not mandatory.

        # TODO: fix for python backend.
        return

    @rlgraph_api(flatten_ops=True, split_ops=True)
    def _graph_fn_apply(self, *preprocessing_inputs):
        return super(PreprocessLayer, self)._graph_fn_apply(*preprocessing_inputs)
