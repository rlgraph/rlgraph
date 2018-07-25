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

import copy
import tensorflow as tf

from rlgraph.components.layers import Layer
from rlgraph.utils.util import default_dict


class PreprocessLayer(Layer):
    """
    A Layer that - additionally to `apply` - implements the `reset` API-method.
    `apply` is usually used for preprocessing inputs. `reset` is used to reset some state information of this
    preprocessor (e.g reset/reinitialize a variable).

    API:
        apply(input_): Preprocesses a single input_ value and returns the preprocessed data.
        reset(): Optional; Does some reset operations e.g. in case this PreprocessLayer contains variables and state.
    """
    def __init__(self, scope="pre-process", **kwargs):
        default_dict(kwargs, dict(flatten_ops=True, split_ops=True))
        super(PreprocessLayer, self).__init__(scope=scope, **kwargs)

        self.define_api_method("reset", self._graph_fn_reset)

    def get_preprocessed_space(self, space):
        """
        Returns the Space obtained after pushing the space input through this layer.

        Args:
            space (Space): The incoming Space object.

        Returns:
            Space: The Space after preprocessing.
        """
        return space

    def _graph_fn_reset(self):
        """
        Returns:
            An op that resets this processor to some initial state.
            E.g. could be called whenever an episode ends.
            This could be useful if the preprocessor stores certain episode-sequence information
            to do the processing and this information has to be reset after the episode terminates.
        """
        return tf.no_op(name="reset-op")  # Not mandatory.

