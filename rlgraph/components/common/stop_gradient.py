# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    pass


class StopGradient(Component):
    """
    A simple softmax component that produces probabilities from logits.
    """
    def __init__(self, scope="stop-gradient", **kwargs):
        super(StopGradient, self).__init__(scope=scope, **kwargs)

    @rlgraph_api(flatten_ops=True, split_ops=True)
    def _graph_fn_stop(self, input_):
        """
        Converts all inputs into a stop-gradient op.

        Args:
            input_ (DataOp): The inputs to add a stop-gradient to.

        Returns:
            `inputs`, but passed through a stop-gradient operator.
        """
        if get_backend() == "tf":
            return tf.stop_gradient(input_)

        elif get_backend() == "pytorch":
            # TODO: implement stop-gradient in pytorch?
            return input_
