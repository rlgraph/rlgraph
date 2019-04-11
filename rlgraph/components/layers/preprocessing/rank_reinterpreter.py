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
from rlgraph.components.layers.preprocessing.preprocess_layer import PreprocessLayer
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf


class RankReinterpreter(PreprocessLayer):
    """
    Re-interprets the given ranks (ints) into batch and/or time ranks.
    """
    def __init__(self, batch_rank=None, time_rank=None, scope="rank-reinterpreter",  **kwargs):
        """
        Args:
            batch_rank (Optional[int]): The batch rank to set for the input.
            time_rank (Optional[int]): The time rank to set for the input.
        """
        super(RankReinterpreter, self).__init__(space_agnostic=True, scope=scope, **kwargs)
        self.batch_rank = batch_rank
        self.time_rank = time_rank

    @rlgraph_api
    def _graph_fn_call(self, inputs):
        if get_backend() == "tf":
            ret = tf.identity(inputs, name="rank-reinterpreted")
            # We have to re-interpret the batch rank.
            if self.batch_rank is not None:
                ret._batch_rank = self.batch_rank
            # We have to re-interpret the time rank.
            if self.time_rank is not None:
                ret._time_rank = self.time_rank

            return ret
