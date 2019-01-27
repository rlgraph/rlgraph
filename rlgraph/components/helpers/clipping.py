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
    import torch


class Clipping(Component):
    """
    Clipping utility (e.g. to clip rewards).

    API:
        clip(values) -> returns clipped values.
    """
    def __init__(self, clip_value=0.0, scope="clipping", **kwargs):
        super(Clipping, self).__init__(scope=scope, **kwargs)
        self.clip_value = clip_value

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_clip_if_needed(self, values):
        """
        Clips values if cli pvalue specified, otherwise passes through.
        Args:
            values (SingleDataOp): Values to clip.

        Returns:
            SingleDataOp: Clipped values.

        """
        if self.clip_value == 0.0:
            return values
        elif get_backend() == "tf":
            return tf.clip_by_value(t=values, clip_value_min=-self.clip_value, clip_value_max=self.clip_value)
        elif get_backend() == "pytorch":
            torch.clamp(values, min=-self.clip_value, max=-self.clip_value)

