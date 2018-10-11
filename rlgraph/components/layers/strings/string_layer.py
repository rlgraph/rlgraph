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

from rlgraph.components.layers.layer import Layer
from rlgraph.spaces import TextBox
from rlgraph.spaces.space_utils import sanity_check_space


class StringLayer(Layer):
    """
    A generic string processing layer object.
    """
    def __init__(self, **kwargs):
        super(StringLayer, self).__init__(scope=kwargs.pop("scope", "str-layer"), **kwargs)

    def check_input_spaces(self, input_spaces, action_space=None):
        """
        Do some sanity checking on the incoming Space:
        Must be string type.
        """
        sanity_check_space(input_spaces["text_inputs"], allowed_types=[TextBox], must_have_batch_rank=True)
