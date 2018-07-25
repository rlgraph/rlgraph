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

from rlgraph.components.layers.stack import Stack


class NeuralNetwork(Stack):
    """
    Simple placeholder class that's a Stack which optionally owns a Synchronizable Component and is thus
    writable from another, equally built NeuralNetwork.
    """
    def __init__(self, *layers, **kwargs):
        """
        Args:
            *layers (Component): Same as `sub_components` argument of Stack. Can be used to add Layer Components
                (or any other Components) to this Network.

        Keyword Args:
            layers (Optional[list]): An optional list of Layer objects or spec-dicts to overwrite(!)
                *layers.
        """
        # In case layers come in via a spec dict -> push it into *layers.
        layers_args = kwargs.pop("layers", layers)
        # Add a default scope (if not given) and pass on via kwargs.
        kwargs["scope"] = kwargs.get("scope", "neural-network")
        super(NeuralNetwork, self).__init__(*layers_args, **kwargs)

