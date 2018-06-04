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

from yarl.utils.util import default_dict
from yarl.components.layers import Stack
from yarl.components.common import Synchronizable


class NeuralNetwork(Stack, Synchronizable):
    """
    Simple placeholder class that's a Stack and a Synchronizable Component.
    """
    def  __init__(self, writable=True, scope="neural-network", *layers, **kwargs):
        """
        Args:
            writable (bool): Whether this NN can be synched to by another (equally structured) NN.
                Default: True.
            *layers (Component): Same as `sub_components` argument of Stack. Can be used to add Layer Components
                (or any other Components) to this Network.
        """
        # In case layers come in via a spec dict -> push it into *layers.
        layers_args = kwargs.pop("layers", layers)
        # Sort out our kwargs and split them for the calls to the two super constructors.
        default_dict(kwargs, {"writable": writable})
        super(NeuralNetwork, self).__init__(scope=scope, *layers_args, **kwargs)

