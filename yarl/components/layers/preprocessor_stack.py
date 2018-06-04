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

from yarl import backend
from yarl.components import Component
from yarl.components.layers.preprocessing import PreprocessLayer
from yarl.utils.util import default_dict

from .stack import Stack


class PreprocessorStack(Stack):
    """
    A special Stack that only carries PreprocessLayer Components and bundles all their `reset` out-Sockets
    into one exposed `reset` out-Socket. Otherwise, behaves like a Stack in connecting the PreprocessorLayers
    from out-Socket(s) to in-Socket(s) all the way through.
    """
    def __init__(self, *preprocessors, **kwargs):
        """
        Args:
            preprocessors (PreprocessorLayer): The PreprocessorLayers to add to the Stack and connect to each other.

        Raises:
            YARLError: If sub-components' number of inputs/outputs do not match.
        """
        default_dict(kwargs, dict(scope=kwargs.pop("scope", "preprocessor-stack"),
                                  sub_component_inputs="input", sub_component_outputs="output",
                                  flatten_ops=False))
        super(PreprocessorStack, self).__init__(*preprocessors, **kwargs)

        # Now that the sub-components are constructed, make sure they are all ProprocessorLayer objects.
        for key, preprocessor in self.sub_components.items():
            assert isinstance(preprocessor, PreprocessLayer), \
                "ERROR: sub-Component '{}' in PreprocessorStack '{}' is not a PreprocessorLayer!".\
                format(preprocessor.name, self.name)

        # Connect each pre-processor's "reset" out-Socket to our graph_fn.
        resets = list()
        for preprocessor in self.sub_components.values():  # type: PreprocessLayer
            resets.append(preprocessor.get_output("reset"))
        self.add_graph_fn(resets, "reset", self._graph_fn_reset)

    def _graph_fn_reset(self, *preprocessor_resets):
        if backend == "tf":
            import tensorflow as tf
            with tf.control_dependencies(preprocessor_resets):
                return tf.no_op()

