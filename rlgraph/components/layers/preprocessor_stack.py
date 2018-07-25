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

from rlgraph import get_backend
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.utils.util import default_dict

from rlgraph.components.layers.stack import Stack

if get_backend() == "tf":
    import tensorflow as tf


class PreprocessorStack(Stack):
    """
    A special Stack that only carries PreprocessLayer Components and bundles all their `reset` out-Sockets
    into one exposed `reset` out-Socket. Otherwise, behaves like a Stack in connecting the PreprocessorLayers
    from out-Socket(s) to in-Socket(s) all the way through.

    API:
        preprocess(input_): Outputs the preprocessed input_ after sending it through all sub-Components of this Stack.
        reset(): An op to trigger all PreprocessorLayers of this Stack to be reset.
    """
    def __init__(self, *preprocessors, **kwargs):
        """
        Args:
            preprocessors (PreprocessorLayer): The PreprocessorLayers to add to the Stack and connect to each other.

        Raises:
            RLGraphError: If a sub-component is not a PreprocessLayer object.
        """
        # Link sub-Components' `apply` methods together to yield PreprocessorStack's `preprocess` method.
        # NOTE: Do not include `reset` here as it is defined explicitly below.
        kwargs["api_methods"] = {("preprocess", "apply")}
        default_dict(kwargs, dict(scope=kwargs.pop("scope", "preprocessor-stack")))
        super(PreprocessorStack, self).__init__(*preprocessors, **kwargs)

        # Now that the sub-components are constructed, make sure they are all PreprocessorLayer objects.
        for key, preprocessor in self.sub_components.items():
            assert isinstance(preprocessor, PreprocessLayer), \
                "ERROR: sub-Component '{}' in PreprocessorStack '{}' is not a PreprocessorLayer!".\
                format(preprocessor.name, self.name)

    def reset(self):
        # Connect each pre-processor's "reset" output op via our graph_fn into one op.
        resets = list()
        for preprocessor in self.sub_components.values():  # type: PreprocessLayer
            resets.append(self.call(preprocessor.reset))
        reset_op = self.call(self._graph_fn_reset, *resets)
        return reset_op

    def _graph_fn_reset(self, *preprocessor_resets):
        if get_backend() == "tf":
            with tf.control_dependencies(preprocessor_resets):
                return tf.no_op()

    def get_preprocessed_space(self, space):
        """
        Returns the Space obtained after pushing the input through all layers of this Stack.

        Args:
            space (Space): The incoming Space object.

        Returns:
            Space: The Space after preprocessing.
        """
        for pp in self.sub_components.values():
            space = pp.get_preprocessed_space(space)
        return space
