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
from rlgraph.components.neural_networks.preprocessor_stack import PreprocessorStack
from rlgraph.spaces import ContainerSpace, Dict
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import flatten_op, unflatten_op
from rlgraph.utils.util import default_dict

if get_backend() == "tf":
    import tensorflow as tf


class DictPreprocessorStack(PreprocessorStack):
    """
    A generic PreprocessorStack that can handle Dict/Tuple Spaces and parallely preprocess different Spaces within
    different (and separate) single PreprocessorStack components.
    The output is again a dict of preprocessed inputs.

    API:
        preprocess(input\_): Outputs the preprocessed input after sending it through all sub-Components of this Stack.
        reset(): An op to trigger all PreprocessorStacks of this Vector to be reset.
    """
    def __init__(self, preprocessors, **kwargs):
        """
        Args:
            preprocessors (dict):

        Raises:
            RLGraphError: If a sub-component is not a PreprocessLayer object.
        """
        # Create one separate PreprocessorStack per given key.
        # All possibly other keys in an input will be pass through un-preprocessed.
        self.preprocessors = flatten_op(preprocessors)
        for i, (flat_key, spec) in enumerate(self.preprocessors.items()):
            self.preprocessors[flat_key] = PreprocessorStack.from_spec(
                spec, scope="preprocessor-stack-{}".format(i)
            )

        # NOTE: No automatic API-methods. Define them all ourselves.
        kwargs["api_methods"] = {}
        default_dict(kwargs, dict(scope=kwargs.pop("scope", "dict-preprocessor-stack")))
        super(DictPreprocessorStack, self).__init__(*list(self.preprocessors.values()), **kwargs)

    @rlgraph_api(flatten_ops=True, split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_preprocess(self, key, inputs):
        # Is a PreprocessorStack defined for this key?
        if key in self.preprocessors:
            return self.preprocessors[key].preprocess(inputs)
        # Simple pass through, no preprocessing.
        else:
            return inputs

    @rlgraph_api
    def reset(self):
        # TODO: python-Components: For now, we call each preprocessor's graph_fn directly.
        if self.backend == "python" or get_backend() == "python":
            for preprocessor in self.preprocessors.values():  # type: PreprocessLayer
                preprocessor.reset()

        elif get_backend() == "tf":
            # Connect each pre-processor's "reset" output op via our graph_fn into one op.
            resets = list()
            for preprocessor in self.preprocessors.values():  # type: PreprocessorStack
                resets.append(preprocessor.reset())
            reset_op = self._graph_fn_reset(*resets)
            return reset_op

    def _graph_fn_reset(self, *preprocessor_resets):
        if get_backend() == "tf":
            with tf.control_dependencies(preprocessor_resets):
                return tf.no_op()

    def get_preprocessed_space(self, space):
        """
        Returns the Space obtained after pushing the input through all layers of this Stack.

        Args:
            space (Dict): The incoming Space object.

        Returns:
            Space: The Space after preprocessing.
        """
        assert isinstance(space, ContainerSpace)
        dict_spec = dict()
        for flat_key, sub_space in space.flatten().items():
            if flat_key in self.preprocessors:
                dict_spec[flat_key] = self.preprocessors[flat_key].get_preprocessed_space(sub_space)
            else:
                dict_spec[flat_key] = sub_space
        dict_spec = unflatten_op(dict_spec)
        return Dict(dict_spec)
