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

import re

from rlgraph import get_backend
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.components.neural_networks.stack import Stack
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.util import default_dict

if get_backend() == "tf":
    import tensorflow as tf


class PreprocessorStack(Stack):
    """
    A special Stack that only carries PreprocessLayer Components and bundles all their `reset` output ops
    into one exposed `reset` output op. Otherwise, behaves like a Stack in feeding the outputs
    of one sub-Component to the inputs of the next sub-Component, etc..

    API:
        preprocess(input\_): Outputs the preprocessed input after sending it through all sub-Components of this Stack.
        reset(): An op to trigger all PreprocessorLayers of this Stack to be reset.
    """
    def __init__(self, *preprocessors, **kwargs):
        """
        Args:
            preprocessors (PreprocessorLayer): The PreprocessorLayers to add to the Stack and connect to each other.

        Keyword Args:
            fold_time_rank (bool): Whether to fold the time rank for the `preprocess` API-method stack.
            unfold_time_rank (bool): Whether to unfold the time rank for the `preprocess` API-method stack.

        Raises:
            RLGraphError: If a sub-component is not a PreprocessLayer object.
        """
        self.fold_time_rank = kwargs.get("fold_time_rank", False)
        self.unfold_time_rank = kwargs.get("unfold_time_rank", False)
        # Link sub-Components' `call` methods together to yield PreprocessorStack's `preprocess` method.
        # NOTE: Do not include `reset` here as it is defined explicitly below.
        kwargs["api_methods"] = [dict(api="preprocess", component_api="call", fold_time_rank=self.fold_time_rank,
                                      unfold_time_rank=self.unfold_time_rank)]
        default_dict(kwargs, dict(scope=kwargs.pop("scope", "preprocessor-stack")))
        super(PreprocessorStack, self).__init__(*preprocessors, **kwargs)

    @rlgraph_api
    def reset(self):
        # TODO: python-Components: For now, we call each preprocessor's graph_fn directly.
        if self.backend == "python" or get_backend() == "python":
            for preprocess_layer in self.sub_components.values():  # type: PreprocessLayer
                if re.search(r'^\.helper-', preprocess_layer.scope):
                    continue
                preprocess_layer._graph_fn_reset()

        elif get_backend() == "tf":
            # Connect each pre-processor's "reset" output op via our graph_fn into one op.
            resets = list()
            for preprocess_layer in self.sub_components.values():  # type: PreprocessLayer
                if re.search(r'^\.helper-', preprocess_layer.scope):
                    continue
                resets.append(preprocess_layer.reset())
            reset_op = self._graph_fn_reset(*resets)
            return reset_op

    @graph_fn
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
            # not in ["time-rank-folder_", "time-rank-unfolder_"] or \
            if not re.search(r'\.helper-', pp.scope) or \
                    (pp.scope == ".helper-time-rank-folder" and self.fold_time_rank) or \
                    (pp.scope == ".helper-time-rank-unfolder" and self.unfold_time_rank):
                space = pp.get_preprocessed_space(space)
        return space
