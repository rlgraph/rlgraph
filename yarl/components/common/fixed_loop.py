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

from yarl import YARLError, get_backend
from yarl.components import Component

if get_backend() == "tf":
    import tensorflow as tf


class FixedLoop(Component):
    """
    A FixedLoop component is used to iteratively call other GraphFunctions, e.g. in an optimization.
    """
    def __init__(self, num_iterations, call_component, graph_fn_name, scope="fixed-loop", **kwargs):
        """
        Args:
            num_iterations (int): How often to call the given GraphFn.
            call_component (Component): Component providing graph fn to call within loop.
            graph_fn_name (str): The name of the graph_fn in call_component.
        """
        assert num_iterations > 0

        super(FixedLoop, self).__init__(scope=scope, **kwargs)

        self.num_iterations = num_iterations
        self.graph_fn_to_call = None
        for graph_fn in call_component.graph_fns:
            if graph_fn.name == graph_fn_name:
                self.graph_fn_to_call = graph_fn.get_method()
                break
        if not self.graph_fn_to_call:
            raise YARLError("ERROR: GraphFn '{}' not found in Component '{}'!".format(graph_fn_name,
                                                                                      call_component.global_scope))
        # TODO: Do we sum up, append to list, ...?
        self.define_inputs("api_methods")
        self.define_outputs("fixed_loop_result")
        self.add_component(call_component)
        self.add_graph_fn("api_methods", "fixed_loop_result", self._graph_fn_call_loop, flatten_ops={"api_methods"})

    def _graph_fn_call_loop(self, *inputs):
        """
        Calls the sub-component of this loop the specified number of times and returns the final result.

        Args:
            *inputs (FlattenedDataOp): Parameters for the call component.

        Returns:
            any: Result of the call.
        """

        if get_backend() == "tf":
            # Initial call.
            result = self.graph_fn_to_call(*inputs)

            def body(result_, i):
                with tf.control_dependencies(control_inputs=result_):
                    result_ = self.graph_fn_to_call(*inputs)
                return result_, i + 1

            def cond(result_, i):
                return i < self.num_iterations - 1

            result, _ = tf.while_loop(cond=cond, body=body, loop_vars=(result, 0))
            return result

