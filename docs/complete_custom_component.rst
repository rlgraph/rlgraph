.. Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   ============================================================================

.. image:: images/rlcore-logo-full.png
   :scale: 25%
   :alt:


The Complete Code for Our Custom Component
==========================================

Here you can see the complete code for our custom component. On the next page, we will talk about how we can test
this component via RLgraph's special `ComponentTest` class.

.. code-block:: python
   :linenos:

    import tensorflow as tf
    from rlgraph.components.component import Component
    from rlgraph.utils.decorators import rlgraph_api, graph_fn
    # To be able to do input-space sanity checking.
    from rlgraph.spaces import ContainerSpace

    # Define our new Component class.
    class MyComponent(Component):
        # Ctor.
        def __init__(self, initial_value=1.0, scope="my-component", **kwargs):
            # It is good practice to pass through **kwargs to parent class.
            super(MyComponent, self).__init__(scope, **kwargs)
            # Store the initial value.
            # This will be assigned equally to all items in the memory.
            self.initial_value = initial_value
            # Placeholder for our variable (will be created inside self.create_variables).
            self.value = None

        @rlgraph_api
        def get_value(self):
            return self._graph_fn_get()

        @rlgraph_api
        def set_value(self, value):
            return self._graph_fn_set(value)

        @rlgraph_api
        def get_value_plus_n(self, n):
            return self._graph_fn_get_value_plus_n(n)

        def check_input_spaces(self, input_spaces, action_space=None):
            # Make sure, we have a non-container space.
            in_space = input_spaces["value"]
            assert not isinstance(in_space, ContainerSpace), "ERROR: No containers allowed!"

        def create_variables(self, input_spaces, action_space=None):
            in_space = input_spaces["value"]
            # Create the variable as non-trainable and with
            # the given initial value (from the c'tor).
            self.value = in_space.get_variable(
                trainable=False, initializer=self.initial_value
            )

        @graph_fn
        def _graph_fn_get(self):
            # Note: read_value() is the tf way to make sure a read op is added to the graph.
            # (remember that self.value is an actual tf.Variable).
            return self.value.read_value()

        @graph_fn
        def _graph_fn_set(self, value):
            # We use the RLgraph `Component.assign_variable()` helper here.
            assign_op = self.assign_variable(self.value, value)
            # Make sure the value gets assigned via the no_op trick
            # (no_op is now dependent on the assignment op).
            with tf.control_dependencies([assign_op]):
               return tf.no_op()

        @graph_fn
        def _graph_fn_get_value_plus_n(self, n):
            # Simple tf.add operation as return value.
            return tf.add(self.value, n)


Now, lets take a look at how to `test the above component <rlgraphs_testing_system.html>`_ using simple python
unittest semantics and RLgraph's ComponentTest class.
