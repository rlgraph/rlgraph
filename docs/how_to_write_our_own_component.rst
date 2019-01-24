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

How to Write Your Own Custom Component
======================================

In the following, we will build an entire component from scratch in RLgraph, including the component's API-methods,
its graph functions, and its variable generating code.

A Simple Single-Value Memory Component
--------------------------------------

Our component, once done, will look as follows:

.. figure:: images/custom-single-value-memory.png
   :scale: 60%
   :alt: The custom memory component we are about to build from scratch.

   Above: The custom memory component we are about to build from scratch.

We are building a simplistic memory that holds some value (or a tensor of values) in a variable stored under
`self.value`. Clients of our component will be able to read the current value via the API-method `get_value`, overwrite
it via `set_value`, and do some simple calculation by calling `get_value_plus_n` (which is not shown in the figure
above), which adds some number (`n`) to the current value of the variable and returns the result of that computation.


Class Definition and Constructor
++++++++++++++++++++++++++++++++

First we will create a new python file called `my_component.py` and will import all necessary RLgraph modules
as well as `tensorflow`, which will be the only supported backend of this component for simplicity reasons.

.. TODO: Add chapter on pytorch semantics.

.. code-block:: python

    import tensorflow as tf
    from rlgraph.components.component import Component
    from rlgraph.utils.decorators import rlgraph_api, graph_fn

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



API-Methods and Input Spaces
++++++++++++++++++++++++++++

Let's now define all our API-methods. Each of these will simply make a call to an underlying graph function
in which the actual magic is implemented. Note that all API-methods must be tagged with the `@rlgraph_api` decorator:

.. code-block:: python

    @rlgraph_api
    def get_value(self):
        return self._graph_fn_get()

    @rlgraph_api
    def set_value(self, value):
        return self._graph_fn_set(value)

    @rlgraph_api
    def get_value_plus_n(self, n):
        return self._graph_fn_get_value_plus_n(n)

Note that the set of our API-method call arguments is now: `value` and `n`. The spaces of both `value` and `n` must
thus be known to the RLgraph build system, before the `create_variables()` method will be called automatically.
In case our component is the root component, the user will have to provide these spaces manually in the Agent (which
is the owner of the root). Remember that this manual space is always necessary for all of the root component's API-method call arguments).


The Single Value Variable
+++++++++++++++++++++++++

Now it's time to specify, which variables our component needs. All variables should be generated inside a component's
`create_variables` method, which is called automatically, once all input spaces are known. In our case, the input
space for the `value` arg is important as that signals us, which type of variable we want (rank, dtype, etc.).
We can apply some restrictions to this space by implementing the `check_input_spaces()` method, which gets
called (automatically) right before `create_variables`. For example:

.. code-block:: python

    # Add this to the import section at the top of the file
    from rlgraph.spaces import ContainerSpace

    # Then, in our component class ...

    def check_input_spaces(self, input_spaces, action_space=None):
        # Make sure, we have a non-container space.
        in_space = input_spaces["value"]
        assert not isinstance(in_space, ContainerSpace), "ERROR: No containers allowed!"


The above code will make sure that only simple spaces are allowed as our variable space (e.g. a FloatBox with
some arbitrary shape).

Now that we have sanity checked our variable space, let's write the code to create the variable:

.. code-block:: python

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["value"]
        # Create the variable as non-trainable and with
        # the given initial value (from the c'tor).
        self.value = in_space.get_variable(
            trainable=False, initializer=self.initial_value
        )


Under the Hood Coding: Our Graph Functions
++++++++++++++++++++++++++++++++++++++++++

Finally, we need to implement the actual under-the-hood computation magic using our favourite machine learning backend.
We currently support `tensorflow <https://www.tensorflow.org/>`_ and `pytorch <https://pytorch.org/>`_,
but if you are interested in other backends, we would love to receive your contributions on this and PRs (see
`here for our contrib guidelines <https://github.com/rlgraph/rlgraph/blob/master/contrib/README.md>`_).

We will implement three graph functions, exactly those three that we have already been calling from within our
API-methods. Graph functions usually start with `_graph_fn_` and we should stick to that convention here as well.
The exact code for all three is shown below. Note the sudden change from abstract glue-code like coding to actual
tensorflow code. Graph functions can return one or more (a tuple) tensorflow ops. But we will also later learn
(`when we write an entire algorithm from scratch <how_to_build_an_algorithm_with_rlgraph.html>`_) how to bundle
and nest these ops into more complex tuple and dict structures and return these to the API-method caller.

.. code-block:: python

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


It might seem a little strange that our API-methods in this example are only very thin wrappers around the
actual computations (graph functions). However, in a later chapter on
`agent implementations <how_to_build_an_algorithm_with_rlgraph.html>`_, we will see how useful API-methods really are,
not for wrapping calls to graph functions, but to delegate and connect different graph functions and also other
API-methods with each other.
