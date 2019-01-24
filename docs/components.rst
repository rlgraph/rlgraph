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


What is an RLgraph Component?
=============================

Components are the basic building blocks, which you will use to build any machine learning and reinforcement learning
models with. A component is the smallest unit, which can be run and tested in and by itself via RLgraph's different
executor and testing classes. RLgraph components span from simple (and single) neural network layers to highly complex
policy networks, memories, optimizers and mathematical components, such as loss functions.

Each component contains:

- ... any number of sub-components, each of which may again contain their own sub-components (also sometimes
  called "child components"). Hence components are arbitrarily nestable inside each other.

- ... at least one API-method, so that clients of the component (in the end this will be our reinforcement learning agent)
  can use it.

.. figure:: images/dense-layer-component.png
   :alt: A DenseLayer component (1) with two API-methods (2), one graph function (3) and two variables (kernel and bias) (4).
   :scale: 60%

   Above: A DenseLayer component (1) with two API-methods (2), one graph function (3) and two variables (kernel and
   bias) (4).

- ... any number of "graph functions", which are special component methods, which contain the actual
  computation code. These are the only places, where you will find backend (tensorflow, pytorch, etc..) specific code.

- ... any number of variables for the component to use for its computations (graph functions).

On the `following page <how_to_write_your_own_component.html>`_, we will walk through building our own custom
component, which will include all of the above items. But let's first talk in some more detail about RLgraph's
Component base class.


The Component Base Class
------------------------

The `Component` base class contains the core functionality, which every RLgraph Component inherits from.
Its most important methods are listed below. For a more detailed overview, please take a look at the
`Component reference documentation <reference/components/component_base.html>`_.

#. `add_components`: This method is used to add an arbitrary number of sub-components to the component itself.
#. `check_input_spaces`: Can be used to sanity check the incoming spaces (see the
   `documentation on RLgraph's Space classes <spaces.rst>`_) of all API-method call arguments.
#. `create_variables`: This method is called automatically by the RLgraph build system and can be implemented
   in order to create an arbitrary number of variables used by the component's computation functions
   ("graph functions").
#. `copy`: Copies the component and returns a new Component object that is identical to the original one. This is
   useful, for example, to create a target network as a copy of the main policy network in a DQN-type agent.


API-Methods
+++++++++++

A component's API-methods are its outside facing handles through which clients of the component (either another
component or an agent that contains the component in question) can access and control its behavior.
For example, a typical memory component would need an `insert_records` API-method to insert some data into the memory,
a `get_records` method to retrieve a certain number of already stored records, and maybe a `clear` method to wipe out
all stored information from the memory.

API-methods are normal class methods, but must be tagged with the `@rlgraph_api` decorator, which can be imported as
follows:

.. code-block:: python

    from rlgraph.utils.decorators import rlgraph_api

An API-method can have any arbitrary combination of regular python args and kwargs, as well as define default
values for some of these.
For example:

.. code-block:: python

    # inside some component class ...
    ...
    @rlgraph_api
    def my_api_method(self, a, b=5, c=None):
        # do and return something

Calling the above API-method (e.g. from its parent component) requires the call argument `a` to be provided, whereas
`b` and `c` are optional arguments. As you may recall from the `spaces chapter <spaces.rst>`_, information in RLgraph
is passed around between components within fixed space constraints. In fact, each API-method call argument (`a`, `b`,
and `c` in our example above) has a dedicated space after the final graph has been built from all components in it.

**Important note:** Up until now, if an API-method is called more than once by the component's client(s), the spaces of
the provided call arguments (e.g. the space of `a`) in the different API-calls have to match. So if in the first
call, `a` is an IntBox, in the second call, it has to be an IntBox as well.
This is because of a possible dependency of the component's variables (see below) on these "input-spaces". We
will try to further loosen this restriction in future releases
and only require it if RLgraph knows for sure that the space of the argument in question is being used to define
variables of the component.


Variables
+++++++++

Variables are the data that each component can store for the lifetime of the computation graph. A variable has a
fixed data type and shape, hence a fixed Rlgraph space. As a matter of fact, variables are often created directly
from `Space` instances via the practical `Space.get_variable()` method.

Variables can be accessed inside graph functions (see below) and can be read as well as be written to.
Examples for variables are:

- The buffer of a memory that stores a certain part of a memory record, for example an image (rank-3 uint8 tensor).

- A memory component's index pointer (which record should we retrieve next?). This is usually a single int scalar.

- The weights matrix of some neural network layer. This is always a rank-2 float tensor.

Variables are created in a component's `create_variables` method, which gets called automatically, once all input
spaces of the component (all its API-method arguments' spaces) are known to the RLgraph build system. In the
next paragraph, we will explain how this stage of "input-completeness" is reached and why it's important for
the component.

Input Spaces and the concept of "input-completeness"
++++++++++++++++++++++++++++++++++++++++++++++++++++

Let's look at a Component's API-method and its variable generating code to understand the concept of
"input-completeness".

.. code-block:: python

    # inside some component class ...
    ...
    @rlgraph_api
    def insert(self, record):
        # Call a graph function that will take care of the assignment.
        return self._graph_fn_insert(record)

    def create_variables(input_spaces, action_space=None):
        """
        Override this base class method to create variables based on the
        spaces that are underlying each API-method's call argument
        (in our case, this is only the call arg "records" of the "insert" API-method).
        """
        # Lookup the input space by the name of the API-method's call arg ("record").
        in_space = input_spaces["record"]
        self.storage_buffer = in_space.get_variable(trainable=False, ... other options)

A component reaches input-completeness, if all spaces to all its unique call parameters (by their names) are known.
A space for a call argument (e.g. `record`) gets known once the respective API-method (here: `insert`) gets called by a
client (a parent component). Only the outermost component, also called the "root", needs its spaces to be provided
manually by the user, since its API-methods are only executed (called) at graph-execution time.

If a component has many API-methods, each with the only call argument `a` , which share the call parameter's names (e.g. a component has API-methods:
`one(a, b)`)

A client of this component (a parent component or the RL agent directly) will eventually make a call to the
component's API-method `insert()`. At that point, the space of the `record` argument will be known. If the component
above only has that one API-method, and hence only that one API-method call argument (`record`), it is then
input-complete.


Graph Functions
+++++++++++++++

Every component serves a certain computation purpose within a machine learning model. A neural network layer maps
input data to output data via, for example, a matrix-matrix multiplication (and adding maybe some bias). An optimizer
calculates the gradient of a loss function over the weights of a trainable layer and applies the resulting gradients
in a certain way to these weights. All these calculation steps happen inside a component's graph functions, the
only place in RLgraph, were we can find backend specific code, such as calls to tensorflow's static graph building
functions or computations on pytorch tensors.

Unlike API-methods, graph functions can only be called from within the same component that owns them (not by parents
or grandparents of the component). These calls happen from within the component's different API-methods (similar to
calling another API-method).

Graph functions are - similar to API-methods - regular python class methods, but must be tagged with the `@graph_fn`
decorator as follows:

.. code-block:: python

    # inside some component class ...
    ...
    @graph_fn
    def _graph_fn_do_some_computation(self, a, b):
        # All backend-specific code in RLgraph goes into graph functions.
        if get_backend() == "tf":
            # Do some computation in tf.
            some_result = tf.add(a, b)

        elif get_backend() == "pytorch":
            # Do some computation in pytorch.
            some_result = a + b

        return some_result



Inside a graph function, any type of backend specific computations are allowed to be coded. A graph function then
returns the result of the computation or many results as a tuple.

