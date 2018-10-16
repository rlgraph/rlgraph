.. Copyright 2018 The RLgraph authors. All Rights Reserved.
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
executor and testing classes.

Each component contains:

- Any number of sub-components, each of which may again contain their own sub-components. Sub-components are also \
  sometimes called "child components".

- At least one API-method, so that clients of the component (in the end this will be our reinforcement learning agent) \
  can use it.

- Any number of so called "Graph Functions", which are special component methods, which contain the actual \
  computation code. These are the only places, where you will find backend (tensorflow, pytorch, etc..) specific code.

- Any number of variables for the component to use for its computations (graph functions).

On the `following page <how_to_write_your_own_component.html>`_, we will walk through building our own custom
component, which will include all of the above items. But let's first talk in some more detail about RLgraph's
Component base class.


The Component Base Class
------------------------

The `Component` base class contains the core functionality, which every RLgraph Component inherits. The most important
methods are listed below. For a more detailed overview, please take a look at the
`Component reference documentation <reference/components/component_base.html>`_.

#. `add_components`: This method is used to add an arbitrary number sub-components to the Component.
#. `check_input_spaces`:
#. `create_variables`: This method is called automatically by the RLgraph build process and can be implemented
   in order to create an arbitrary number of variables to use by the component's computation functions
   ("graph functions").
#. `copy`: Copies the component and returns a new Component object that is identical to the original one. This is
   useful - for example - to create a target network as a copy of the main policy network in a DQN-type agent.


API-Methods
+++++++++++

A component's API-methods are its outside facing handles through which clients of the component (either a component
or an agent that contain the component in question) can access and control its behavior.
For example, a typical memory component would need an `insert_records` API-method to insert some records into the memory,
a `get_records` method to retrieve a certain number of already stored records or a `clear` method to wipe out
all stored data from the memory.

API-methods are regular class methods, but must be tagged with the `@rlgraph_api` decorator, which can be imported as
follows:

.. code-block:: python

    from rlgraph.utils.decorators import rlgraph_api

An API-method can have any arbitrary combination of args and kwargs, as well as define default values for some of these.
For example:

.. code-block:: python

    # inside some component class ...
    ...
    @rlgraph_api
    def my_api_method(self, a, b=5, c=None)
        # do and return something

Calling the above API-method (e.g. from its parent component) requires the call argument `a` to be provided, whereas
`b` and `c` are optional arguments.


Input Spaces and the concept of "input-completeness"
++++++++++++++++++++++++++++++++++++++++++++++++++++



Variables
+++++++++



Graph Functions
+++++++++++++++


