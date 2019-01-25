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

How to Test Your Components
===========================

Now we will show you, how one can very easily test a single component via RLgraph's testing system.
As an example, we will use our custom component built from scratch in
`this chapter here <how_to_write_your_own_component.rst>`_.


Writing a New Test Case with Python's Unittest Module
-----------------------------------------------------

Create a new python file and call it `test_my_component.py`. Then add the following import statements to it:

.. code-block:: python

   import numpy as np
   import unittest

   from rlgraph.spaces import *
   from rlgraph.tests import ComponentTest

   # Import our custom component.
   from my_component import MyComponent


And start a unittest.TestCase class stub as follows:

.. code-block:: python

   class TestMyComponent(unittest.TestCase):

       # The user has to provide all input spaces because our component will be the root.
       # (RLgraph always needs only the root component's input spaces).
       input_spaces = dict(
           value=FloatBox(shape=(2,)),  # Our memory will store float vectors of dimension 2.
           n=float  # <- same as "FloatBox()", same as "FloatBox(shape=())"
       )

The above code creates a TestCase class and defines the input spaces to our root component (the one we would like to
test). RLgraph must always know up front the spaces going into the root component (not those of any of the inner
components) because it has to create the placeholders for any values of future API-method call arguments.

Test 1: Retrieving the Value
++++++++++++++++++++++++++++

.. code-block:: python

    def test_read_value(self):
        # Create an instance of the Component to test.
        my_component = MyComponent(initial_value=2.0)

        # Create a ComponentTest object.
        test = ComponentTest(
            component=my_component,
            input_spaces=self.input_spaces
        )

        # Make the API-method call.
        # Get the current value of the memory (should be all 2.0).
        # Output values are always numpy arrays.
        test.test(("get_value"), expected_outputs=np.array([2.0, 2.0]))

From here on, you can run the above test case via a simple: `python -m unittest test_my_component.py`.
Note that the API-call via `test.test()` is defined by the tuple `("get_value")`. This is due to the fact, that
you can:

a) Execute more than one single API-method call inside a single call to `ComponentTest.test()`.

b) Sometimes, you have to specify the call arguments (`get_value` doesn't have any). See the next examples on
   how to do so.


Test 2: Writing a New Value (and then checking it)
++++++++++++++++++++++++++++++++++++++++++++++++++

Next, we will test overwriting our memory's 2 floating point numbers, then check again their current values.

.. code-block:: python

    def test_write_values(self):
        # Create an instance of the Component to test.
        my_component = MyComponent()

        # Create a ComponentTest object.
        test = ComponentTest(
            component=my_component,
            input_spaces=self.input_spaces
        )

        # Make the API-method call to overwrite the memory's values with the vector [0.5, 0.6].
        # Note that we do not expect any output from that API-call.
        test.test(("set_value", np.array([0.5, 0.6])), expected_outputs=None)

        # Now test, whether the new value has actually been written.
        test.test(("get_value"), expected_outputs=np.array([0.5, 0.6]))



Test 3: Testing for the Correct Computation Results
+++++++++++++++++++++++++++++++++++++++++++++++++++

Finally, we test for the correct execution of our "complicated" computation method, the one where we add a constant
value (via tf.add) to all numbers in the memory.

.. code-block:: python

    def test_computation_plus_n(self):
        # Create an instance of the Component to test.
        my_component = MyComponent(initial_value=10.0)

        # Create a ComponentTest object.
        test = ComponentTest(
            component=my_component,
            input_spaces=self.input_spaces
        )

        # Make the API-method call to add 5.0 to all values (they should all be 10.0 right now)
        # and expect the result as the return value.
        test.test(("get_value_plus_n", 5.0), expected_outputs=np.array([15.0, 15.0]))

That's it. Our custom component is now fully tested an operational.

Now that we know (almost) everything about single components, let's take a deeper look at
the single building blocks of a Component. In particular, we would like to understand how
RLgraph handles data and the computation graph under the hood. For that,
`we need to learn, what DataOps and DataOpRecords are <op_records_and_data_ops.html>`_ and what the
difference between those two concepts is. After that, we will be able to understand entire agent implementations
and `write our own RLgraph agent from scratch <how_to_build_an_algorithm_with_rlgraph.html>`_.
