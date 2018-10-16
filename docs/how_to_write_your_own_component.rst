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

How to Write Your Own Custom Component
======================================

In the following, we will build in RLgraph an entire component from scratch, including its API-methods, graph
functions, and variable generating code.
Our component, once done, will look as follows:

We are building a simplistic memory that holds a single scalar floating point value. Clients of the component will be
able to read that value via the API-method `get_value`, overwrite the value via `set_value`, and do some simple
calculation via `get_value_plus_n` (which adds some given number to the current value of the variable and returns
the result).




A Simple Single-Value Memory Component
--------------------------------------

Describe how one can write a multi-backend single-variable/value component
with a simple API.


API-Methods and Input Spaces
++++++++++++++++++++++++++++


The single value Variable
+++++++++++++++++++++++++



Under the Hood Coding: Our Graph Functions
++++++++++++++++++++++++++++++++++++++++++



