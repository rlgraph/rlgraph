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
- Any number of sub-components, each of which may again contain further sub-components. Sub-components are also \
   sometimes called child components.
- At least one API-method, so that clients of the component can use it.
- Any number of so called "Graph Functions", which are special component methods, which contain the actual \
   computation code. These are the only places, where you will find backend (tensorflow, pytorch, etc..) specific code.
- Any number of variables for the component to use for its computations (graph functions).


The Component Base Class
------------------------





TODO



API-Methods
+++++++++++


Graph Functions
+++++++++++++++

Variables
+++++++++


