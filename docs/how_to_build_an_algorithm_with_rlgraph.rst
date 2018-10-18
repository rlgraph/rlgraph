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

How to Write Our Own Agent?
===========================

In the following, we will build a the 2015 DQN Algorithm from scratch and code its logic into a DQNAgent class
using only RLgraph's already existing components and Agent base class.

`The entire DQNAgent can be seen here <https://github.com/rlgraph/rlgraph/blob/master/rlgraph/agents/dqn_agent.py>`_
(as it's already part of the RLgraph library).


Writing the Agent's Class Stub and Ctor
---------------------------------------

