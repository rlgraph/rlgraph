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

Data-Ops and Op-Records
=======================

Now that we have built an entire Component in RLgraph, let's look at what we did in a little more detail.
In this chapter, we will especially cover the question of why we separate API-methods from graph functions
and what in particular gets returned from both of these function types.


What is a DataOp?
-----------------

DataOps are the raw data that we work with and pass around inside RLgraph's graph functions.
Simple DataOps (not DataOpTuples or DataOpDicts) can be tensorflow tensor objects, such as:
`tf.placeholder(shape=(), dtype=tf.float32)` or `tf.`.




