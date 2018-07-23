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

from yarl.agents.impala_agent import IMPALAAgent

# Create a simple performance test comparing different approaches to Env-stepping:
# 1) classic Env stepping with Agent's get_action as  a single session call per action.
# 2) wrap the env.step calls with tf.scan over a tf.range (n-step) and using a func that calls env.step().
# 3) Try to place the entire env in-graph via tf.py_func.

