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

import unittest

from rlgraph.environments.environment import Environment
from rlgraph.utils.specifiable_server import SpecifiableServer
from rlgraph.utils.util import dtype
from rlgraph.spaces import IntBox, FloatBox


class TestSpecifiableServer(unittest.TestCase):
    """
    Tests a SpecifiableServer with a simple environment and make some calls to it to see how it reacts.
    """
    def test_specifiable_server(self):
        action_space = IntBox(2)
        state_space = FloatBox()
        env_spec = dict(type="random_env", state_space=state_space, action_space=action_space, deterministic=True)
        specifiable_server = SpecifiableServer(Environment, env_spec, dict(step=[state_space, float, bool, None]),
                                               "terminate")
        specifiable_server.start()

        ret = specifiable_server.step(action_space.sample())

        # Check all 3 outputs of the Env step (next state, reward, terminal).
        self.assertEqual(ret[0].shape, ())
        self.assertEqual(ret[0].dtype, dtype("float"))
        self.assertEqual(ret[1].shape, ())
        self.assertEqual(ret[1].dtype, dtype("float"))
        self.assertEqual(ret[2].shape, ())
        self.assertEqual(ret[2].dtype, dtype("bool"))
