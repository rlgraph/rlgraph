# Copyright 2018 The RLgraph authors, All Rights Reserved.
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

import numpy as np
import tensorflow as tf
import unittest

from rlgraph.environments.environment import Environment
from rlgraph.utils.specifiable_server import SpecifiableServer, SpecifiableServerHook
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
        # Create the server, but don't start it yet. This will be done fully automatically by the tf-Session.
        specifiable_server = SpecifiableServer(Environment, env_spec, dict(
            step_for_env_stepper=[state_space, float, bool]
        ), "terminate")

        # ret are ops now in the graph.
        ret1 = specifiable_server.step_for_env_stepper(action_space.sample())
        ret2 = specifiable_server.step_for_env_stepper(action_space.sample())

        # Check all 3 outputs of the Env step (next state, reward, terminal).
        self.assertEqual(ret1[0].shape, ())
        self.assertEqual(ret1[0].dtype, dtype("float32"))
        self.assertEqual(ret1[1].shape, ())
        self.assertEqual(ret1[1].dtype, dtype("float32"))
        self.assertEqual(ret1[2].shape, ())
        self.assertEqual(ret1[2].dtype, dtype("bool"))
        self.assertEqual(ret2[0].shape, ())
        self.assertEqual(ret2[0].dtype, dtype("float32"))
        self.assertEqual(ret2[1].shape, ())
        self.assertEqual(ret2[1].dtype, dtype("float32"))
        self.assertEqual(ret2[2].shape, ())
        self.assertEqual(ret2[2].dtype, dtype("bool"))

        # Start the session and run the op, then check its actual values.
        with tf.train.SingularMonitoredSession(hooks=[SpecifiableServerHook()]) as sess:
            out1 = sess.run(ret1)
            out2 = sess.run(ret2)

        # next state
        self.assertAlmostEqual(out1[0], 0.7713, places=4)
        self.assertAlmostEqual(out2[0], 0.7488, places=4)
        # reward
        self.assertAlmostEqual(out1[1], 0.0208, places=4)
        self.assertAlmostEqual(out2[1], 0.4985, places=4)
        # terminal
        self.assertTrue(out1[2] is np.bool_(False))
        self.assertTrue(out2[2] is np.bool_(False))

