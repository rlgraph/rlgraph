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
import numpy as np

from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.spaces import FloatBox, IntBox, BoolBox
from rlgraph.tests import ComponentTest


class TestEnvironmentStepper(unittest.TestCase):
    """
    Tests for the EnvironmentStepper Component using a simple RandomEnv.
    """
    def test_environment_stepper_on_random_env(self):
        state_space = FloatBox()
        action_space = IntBox(2)
        preprocessor_spec = None
        neural_network_spec = "../configs/test_simple_nn.json"
        exploration_spec = None
        actor_component = ActorComponent(preprocessor_spec, neural_network_spec, exploration_spec)
        environment_stepper = EnvironmentStepper(
            environment_spec=dict(type="random_env", state_space=state_space, action_space=action_space,
                                  deterministic=True),
            actor_component_spec=actor_component,
            state_space=state_space
        )

        test = ComponentTest(component=environment_stepper, input_spaces=dict(num_steps=int))

        # Step 3 times through the Env and collect results.
        expected = None
        print(test.test(("step", 3), expected_outputs=expected))

    def test_environment_stepper_on_pong(self):
        pass


def main():
    from rlgraph.environments.environment import Environment
    from rlgraph.utils.specifiable_server import SpecifiableServer, SpecifiableServerHook
    import tensorflow as tf
    import numpy as np

    num_steps = 3000

    state_space = FloatBox()
    action_space = IntBox(2)
    env_spec = {"type": "random_env", "deterministic": True, "state_space": state_space, "action_space": action_space}
    dummy_env = Environment.from_spec(env_spec)
    env_server = SpecifiableServer(
        Environment, env_spec, dict(step=[state_space, FloatBox(), BoolBox(), None], reset=[state_space])
    )

    def fake_tf_policy(state):
        return tf.random_uniform((), 0, 2, dtype=tf.int32)

    # build the step-graph
    def scan_func(accum, _):
        states, actions, rewards, terminal = accum

        # Check whether we were terminal -> if yes, reset and keep working with state after reset
        tensor_terminal = tf.convert_to_tensor(terminal)
        with tf.control_dependencies([tensor_terminal]):
            tensor_state = tf.cond(
                tensor_terminal,
                true_fn=lambda: env_server.reset(),
                false_fn=lambda: tf.convert_to_tensor(states, dtype=tf.float32)
            )

            # Get a tf action.
            a = fake_tf_policy(tensor_state)

            print("HERE")

            # Push the action to the env server.
            s_, r, t = env_server.step(a)

        # Accumulate.
        return s_, a, r, t

    # Before the first step.
    initializer = (np.array(0.0, dtype=np.float32), np.array(0, dtype=np.int32), np.array(0.0, dtype=np.float32),
                   np.array(True))

    op = tf.scan(fn=scan_func, elems=tf.range(num_steps), initializer=initializer)

    with tf.train.SingularMonitoredSession(hooks=[SpecifiableServerHook()]) as sess:
        result = sess.run(op)

    print(result)

    # Compare with single step (1 session call per action) method.


# toy program for testing purposes
if __name__ == "__main__":
    main()

