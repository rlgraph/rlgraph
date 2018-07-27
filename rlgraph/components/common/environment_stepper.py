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

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.components.layers.preprocessor_stack import PreprocessorStack

if get_backend() == "tf":
    import tensorflow as tf


class EnvironmentStepper(Component):
    """
    A Component that takes an Environment object, a PreprocessorStack and a Policy to step
    n times through the environment, each time picking actions depending on the states that the environment produces.

    API:
        step(num_steps): Performs n steps through the environment and returns some collected stats:
            preprocessed_states, actions taken, action log-probabilities, rewards, terminals, discounts
    """

    def __init__(self, environment, policy, preprocessor=None, **kwargs):
        super(EnvironmentStepper, self).__init__(scope=kwargs.pop("scope", "env-stepper"), **kwargs)

        self.environment = environment
        self.policy = policy
        self.preprocessor = preprocessor

        self.current_state = self.environment.reset()

        # Add the sub-components.
        self.add_components(self.preprocessor, self.policy)

        self.define_api_method("step", self._graph_fn_step)

    def create_variables(self, input_spaces, action_space):
        self.state_space = self.environment.state_space.with_batch_rank()

    def _graph_fn_step(self, num_steps):
        """

        Args:
            num_steps (int): The number of steps to perform in the environment.

        Returns:
            tuple:
                - preprocessed_states: Starting with the initial state of the environment and ending with the last state
                    reached.
                - actions_taken: The actions actually picked by our policy.
                - action_log_probs: The log-probabilities of all actions per step.
                TODO: add more necessary stats here.
        """

        def scan_func(accum, elems):
            # TODO: preprocess state if self.preprocessor not None
            # TODO: get_action
            # TODO: state = env.step(action)
            self.preprocessor._graph_fn_apply()

            return accum

        if get_backend() == "tf":
            initializer = [self.current_state, ]
            n_steps = tf.scan(fn=scan_func, elems=tf.range(num_steps), initializer=initializer)
            return n_steps


def main():
    from rlgraph.environments.openai_gym import OpenAIGymEnv
    import tensorflow as tf
    import numpy as np

    num_steps = 3000

    env = OpenAIGymEnv("Pong-v0")
    state = env.reset()

    def fake_policy(state):
        return np.random.randint(0, 5)

    # build the step-graph
    def scan_func(accum, _):
        states, actions, rewards, terminals = accum
        # fake policy
        a = fake_policy(states)
        print("HERE")

        s_, r, t, _ = env.step(a)
        return s_, a, r, t

    # Before the first step.
    initializer = (state, np.array(0, dtype=np.int32), np.array(0.0, dtype=np.float32), np.array(False))

    op = tf.scan(fn=scan_func, elems=tf.range(3000), initializer=initializer, parallel_iterations=1)

    with tf.Session() as sess:
        result = sess.run(op)

    print(result)

    # Compare with single step (1 session call per action) method.


# toy program for testing purposes
if __name__ == "__main__":
    main()
