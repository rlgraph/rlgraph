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

import numpy as np

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.environments.environment import Environment
from rlgraph.utils.ops import DataOpTuple
from rlgraph.utils.specifiable_server import SpecifiableServer
from rlgraph.utils.util import dtype as dtype_

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

    def __init__(self, environment_spec, actor_component_spec, state_space=None, reward_space=None, **kwargs):
        """
        Args:
            environment_spec (dict): A specification dict for constructing an Environment object that will be run
                inside a SpecifiableServer for in-graph stepping.
            actor_component_spec (Union[ActorComponent,dict]): A specification dict to construct this EnvStepper's
                ActionComponent (to generate actions) or an already constructed ActionComponent object.
            state_space (Optional[Space]): The state Space of the Environment. If None, will construct a dummy
                environment to get the state Space from there.
            reward_space (Optional[Space]): The reward Space of the Environment. If None, will construct a dummy
                environment to get the reward Space from there.
        """
        super(EnvironmentStepper, self).__init__(scope=kwargs.pop("scope", "env-stepper"), **kwargs)

        # Create the SpecifiableServer with the given env spec.
        if state_space is None or reward_space is None:
            dummy_env = Environment.from_spec(environment_spec)  # type: Environment
            state_space = dummy_env.state_space
            if reward_space is None:
                _, reward, _, _ = dummy_env.step(dummy_env.action_space.sample())
                reward_space = "float64" if type(reward) == float else float

        self.state_space = state_space
        self.reward_space = reward_space
        self.environment_spec = environment_spec
        self.environment_server = SpecifiableServer(
            Environment, environment_spec, dict(step=[self.state_space, self.reward_space, bool, None],
                                                reset=self.state_space), "terminate"
        )
        self.action_space = None

        # Add the sub-components.
        self.actor_component = ActorComponent.from_spec(actor_component_spec)  # type: ActorComponent
        self.preprocessed_state_space = self.actor_component.preprocessor.get_preprocessed_space(self.state_space)
        self.add_components(self.actor_component)

        # Variables that hold information of last step through Env.
        self.episode_return = None
        self.current_terminal = None
        self.current_state = None

        # Define our API methods.
        self.define_api_method("reset", self._graph_fn_reset)
        self.define_api_method("step", self._graph_fn_step)

    def create_variables(self, input_spaces, action_space=None):
        self.episode_return = self.get_variable(name="episode-return", initializer=0.0, dtype=self.reward_space)
        self.current_terminal = self.get_variable(name="current-terminal", dtype="bool", initializer=False)
        self.current_state = self.get_variable(name="current-state", from_space=self.state_space)

        assert action_space is not None
        self.action_space = action_space

    def _graph_fn_reset(self):
        """
        Resets the EnvStepper and stores the state after resetting in `self.current_state`.
        This is only necessary at the very beginning as the step method itself will take care of resetting the Env
        in between or during stepping runs (depending on terminal signals from the Env).

        Returns:
            SingleDataOp: The assign op that stores the state after the Env reset in `last_state` variable.
        """
        if get_backend() == "tf":
            state_after_reset = self.environment_server.reset()
            assigns = [
                self.assign_variable(self.current_state, state_after_reset),
                tf.variables_initializer([self.episode_return, self.current_terminal])
            ]
            with tf.control_dependencies(assigns):
                return tf.no_op()

    def _graph_fn_step(self, num_steps=1, time_step=0):
        """
        Performs n steps through the environment starting with the current state of the environment and returning
        accumulated tensors for the n steps.

        Args:
            num_steps (int): The number of steps to perform in the environment.
            time_step (int): The time_step at which we start stepping.

        Returns:
            Tuple[SingleDataOp,List[SingleDataOp]]:
                1) The step-op to be pulled to execute the stepping.
                2) The step results folded into a list of:
                - preprocessed_previous_states: Starting with the initial state of the environment and ending with
                    the one state before the last element in `next_states` (see below).
                - actions: The actions actually picked by our policy.
                - action_log_probs: The log-probabilities of all actions per step.
                - returns: The accumulated reward values for the ongoing episode up to after taking an action.
                TODO: discounting?
                - rewards: The rewards actually observed during stepping.
                - terminals: The terminal signals from the env. Values refere to whether the states in `next_states`
                    (see below) are terminal or not.
                - next_states: The (non-preprocessed) next states.
        """

        """
        Timeline:
        1) Session starts -> server is started: Env is created on server.
        2) reset is called: state (or preprocessed state??) is stored in var.
        3) Call step(3) -> takes values for initializer state, return-this-episode, terminal from the variable.
        4) get action + step + get_action + step + get_action + step
        5) return 
        """

        if get_backend() == "tf":
            def scan_func(accum, time_delta):
                _, _, _, episode_return, t, s = accum  # preprocessed-previous-state, prev-action, prev-r not needed

                # Add control dependency to make sure we don't step parallelly through the Env.
                t = tf.convert_to_tensor(t)
                with tf.control_dependencies([t]):
                    # If state (s) was terminal, reset the env (in this case, we will never need s (or a preprocessed
                    # version thereof for any NN runs (q-values, probs, values, etc..) as no actions are taken from s).
                    s = tf.cond(
                        t,
                        true_fn=lambda: self.environment_server.reset(),
                        false_fn=lambda: tf.convert_to_tensor(s)
                    )
                    # Add a simple (size 1) batch rank to the state so it'll pass through the NN.
                    s = tf.expand_dims(s, axis=0)
                    # Make None so it'll be recognized as batch-rank by the auto-Space detector.
                    s = tf.placeholder_with_default(s, shape=(None,) + self.state_space.shape)
                    # Get action and preprocessed state (as batch-size 1).
                    preprocessed_s, a = self.call(self.actor_component.get_preprocessed_state_and_action, s,
                                                  time_step + time_delta, return_ops=True)

                    # Step through the Env and collect next state, reward and terminal as single values (not batched).
                    s_, r, t_ = self.environment_server.step(a)

                    # Add up return (if s was not terminal).
                    new_episode_return = tf.where(t, x=r, y=(r + episode_return))

                # Accumulate return values (remove batch again from preprocessed_s and a).
                return preprocessed_s[0], a[0], r, new_episode_return, t_, s_

            # Initialize the tf.scan run and make sure nothing is float64.
            initializer = (self.preprocessed_state_space.zeros(),
                           self.action_space.zeros(),  # zero previous action (doesn't matter)
                           np.asarray(0.0, dtype=dtype_(self.reward_space, "np")),  # zero previous reward (doesn't matter)
                           self.episode_return,  # return so far
                           self.current_terminal,  # whether the current state is terminal
                           self.current_state  # current (raw) state
                           )
            step_results = DataOpTuple(tf.scan(fn=scan_func, elems=tf.range(num_steps), initializer=initializer))

            # Store the return so far, current terminal and current state.
            assigns = [
                self.assign_variable(self.episode_return, step_results[3][-1]),
                self.assign_variable(self.current_terminal, step_results[4][-1]),
                self.assign_variable(self.current_state, step_results[5][-1])
            ]
            with tf.control_dependencies(assigns):
                step_op = tf.no_op()

            return step_op, step_results
