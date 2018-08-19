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

from collections import OrderedDict
import numpy as np

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.environments.environment import Environment
from rlgraph.utils.ops import DataOpTuple, DataOpDict, flatten_op, unflatten_op
from rlgraph.utils.specifiable_server import SpecifiableServer
from rlgraph.utils.util import dtype as dtype_, force_tuple

if get_backend() == "tf":
    import tensorflow as tf


class EnvironmentStepper(Component):
    """
    A Component that takes an Environment object, a PreprocessorStack and a Policy to step
    n times through the environment, each time picking actions depending on the states that the environment produces.

    API:
        reset(): Resets the Environment stepper including its environment and gets everything ready for stepping.
            Resets the stored state, return and terminal of the env.
        step(internal_states, num_steps, time_step): Performs n steps through the environment and returns some
            collected stats: preprocessed_states, actions taken, action log-probabilities, rewards,
            terminals, discounts.
    """

    def __init__(self, environment_spec, actor_component_spec, state_space=None, reward_space=None,
                 add_previous_action=False, add_previous_reward=False, **kwargs):
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
            add_previous_action (bool): Whether to add the previous action as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_action". Default: False.
            add_previous_reward (bool): Whether to add the previous reward as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_reward". Default: False.
        """
        super(EnvironmentStepper, self).__init__(scope=kwargs.pop("scope", "env-stepper"), **kwargs)

        # Only to retrieve some information about the particular Env.
        dummy_env = Environment.from_spec(environment_spec)  # type: Environment

        # Create the SpecifiableServer with the given env spec.
        if state_space is None or reward_space is None:
            state_space = dummy_env.state_space
            if reward_space is None:
                _, reward, _, _ = dummy_env.step(dummy_env.action_space.sample())
                # TODO: this may break on non 64-bit machines. tf seems to interpret a python float as tf.float64.
                reward_space = "float64" if type(reward) == float else float

        self.state_space = state_space

        # The Problem with ContainerSpaces here is that py_func (SpecifiableServer) cannot handle container
        # spaces, which is why we need to painfully convert these into flat spaces and tuples here whenever
        # we make a call to the env. So to keep things unified, we treat all container spaces
        # (state space, preprocessed state) from here on as tuples of primitive spaces sorted by their would be
        # flat-keys in a flattened dict).

        self.state_space_flattened = self.state_space.flatten()
        # Need to flatten the state-space in case it's a ContainerSpace for the return dtypes.
        self.state_space_list = list(self.state_space_flattened.values())
        self.reward_space = reward_space
        self.add_previous_action = add_previous_action
        self.add_previous_reward = add_previous_reward

        self.environment_spec = environment_spec
        self.environment_server = SpecifiableServer(
            class_=Environment,
            spec=environment_spec,
            output_spaces=dict(
                step=self.state_space_list + [self.reward_space, bool, None],
                reset=self.state_space_list
            ),
            shutdown_method="terminate"
        )
        self.action_space = None

        # Add the sub-components.
        self.actor_component = ActorComponent.from_spec(actor_component_spec)  # type: ActorComponent
        # ActorComponent has - maybe - more than one preprocessor.
        # Collect a flattened ordered dict of preprocessed Spaces here (including non-preprocessed components of the
        # state space).
        self.preprocessed_state_space = OrderedDict()
        for flat_key, space in self.state_space_flattened.items():
            self.preprocessed_state_space[flat_key] = self.actor_component.preprocessors[flat_key].\
                get_preprocessed_space(self.state_space_flattened[flat_key]) if \
                flat_key in self.actor_component.preprocessors else space

        # Variables that hold information of last step through Env.
        self.episode_return = None
        self.current_terminal = None
        self.current_state = None

        self.has_rnn = self.actor_component.policy.neural_network.has_rnn()
        self.internal_state_spaces = None

        # Add all sub-components (only ActorComponent).
        self.add_components(self.actor_component)

        # Define our API methods.
        self.define_api_method("reset", self._graph_fn_reset)
        self.define_api_method("step", self._graph_fn_step)

    def check_input_spaces(self, input_spaces, action_space=None):
        if self.has_rnn:
            self.internal_state_spaces = input_spaces["internal_states"]

    def create_variables(self, input_spaces, action_space=None):
        self.episode_return = self.get_variable(name="episode-return", initializer=0.0, dtype=self.reward_space)
        self.current_terminal = self.get_variable(name="current-terminal", dtype="bool", initializer=False)
        self.current_state = self.get_variable(name="current-state", from_space=self.state_space, flatten=True)

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
            # Store current state (support ContainerSpaces as well) in our variable(s).
            assigns = [
                self.assign_variable(var, s) for var, s in zip(list(self.current_state.values()), state_after_reset)
            ]
            # Store current return and whether current state is terminal.
            assigns.append(tf.variables_initializer(var_list=[self.episode_return, self.current_terminal]))

            with tf.control_dependencies(assigns):
                return tf.no_op()

    def _graph_fn_step(self, internal_states=None, num_steps=1, time_step=0):
        """
        Performs n steps through the environment starting with the current state of the environment and returning
        accumulated tensors for the n steps.

        Args:
            internal_states (DataOp): The internal states data being passed to the ActorComponent if it carries an RNN.
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
        if get_backend() == "tf":

            def scan_func(accum, time_delta):
                # Not needed: preprocessed-previous-states (tuple!)
                # `state` is a tuple as well. See comment in ctor for why tf cannot use ContainerSpaces here.
                if self.has_rnn is False:
                    _, prev_a, prev_r, episode_return, terminal, state = accum
                    internal_states = None
                else:
                    _, prev_a, prev_r, episode_return, terminal, state, internal_states = accum

                # Add control dependency to make sure we don't step parallelly through the Env.
                terminal = tf.convert_to_tensor(value=terminal)
                with tf.control_dependencies(control_inputs=[terminal]):
                    # If state (s) was terminal, reset the env (in this case, we will never need s (or a preprocessed
                    # version thereof for any NN runs (q-values, probs, values, etc..) as no actions are taken from s).
                    state = tf.cond(
                        pred=terminal,
                        true_fn=lambda: force_tuple(self.environment_server.reset()),
                        false_fn=lambda: tuple(tf.convert_to_tensor(s) for s in state)
                    )

                flat_state = OrderedDict()
                for i, flat_key in enumerate(self.state_space_flattened.keys()):
                    # Add a simple (size 1) batch rank to the state so it'll pass through the NN.
                    # - Also maybe have to add a time-rank for RNN processing.
                    expanded = state[i]
                    for _ in range(1 if self.has_rnn is False else 2):
                        expanded = tf.expand_dims(input=expanded, axis=0)
                    # Make None so it'll be recognized as batch-rank by the auto-Space detector.
                    flat_state[flat_key] = tf.placeholder_with_default(
                        input=expanded, shape=(None,) + ((None,) if self.has_rnn is True else ()) +
                                              self.state_space_list[i].shape
                    )

                # Recreate state as the original Space to pass it into the action-component.
                state = unflatten_op(flat_state)

                # Add prev_a and/or prev_r?
                if self.add_previous_action is True:
                    assert isinstance(state, dict), "ERROR: Cannot add previous action to non Dict state space!"
                    for _ in range(1 if self.has_rnn is False else 2):
                        prev_a = tf.expand_dims(prev_a, axis=0)
                    state["previous_action"] = tf.placeholder_with_default(
                        prev_a, shape=(None,) + ((None,) if self.has_rnn is True else ()) + self.action_space.shape
                    )
                if self.add_previous_reward is True:
                    assert isinstance(state, dict), "ERROR: Cannot add previous reward to non Dict state space!"
                    # Always cast rewards to float32 (to match possible other spaces when concatenating).
                    prev_r = tf.cast(prev_r, dtype=dtype_("float"))
                    for _ in range(2 if self.has_rnn is False else 3):  # 2=batch+value rank; 3=batch+time+value rank
                        prev_r = tf.expand_dims(prev_r, axis=0)
                    state["previous_reward"] = tf.placeholder_with_default(
                        prev_r, shape=(None,) + ((None,) if self.has_rnn is True else ()) + (1,)  # 1 node
                    )

                # Get action and preprocessed state (as batch-size 1).
                out = self.call(
                    self.actor_component.get_preprocessed_state_and_action,
                    state,
                    DataOpTuple(internal_states),  # <- None for non-RNN systems
                    time_step=time_step + time_delta,
                    return_ops=True
                )

                # Get output depending on whether it contains internal_states or not.
                if self.has_rnn is True:
                    preprocessed_s, a, current_internal_states = out
                else:
                    preprocessed_s, a = out
                    current_internal_states = None

                # Remove the prev_a, prev_r again (not really part of the state).
                if self.add_previous_action is True:
                    del preprocessed_s["previous_action"]
                if self.add_previous_reward is True:
                    del preprocessed_s["previous_reward"]

                # Strip the batch (and maybe time) ranks again from the action in case the Env doesn't like it.
                a = a[0, 0] if self.has_rnn is True else a[0]
                # Step through the Env and collect next state (tuple!), reward and terminal as single values
                # (not batched).
                out = self.environment_server.step(a)
                s_, r, t_ = out[:-2], out[-2], out[-1]

                # Add up return (if s was not terminal).
                new_episode_return = tf.where(condition=terminal, x=r, y=(r + episode_return))

                # Accumulate return values (remove batch (and maybe time) rank again from preprocessed_s).
                # - preprocessed_s is also still a possible container, make it a tuple again.
                preprocessed_s_no_batch = tuple(map(
                    lambda tensor: (tensor[0, 0] if self.has_rnn is True else tensor[0]),
                    flatten_op(preprocessed_s).values()
                ))
                # Note: Preprocessed_s and s_ are packed as tuples.
                ret = [preprocessed_s_no_batch, a, r, new_episode_return, t_, s_]
                # Add internal_states of an RNN to the return->input cycle.
                if self.has_rnn is True:
                    ret.append(current_internal_states)
                return tuple(ret)

            # Initialize the tf.scan run.
            initializer = [
                tuple(map(lambda space: space.zeros(), self.preprocessed_state_space.values())),
                self.action_space.zeros(),  # zero previous action (doesn't matter)
                np.asarray(0.0, dtype=dtype_(self.reward_space, "np")),  # zero previous reward (doesn't matter)
                self.episode_return,  # return so far
                self.current_terminal,  # whether the current state is terminal
                tuple(self.current_state.values())  # current (raw) state (flattened components if ContainerSpace).
            ]
            # Append internal states if needed.
            if internal_states is not None:
                initializer.append(internal_states)

            # Scan over n time-steps (tf.range produces the time_delta with respect to the current time_step).
            step_results = list(tf.scan(fn=scan_func, elems=tf.range(num_steps), initializer=tuple(initializer)))

            # Store the return so far, current terminal and current state.
            assigns = [
                self.assign_variable(self.episode_return, step_results[3][-1]),
                self.assign_variable(self.current_terminal, step_results[4][-1])
            ]

            # Re-build DataOpDicts from preprocessed-states and states (from tuple right now).
            rebuild_preprocessed_s = DataOpDict()
            rebuild_s = DataOpDict()
            for flat_key, var_ref, preprocessed_s_comp, s_comp in zip(
                    self.state_space_flattened.keys(), self.current_state.values(), step_results[0], step_results[5]
            ):
                assigns.append(self.assign_variable(var_ref, s_comp[-1]))  # -1: current state (last observed)
                rebuild_preprocessed_s[flat_key] = preprocessed_s_comp
                rebuild_s[flat_key] = s_comp
            rebuild_preprocessed_s = unflatten_op(rebuild_preprocessed_s)
            rebuild_s = unflatten_op(rebuild_s)
            step_results[0] = rebuild_preprocessed_s
            step_results[5] = rebuild_s

            with tf.control_dependencies(control_inputs=assigns):
                step_op = tf.no_op()

            return step_op, DataOpTuple(step_results)
