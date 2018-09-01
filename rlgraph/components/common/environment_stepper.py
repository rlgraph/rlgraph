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

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.environments.environment import Environment
from rlgraph.utils.ops import DataOpTuple, DataOpDict, flatten_op, unflatten_op
from rlgraph.spaces import Space, Dict
from rlgraph.utils.specifiable_server import SpecifiableServer
from rlgraph.utils.util import force_tuple

if get_backend() == "tf":
    import tensorflow as tf


class EnvironmentStepper(Component):
    """
    A Component that takes an Environment object, a PreprocessorStack and a Policy to step
    n times through the environment, each time picking actions depending on the states that the environment produces.

    API:
        reset(): Resets the Environment stepper including its environment and gets everything ready for stepping.
            Resets the stored state, return and terminal of the env.
        step(): Performs n steps through the environment and returns some
            collected stats: preprocessed_states, actions taken, (optional: action log-probabilities)?, rewards,
            accumulated episode returns, terminals, next states (un-preprocessed), (optional: internal states, only
            for RNN based ActorComponents).
    """

    def __init__(self, environment_spec, actor_component_spec, num_steps=20, state_space=None, reward_space=None,
                 internal_states_space=None,
                 add_action_probs=False, action_probs_space=None, add_previous_action=False, add_previous_reward=False,
                 **kwargs):
        """
        Args:
            environment_spec (dict): A specification dict for constructing an Environment object that will be run
                inside a SpecifiableServer for in-graph stepping.
            actor_component_spec (Union[ActorComponent,dict]): A specification dict to construct this EnvStepper's
                ActionComponent (to generate actions) or an already constructed ActionComponent object.
            num_steps (int): The number of steps to perform per `step` call.
            state_space (Optional[Space]): The state Space of the Environment. If None, will construct a dummy
                environment to get the state Space from there.
            reward_space (Optional[Space]): The reward Space of the Environment. If None, will construct a dummy
                environment to get the reward Space from there.
            internal_states_space (Optional[Space]): The internal states Space (when using an RNN inside the
                ActorComponent).
            add_action_probs (bool): Whether to add all action probabilities for each step to the ActionComponent's
                outputs at each step. These will be added as additional tensor inside the
                Default: False.
            action_probs_space (Optional[Space]): If add_action_probs is True, the Space that the action_probs will have.
                This is usually just the flattened (one-hot) action space.
            add_previous_action (bool): Whether to add the previous action as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_action". Default: False.
            add_previous_reward (bool): Whether to add the previous reward as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_reward". Default: False.
        """
        super(EnvironmentStepper, self).__init__(scope=kwargs.pop("scope", "environment-stepper"), **kwargs)

        # Only to retrieve some information about the particular Env.
        dummy_env = Environment.from_spec(environment_spec)  # type: Environment

        # Create the SpecifiableServer with the given env spec.
        if state_space is None or reward_space is None:
            state_space = dummy_env.state_space
            if reward_space is None:
                _, reward, _, _ = dummy_env.step(dummy_env.action_space.sample())
                # TODO: this may break on non 64-bit machines. tf seems to interpret a python float as tf.float64.
                reward_space = Space.from_spec(
                    "float64" if type(reward) == float else float, shape=(1,)
                ).with_batch_rank()
        else:
            reward_space = Space.from_spec(reward_space).with_batch_rank()

        self.reward_space = reward_space
        self.action_space = dummy_env.action_space

        dummy_env.terminate()

        # The state that the environment produces.
        self.state_space_env = state_space
        # The state that must be fed into the actor-component to produce an action.
        # May contain prev_action and prev_reward.
        self.state_space_actor = state_space
        self.add_previous_action = add_previous_action
        self.add_previous_reward = add_previous_reward

        # The Problem with ContainerSpaces here is that py_func (SpecifiableServer) cannot handle container
        # spaces, which is why we need to painfully convert these into flat spaces and tuples here whenever
        # we make a call to the env. So to keep things unified, we treat all container spaces
        # (state space, preprocessed state) from here on as tuples of primitive spaces sorted by their would be
        # flat-keys in a flattened dict).
        self.state_space_env_flattened = self.state_space_env.flatten()
        # Need to flatten the state-space in case it's a ContainerSpace for the return dtypes.
        self.state_space_env_list = list(self.state_space_env_flattened.values())

        # TODO: automate this by lookup from the NN Component
        self.internal_states_space = None
        if internal_states_space is not None:
            self.internal_states_space = internal_states_space.with_batch_rank(add_batch_rank=1)

        # Add the action/reward spaces to the state space (must be Dict).
        if self.add_previous_action is True:
            assert isinstance(self.state_space_actor, Dict),\
                "ERROR: If `add_previous_action` is True as input, state_space must be a Dict!"
            self.state_space_actor["previous_action"] = self.action_space
        if self.add_previous_reward is True:
            assert isinstance(self.state_space_actor, Dict),\
                "ERROR: If `add_previous_reward` is True as input, state_space must be a Dict!"
            self.state_space_actor["previous_reward"] = self.reward_space
        self.state_space_actor_flattened = self.state_space_actor.flatten()
        self.state_space_actor_list = list(self.state_space_actor_flattened.values())

        self.add_action_probs = add_action_probs
        self.action_probs_space = action_probs_space

        self.environment_spec = environment_spec
        self.environment_server = SpecifiableServer(
            class_=Environment,
            spec=environment_spec,
            output_spaces=dict(
                step=self.state_space_env_list + [self.reward_space, bool, None],
                reset=self.state_space_env_list
            ),
            shutdown_method="terminate"
        )
        # Add the sub-components.
        self.actor_component = ActorComponent.from_spec(actor_component_spec)  # type: ActorComponent
        self.preprocessed_state_space = self.actor_component.preprocessor.get_preprocessed_space(self.state_space_actor)

        self.num_steps = num_steps

        # Variables that hold information of last step through Env.
        self.episode_return = None
        self.current_terminal = None
        self.current_state = None
        self.current_internal_states = None
        self.time_step = 0

        self.has_rnn = self.actor_component.policy.neural_network.has_rnn()

        # Add all sub-components (only ActorComponent).
        self.add_components(self.actor_component)

        # Define our API methods.
        self.define_api_method("reset", self._graph_fn_reset)
        self.define_api_method("step", self._graph_fn_step)

    def create_variables(self, input_spaces, action_space=None):
        self.time_step = self.get_variable(name="time-step", dtype="int32", initializer=0, trainable=False)
        self.episode_return = self.get_variable(name="episode-return", dtype="float32",
                                                initializer=0.0, trainable=False)
        self.current_terminal = self.get_variable(name="current-terminal", dtype="bool",
                                                  initializer=True, trainable=False)
        self.current_state = self.get_variable(name="current-state", from_space=self.state_space_actor,
                                               flatten=True, trainable=False)
        if self.has_rnn:
            self.current_internal_states = self.get_variable(
                name="current-internal-states", from_space=self.internal_states_space, initializer=0.0,
                flatten=True, trainable=False
            )

    def _graph_fn_reset(self):
        """
        Resets the EnvStepper and stores:
        - current state, current return, current terminal, current internal state (RNN), global time_step
        This is only necessary at the very beginning as the step method itself will take care of resetting the Env
        in between or during stepping runs (depending on terminal signals from the Env).

        Returns:
            SingleDataOp: The assign op that stores the state after the Env reset in `last_state` variable.
        """
        if get_backend() == "tf":
            state_after_reset = self.environment_server.reset()
            # Reset current state (support ContainerSpaces as well) via our variable(s)' initializer.
            assigns = [self.assign_variable(var, s) for var, s in zip(
                    self.current_state.values(), force_tuple(state_after_reset)
            )]
            # Reset internal-states, current return and whether current state is terminal.
            assigns.append(tf.variables_initializer(
                var_list=[self.episode_return, self.current_terminal]
            ))
            if self.has_rnn:
                assigns.append(tf.variables_initializer(var_list=list(self.current_internal_states.values())))

            # Note: self.time_step never gets reset.

            with tf.control_dependencies(assigns):
                return tf.no_op()

    def _graph_fn_step(self):  #, internal_states=None, num_steps=1, time_step=0):
        """
        Performs n steps through the environment starting with the current state of the environment and returning
        accumulated tensors for the n steps.

        #Args:
        #    internal_states (DataOp): The internal states data being passed to the ActorComponent if it carries an RNN.
        #    num_steps (int): The number of steps to perform in the environment.
        #    time_step (int): The time_step at which we start stepping.

        Returns:
            Tuple[SingleDataOp,List[SingleDataOp]]:
                1) The step-op to be pulled to execute the stepping.
                2) The step results folded into a list of items, each one with a time-rank only (no batch rank,
                    b/c single env):
                - preprocessed_previous_states: Starting with the initial state of the environment and ending with
                    the one state before the last element in `next_states` (see below).
                - actions: The actions actually picked by our policy.
                - rewards: The rewards actually observed during stepping.
                - returns: The accumulated reward values for the ongoing episode up to after taking an action.
                TODO: discounting?
                - terminals: The terminal signals from the env. Values refere to whether the states in `next_states`
                    (see below) are terminal or not.
                - next_states: The (non-preprocessed) next states.
                Optional if self.add_action_probs is True:
                - action_log_probs: The log-probabilities of all actions per step.
                Optional if self.has_rnn is True:
                - internal_states: The internal-states outputs of an RNN.
        """
        if get_backend() == "tf":

            def scan_func(accum, time_delta):
                # Not needed: preprocessed-previous-states (tuple!)
                # `state` is a tuple as well. See comment in ctor for why tf cannot use ContainerSpaces here.
                if self.has_rnn is False:
                    if self.add_action_probs is False:
                        _, _, _, episode_return, terminal, state = accum
                    else:
                        _, _, _, episode_return, terminal, state, _ = accum
                    internal_states = None
                else:
                    if self.add_action_probs is False:
                        _, _, _, episode_return, terminal, state, internal_states = accum
                    else:
                        _, _, _, episode_return, terminal, state, _, internal_states = accum

                # Add control dependency to make sure we don't step parallelly through the Env.
                terminal = tf.convert_to_tensor(value=terminal)
                with tf.control_dependencies(control_inputs=[terminal]):
                    # If state (s) was terminal, reset the env (in this case, we will never need s (or a preprocessed
                    # version thereof for any NN runs (q-values, probs, values, etc..) as no actions are taken from s).
                    state = force_tuple(tf.cond(
                        pred=terminal,
                        true_fn=lambda: tuple(force_tuple(self.environment_server.reset()) +
                                              ((self.action_space.zeros(),) if self.add_previous_action else ()) +
                                              ((self.reward_space.zeros(),) if self.add_previous_reward else ())
                                              ),
                        false_fn=lambda: tuple(tf.convert_to_tensor(s) for s in state)
                    ))

                flat_state = OrderedDict()
                for i, flat_key in enumerate(self.state_space_actor_flattened.keys()):
                    # Add a simple (size 1) batch rank to the state so it'll pass through the NN.
                    # - Also have to add a time-rank for RNN processing.
                    expanded = state[i]
                    for _ in range(1 if self.has_rnn is False else 2):
                        expanded = tf.expand_dims(input=expanded, axis=0)
                    # Make None so it'll be recognized as batch-rank by the auto-Space detector.
                    flat_state[flat_key] = tf.placeholder_with_default(
                        input=expanded, shape=(None,) + ((None,) if self.has_rnn is True else ()) +
                                              self.state_space_actor_list[i].shape
                    )

                # Recreate state as the original Space to pass it into the actor-component.
                state = unflatten_op(flat_state)

                # Get action and preprocessed state (as batch-size 1).
                out = self.call(
                    (self.actor_component.get_preprocessed_state_and_action if self.add_action_probs is False else
                     self.actor_component.get_preprocessed_state_action_and_action_probs),
                    state,
                    # Add simple batch rank to internal_states.
                    None if internal_states is None else DataOpTuple(internal_states),  # <- None for non-RNN systems
                    time_step=self.time_step + time_delta,
                    return_ops=True
                )

                # Get output depending on whether it contains internal_states or not.
                current_internal_states = None
                action_probs = None
                if self.has_rnn is True:
                    if self.add_action_probs is False:
                        preprocessed_s, a, current_internal_states = out
                    else:
                        preprocessed_s, a, action_probs, current_internal_states = out
                else:
                    if self.add_action_probs is False:
                        preprocessed_s, a = out
                    else:
                        preprocessed_s, a, action_probs = out

                # Strip the batch (and maybe time) ranks again from the action in case the Env doesn't like it.
                a_no_extra_ranks = a[0, 0] if self.has_rnn is True else a[0]
                # Step through the Env and collect next state (tuple!), reward and terminal as single values
                # (not batched).
                out = self.environment_server.step(a_no_extra_ranks)
                s_, r, t_ = out[:-2], out[-2], out[-1]
                r = tf.cast(r, dtype="float32")

                # Add up return (if s was not terminal).
                new_episode_return = tf.where(condition=terminal, x=r, y=(r + episode_return))

                # Accumulate return values (remove batch (and maybe time) rank again from preprocessed_s).
                # - preprocessed_s is also still a possible container, make it a tuple again.
                preprocessed_s_no_batch = tuple(map(
                    lambda tensor: (tensor[0, 0] if self.has_rnn is True else tensor[0]),
                    flatten_op(preprocessed_s).values()
                ))

                # Add a and/or r to next_state?
                if self.add_previous_action is True:
                    assert isinstance(s_, tuple), "ERROR: Cannot add previous action to non tuple!"
                    s_ = s_ + (a_no_extra_ranks,)
                if self.add_previous_reward is True:
                    assert isinstance(s_, tuple), "ERROR: Cannot add previous reward to non tuple!"
                    s_ = s_ + (r,)

                # Note: Preprocessed_s and s_ are packed as tuples.
                ret = [preprocessed_s_no_batch, a_no_extra_ranks, r, new_episode_return, t_, s_] + \
                    ([(action_probs[0][0] if self.has_rnn is True else action_probs[0])] if
                     self.add_action_probs is True else []) + \
                    ([tuple(current_internal_states)] if self.has_rnn is True else [])

                return tuple(ret)

            # Initialize the tf.scan run.
            initializer = [
                tuple(map(lambda space: space.zeros(), self.preprocessed_state_space.flatten().values())),
                self.action_space.zeros(),  # zero previous action (doesn't matter)
                self.reward_space.zeros(),  #.asarray(0.0, dtype=self.reward_space.dtype),  # zero previous reward (doesn't matter)
                self.episode_return,  # return so far
                self.current_terminal,  # whether the current state is terminal
                tuple(self.current_state.values())  # current (raw) state (flattened components if ContainerSpace).
            ]
            # Append action probs if needed.
            if self.add_action_probs is True:
                initializer.append(self.action_probs_space.zeros())  # zero action probs (don't matter)
            # Append internal states if needed.
            if self.current_internal_states is not None:
                initializer.append(tuple(
                    tf.placeholder_with_default(
                        tf.expand_dims(internal_s, axis=0), shape=(None,) + tuple(internal_s.shape.as_list())
                    ) for internal_s in self.current_internal_states.values()
                ))

            # Scan over n time-steps (tf.range produces the time_delta with respect to the current time_step).
            step_results = list(tf.scan(fn=scan_func, elems=tf.range(self.num_steps, dtype="int32"),
                                        initializer=tuple(initializer)))

            # Store the time-step increment, return so far, current terminal and current state.
            assigns = [
                tf.assign_add(self.time_step, self.num_steps),
                self.assign_variable(self.episode_return, step_results[3][-1]),
                self.assign_variable(self.current_terminal, step_results[4][-1])
            ]
            # Re-build DataOpDicts from preprocessed-states and states (from tuple right now).
            rebuild_preprocessed_s = DataOpDict()
            rebuild_s = DataOpDict()
            for flat_key, var_ref, preprocessed_s_comp, s_comp in zip(
                    self.state_space_actor_flattened.keys(), self.current_state.values(), step_results[0], step_results[5]
            ):
                assigns.append(self.assign_variable(var_ref, s_comp[-1]))  # -1: current state (last observed)
                rebuild_preprocessed_s[flat_key] = preprocessed_s_comp
                rebuild_s[flat_key] = s_comp
            rebuild_preprocessed_s = unflatten_op(rebuild_preprocessed_s)
            rebuild_s = unflatten_op(rebuild_s)
            step_results[0] = rebuild_preprocessed_s
            step_results[5] = rebuild_s

            # Remove batch rank from internal states again.
            if self.current_internal_states is not None:
                slot = 7 if self.add_action_probs is True else 6
                # TODO: what if internal states is a dict? Right now assume some tuple.
                # TODO: what if internal states is not the last item in the list anymore due to some change
                internal_states_wo_batch = list()
                for i in range(len(step_results[slot])):
                    # 1=batch axis (which is 1); 0=time axis.
                    internal_states_wo_batch.append(tf.squeeze(step_results[-1][i], axis=1))
                step_results[slot] = DataOpTuple(internal_states_wo_batch)

            with tf.control_dependencies(control_inputs=assigns):
                step_op = tf.no_op()

            # Let the auto-infer system know, what time rank we have.
            step_results = DataOpTuple(step_results)
            for o in flatten_op(step_results).values():
                o._time_rank = 0  # which position in the shape is the time-rank?

            return step_op, step_results
