# Copyright 2018/2019 The RLgraph authors, All Rights Reserved.
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
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.specifiable_server import SpecifiableServer

if get_backend() == "tf":
    import tensorflow as tf
    nest = tf.contrib.framework.nest


class EnvironmentStepper(Component):
    """
    A Component that takes an Environment object, a PreprocessorStack and a Policy to step
    n times through the environment, each time picking actions depending on the states that the environment produces.
    """

    def __init__(self, environment_spec, actor_component_spec, num_steps=20,
                 state_space=None, action_space=None, reward_space=None,
                 internal_states_space=None,
                 add_action_probs=False, action_probs_space=None,
                 add_action=False, add_reward=False,
                 add_previous_action_to_state=False, add_previous_reward_to_state=False,
                 scope="environment-stepper",
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
            action_space (Optional[Space]): The action Space of the Environment. If None, will construct a dummy
                environment to get the action Space from there.
            reward_space (Optional[Space]): The reward Space of the Environment. If None, will construct a dummy
                environment to get the reward Space from there.
            internal_states_space (Optional[Space]): The internal states Space (when using an RNN inside the
                ActorComponent).
            add_action_probs (bool): Whether to add all action probabilities for each step to the ActionComponent's
                outputs at each step. These will be added as additional tensor inside the
                Default: False.
            action_probs_space (Optional[Space]): If add_action_probs is True, the Space that the action_probs will have.
                This is usually just the flattened (one-hot) action space.
            add_action (bool): Whether to add the action to the output of the `step` API-method.
                Default: False.
            add_reward (bool): Whether to add the reward to the output of the `step` API-method.
                Default: False.
            add_previous_reward_to_state (bool): Whether to add the previous reward as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_reward". Default: False.
            add_previous_action_to_state (bool): Whether to add the previous action as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_action". Default: False.
            add_previous_reward_to_state (bool): Whether to add the previous reward as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_reward". Default: False.
        """
        super(EnvironmentStepper, self).__init__(scope=scope, **kwargs)

        # Create the SpecifiableServer with the given env spec.
        if state_space is None or reward_space is None or action_space is None:
            # Only to retrieve some information about the particular Env.
            dummy_env = Environment.from_spec(environment_spec)  # type: Environment
            if state_space is None:
                state_space = dummy_env.state_space
            if action_space is None:
                action_space = dummy_env.action_space
            if reward_space is None:
                _, reward, _, _ = dummy_env.step(actions=action_space.sample())
                # TODO: this may break on non 64-bit machines. tf seems to interpret a python float as tf.float64.
                reward_space = Space.from_spec(
                    "float64" if type(reward) == float else float, shape=(1,)
                ).with_batch_rank()
            dummy_env.terminate()

        self.reward_space = Space.from_spec(reward_space).with_batch_rank()
        self.action_space = Space.from_spec(action_space)
        # The state that the environment produces.
        self.state_space_env = Space.from_spec(state_space)
        # The state that must be fed into the actor-component to produce an action.
        # May contain prev_action and prev_reward.
        self.state_space_actor = Space.from_spec(state_space)
        self.add_previous_action_to_state = add_previous_action_to_state
        self.add_previous_reward_to_state = add_previous_reward_to_state

        # Circle actions and/or rewards in `step` API-method?
        self.add_action = add_action
        self.add_reward = add_reward

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
        if self.add_previous_action_to_state is True:
            assert isinstance(self.state_space_actor, Dict),\
                "ERROR: If `add_previous_action_to_state` is True as input, state_space must be a Dict!"
            self.state_space_actor["previous_action"] = self.action_space
        if self.add_previous_reward_to_state is True:
            assert isinstance(self.state_space_actor, Dict),\
                "ERROR: If `add_previous_reward_to_state` is True as input, state_space must be a Dict!"
            self.state_space_actor["previous_reward"] = self.reward_space
        self.state_space_actor_flattened = self.state_space_actor.flatten()
        self.state_space_actor_list = list(self.state_space_actor_flattened.values())

        self.add_action_probs = add_action_probs
        # TODO: Auto-infer of self.action_probs_space from action_space.
        self.action_probs_space = action_probs_space
        if self.add_action_probs is True:
            assert isinstance(self.action_probs_space, Space),\
                "ERROR: If `add_action_probs` is True, must provide an `action_probs_space`!"

        self.environment_spec = environment_spec
        self.environment_server = SpecifiableServer(
            specifiable_class=Environment,
            spec=environment_spec,
            output_spaces=dict(
                step_flow=self.state_space_env_list + [self.reward_space, bool],
                reset_flow=self.state_space_env_list
            ),
            shutdown_method="terminate"
        )
        # Add the sub-components.
        self.actor_component = ActorComponent.from_spec(actor_component_spec)  # type: ActorComponent
        self.preprocessed_state_space = self.actor_component.preprocessor.get_preprocessed_space(self.state_space_actor)

        self.num_steps = num_steps

        # Variables that hold information of last step through Env.
        self.current_state = None
        self.current_internal_states = None
        self.time_step = 0

        self.has_rnn = self.actor_component.policy.neural_network.has_rnn()

        # Add all sub-components (only ActorComponent).
        self.add_components(self.actor_component)

    def create_variables(self, input_spaces, action_space=None):
        self.time_step = self.get_variable(
            name="time-step", dtype="int32", initializer=0, trainable=False, local=True, use_resource=True
        )
        self.current_state = self.get_variable(
            name="current-state", from_space=self.state_space_actor, initializer=0, flatten=True, trainable=False,
            local=True, use_resource=True
        )
        if self.has_rnn:
            self.current_internal_states = self.get_variable(
                name="current-internal-states", from_space=self.internal_states_space,
                initializer=0.0, flatten=True, trainable=False, local=True, use_resource=True,
                add_batch_rank=1
            )

    @rlgraph_api(returns=1)
    def _graph_fn_step(self):
        if get_backend() == "tf":
            def scan_func(accum, time_delta):
                # Not needed: preprocessed-previous-states (tuple!)
                # `state` is a tuple as well. See comment in ctor for why tf cannot use ContainerSpaces here.
                internal_states = None
                state = accum[1]
                if self.has_rnn:
                    internal_states = accum[-1]

                state = tuple(tf.convert_to_tensor(value=s) for s in state)

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
                out = (self.actor_component.get_preprocessed_state_and_action if self.add_action_probs is False else
                       self.actor_component.get_preprocessed_state_action_and_action_probs)(
                    state,
                    # Add simple batch rank to internal_states.
                    None if internal_states is None else DataOpTuple(internal_states),  # <- None for non-RNN systems
                    time_step=self.time_step + time_delta
                )

                # Get output depending on whether it contains internal_states or not.
                a = out["action"]
                action_probs = out.get("action_probs")
                current_internal_states = out.get("last_internal_states")

                # Strip the batch (and maybe time) ranks again from the action in case the Env doesn't like it.
                a_no_extra_ranks = a[0, 0] if self.has_rnn is True else a[0]
                # Step through the Env and collect next state (tuple!), reward and terminal as single values
                # (not batched).
                out = self.environment_server.step_flow(a_no_extra_ranks)
                s_, r, t_ = out[:-2], out[-2], out[-1]
                r = tf.cast(r, dtype="float32")

                # Add a and/or r to next_state?
                if self.add_previous_action_to_state is True:
                    assert isinstance(s_, tuple), "ERROR: Cannot add previous action to non tuple!"
                    s_ = s_ + (a_no_extra_ranks,)
                if self.add_previous_reward_to_state is True:
                    assert isinstance(s_, tuple), "ERROR: Cannot add previous reward to non tuple!"
                    s_ = s_ + (r,)

                # Note: s_ is packed as tuple.
                ret = [t_, s_] + \
                    ([a_no_extra_ranks] if self.add_action else []) + \
                    ([r] if self.add_reward else []) + \
                    ([(action_probs[0][0] if self.has_rnn is True else action_probs[0])] if
                     self.add_action_probs is True else []) + \
                    ([tuple(current_internal_states)] if self.has_rnn is True else [])

                return tuple(ret)

            # Initialize the tf.scan run.
            initializer = [
                # terminals
                tf.zeros(shape=(), dtype=tf.bool),
                # current (raw) state (flattened components if ContainerSpace).
                tuple(map(lambda x: x.read_value(), self.current_state.values()))
            ]
            # Append actions and rewards if needed.
            if self.add_action:
                initializer.append(tf.zeros(shape=self.action_space.shape, dtype=self.action_space.dtype))
            if self.add_reward:
                initializer.append(tf.zeros(shape=self.reward_space.shape))
            # Append action probs if needed.
            if self.add_action_probs is True:
                initializer.append(tf.zeros(shape=self.action_probs_space.shape))
            # Append internal states if needed.
            if self.current_internal_states is not None:
                initializer.append(tuple(
                    tf.placeholder_with_default(
                        internal_s.read_value(), shape=(None,) + tuple(internal_s.shape.as_list()[1:])
                    ) for internal_s in self.current_internal_states.values()
                ))

            # Scan over n time-steps (tf.range produces the time_delta with respect to the current time_step).
            # NOTE: Changed parallel to 1, to resolve parallel issues.
            step_results = list(tf.scan(
                fn=scan_func, elems=tf.range(self.num_steps, dtype="int32"), initializer=tuple(initializer),
                back_prop=False
            ))

            # Assign all values that need to be passed again into the next scan.
            assigns = [tf.assign_add(self.time_step, self.num_steps)]  # time step
            # State (or flattened state components).
            for flat_key, var_ref, state_comp in zip(
                    self.state_space_actor_flattened.keys(), self.current_state.values(), step_results[1]
            ):
                assigns.append(self.assign_variable(var_ref, state_comp[-1]))  # -1: current state (last observed)

            # Current internal state.
            if self.current_internal_states is not None:
                # TODO: What if internal states is not the last item in the list anymore due to some change.
                slot = -1
                # TODO: What if internal states is a dict? Right now assume some tuple.
                # Remove batch rank from internal states again.
                internal_states_wo_batch = list()
                for i, var_ref in enumerate(self.current_internal_states.values()):  #range(len(step_results[slot])):
                    # 1=batch axis (which has dim=1); 0=time axis.
                    internal_states_component = tf.squeeze(step_results[slot][i], axis=1)
                    assigns.append(self.assign_variable(var_ref, internal_states_component[-1:]))
                    internal_states_wo_batch.append(internal_states_component)
                step_results[slot] = tuple(internal_states_wo_batch)

            # Concatenate first and rest (and make the concatenated tensors (which are the important return information)
            # dependent on the assigns).
            with tf.control_dependencies(control_inputs=assigns):
                full_results = []
                for slot in range(len(step_results)):
                    first_values, rest_values = initializer[slot], step_results[slot]
                    # Internal states need a slightly different concatenating as the batch rank is missing.
                    if self.current_internal_states is not None and slot == len(step_results) - 1:
                        full_results.append(nest.map_structure(self._concat, first_values, rest_values))
                    # States need concatenating (first state needed).
                    elif slot == 1:
                        full_results.append(nest.map_structure(
                            lambda first, rest: tf.concat([[first], rest], axis=0), first_values, rest_values)
                        )
                    # Everything else does not need concatenating (saves one op).
                    else:
                        full_results.append(step_results[slot])

            # Re-build DataOpDicts of states (from tuple right now).
            rebuild_s = DataOpDict()
            for flat_key, var_ref, s_comp in zip(
                    self.state_space_actor_flattened.keys(), self.current_state.values(), full_results[1]
            ):
                rebuild_s[flat_key] = s_comp
            rebuild_s = unflatten_op(rebuild_s)
            full_results[1] = rebuild_s

            # Let the auto-infer system know, what time rank we have.
            full_results = DataOpTuple(full_results)
            for o in flatten_op(full_results).values():
                o._time_rank = 0  # which position in the shape is the time-rank?

            return full_results

    @staticmethod
    def _concat(first, rest):
        """
        Helper method to concat initial value and scanned collected results.
        """
        shape = first.shape.as_list()
        first.set_shape(shape=(1,) + tuple(shape[1:]))
        return tf.concat([first, rest], axis=0)

    #@rlgraph_api
    def step_with_dict_return(self):
        """
        Simple wrapper to get a dict returned instead of a tuple of values.

        Returns:
            Dict:
                - `terminals`: The is-terminal signals.
                - `states`: The states.
                - `actions` (optional): The actions actually taken.
                - `rewards` (optional): The rewards actually received.
                - `action_probs` (optional): The action probabilities.
                - `internal_states` (optional): The internal-states (only for RNN type NNs in the ActorComponent).
        """
        """
        out = self.step()
        ret = dict(
            terminals=out[0],
            states=out[1]
        )
        if self.has_rnn:
            ret["internal_states"] = out[-1]

        plus = 0
        if self.add_action:
            ret["actions"] = out[2]
            plus += 1
        if self.add_reward:
            ret["rewards"] = out[2 + plus]
            plus += 1
        if self.add_action_probs:
            ret["action_probs"] = out[2 + plus]
            plus += 1

        return ret
        """
        pass