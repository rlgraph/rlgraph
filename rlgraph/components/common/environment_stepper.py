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
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.environments.environment import Environment
from rlgraph.utils.specifiable_server import SpecifiableServer

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

    def __init__(self, environment_spec, actor_component_spec, state_space=None, **kwargs):
        """
        Args:
            environment_spec (dict): A specification dict for constructing an Environment object that will be run
                inside a SpecifiableServer for in-graph stepping.
            actor_component_spec (Union[ActorComponent,dict]): A specification dict to construct this EnvStepper's
                ActionComponent (to generate actions) or an already constructed ActionComponent object.
            state_space (Optional[Space]): The state Space of the Environment. If None, will construct a dummy
                environment to get the state  Space from there.
        """
        super(EnvironmentStepper, self).__init__(scope=kwargs.pop("scope", "env-stepper"), **kwargs)

        # Create the SpecifiableServer with the given env spec.
        if state_space is None:
            dummy_env = Environment.from_spec(environment_spec)
            state_space = dummy_env.state_space

        self.environment_spec = environment_spec
        self.environment_server = SpecifiableServer(
            Environment, environment_spec, dict(step=[state_space, float, bool, None]), "terminate"
        )

        # Add the sub-components.
        self.actor_component = ActorComponent.from_spec(actor_component_spec)
        self.add_components(self.actor_component)

        # Define our API methods.
        self.define_api_method("step", self._graph_fn_step)

    def _graph_fn_step(self, num_steps, previous_state=None, was_terminal=True):
        """
        Performs n steps through the environment starting with the current state of the environment and returning
        accumulated tensors for the n steps.

        Args:
            num_steps (int): The number of steps to perform in the environment.
            previous_state (any): The previously returned state, which is usually the last observed state (next state)
                of the environment (after the last step). If None:

        Returns:
            tuple:
                - preprocessed_states: Starting with the initial state of the environment and ending with the last state
                    reached.
                - actions_taken: The actions actually picked by our policy.
                - action_log_probs: The log-probabilities of all actions per step.
                TODO: add more necessary stats here.
        """

        """
        Timeline:
        1) Session starts -> server is started: Env is created on server.
        2) Call step(3, None)
        3) 
        """

        if get_backend() == "tf":
            def scan_func(accum, _):
                state, action, reward, terminal = accum

                # Do an API-method call here to get the next action,
                # making sure that the previous step has been completed.
                tensor_state = tf.convert_to_tensor(state)
                #with tf.control_dependencies([tensor_state]):
                preprocessed_s, a = self.call(self.actor_component.get_preprocessed_state_and_action, tensor_state)
                # Step through the Env.
                s_, r, t, _ = self.environment_server.step(a)
                # Accumulate return values.
                return s_, a, r, t

            initializer = []
            n_steps = tf.scan(fn=scan_func, elems=tf.range(num_steps), initializer=initializer)
            return n_steps
