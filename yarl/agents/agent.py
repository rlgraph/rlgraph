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
from yarl import Specifiable, backend
from yarl.utils.input_parsing import parse_execution_spec, parse_update_spec
from yarl.components import Component
from yarl.components.layers import Stack
from yarl.components.neural_networks import NeuralNetwork
from yarl.components.optimizers import Optimizer
from yarl.models import Model
from yarl.spaces import Space


class Agent(Specifiable):
    """
    Generic agent defining YARL-API operations.
    """
    def __init__(
        self,
        states_spec,
        actions_spec,
        network_spec=None,
        preprocessing_spec=None,
        exploration_spec=None,
        execution_spec=None,
        optimizer_spec=None,
        update_spec=None
    ):
        """
        Generic agent which parses and sanitizes configuration specs.

        Args:
            states_spec (Union[dict,Space]): Spec dict for the state Space or a direct Space object.
            actions_spec (Union[dict,Space]): Spec dict for the action Space or a direct Space object.
            network_spec (Optional[dict,NeuralNetwork]): Spec dict for a NeuralNetwork Component or the NeuralNetwork
                object itself.
            preprocessing_spec (Optional[dict]): The spec dict for the different necessary states preprocessing steps.
            exploration_spec (Optional[dict]):
            execution_spec (Optional[dict]):
            optimizer_spec:
            update_spec (Optional[dict]):
        """
        self.state_space = Space.from_spec(states_spec)
        self.action_space = Space.from_spec(actions_spec)
        self.neural_network = NeuralNetwork.from_spec(network_spec)

        self.preprocessor_stack = Stack.from_spec(preprocessing_spec)
        self.exploration_spec = exploration_spec
        self.execution_spec = parse_execution_spec(execution_spec)
        self.optimizer = Optimizer.from_spec(optimizer_spec)
        self.update_spec = parse_update_spec(update_spec)

        # Create our Model.
        self.model = Model.from_spec(backend, execution_spec=self.execution_spec)  # type: Model

        # Build a custom, agent-specific algorithm.
        self.build_graph(self.model.core_component)
        # Ask our Model to actually build the Graph.
        self.model.build()

    def build_graph(self, core):
        """
        Assembles the computation graph by combining specified graph components
        and converting non-graph components if using AutoGraph.

        Each agent implements this to combine its necessary graph components.

        Args:
            core (Component): The core Component of the Model we are building. All Components this Agent needs must go
                in there via `core.add_component()`.
        """
        raise NotImplementedError

    def build_preprocessor(self, core):
        core.add_component(self.preprocessor_stack)

    def get_action(self, states, deterministic=False):
        """
        Retrieves action(s) for the passed state(s).

        Args:
            states (Union[dict, ndarray]): State dict or array.
            deterministic (bool): If True, no exploration or sampling may be applied
                when retrieving an action.

        Returns: Actions dict.
        """
        raise NotImplementedError

    def observe(self, states, actions, internals, reward, terminal):
        """
        Observes an experience tuple or a batch of experience tuples.

        Args:
            states (Union[dict, ndarray]): States dict or array.
            actions (Union[dict, ndarray]): Actions dict or array containing actions performed for the given state(s).
            internals (Union[list, None]): Internal state(s) returned by agent for the given states.
            reward (float): Scalar reward(s) observed.
            terminal (bool): Boolean indicating terminal.

        """
        raise NotImplementedError

    def update(self):
        """
        Performs an update on the computation graph.

        Returns: Loss value.
        """
        raise NotImplementedError

    def import_observations(self, observations):
        """
        Bulk imports observations, potentially using device pre-fetching. Can be optionally
        implemented by agents requiring pre-training.

        Args:
            observations (dict): Dict or list of observation data.
        """
        pass

    def export_graph(self, filename=None):
        """
        Any algorithm defined as a full-graph, as opposed to mixed (mixed Python and graph control flow)
        should be able to export its graph for deployment.

        Args:
            filename (str): Export path. Depending on the backend, different filetypes may be required.
        """
        self.model.export_graph_definition(filename)

    def store_model(self, path=None, add_timestep=True):
        """
        Store model using the backend's check-pointing mechanism.

        Args:
            path (str): Path to model directory.
            add_timestep (bool): Indiciates if current training step should be appended to
                exported model. If false, may override previous checkpoints.

        """
        self.model.store_model(path=path, add_timestep=add_timestep)

    def load_model(self, path=None):
        """
        Load model from serialized format.

        Args:
            path (str): Path to checkpoint directory.
        """
        self.model.load_model(path=path)
