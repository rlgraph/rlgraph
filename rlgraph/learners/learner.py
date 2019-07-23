# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

from abc import abstractmethod, ABC
import logging

from rlgraph import get_backend
from rlgraph.graphs.graph_builder import GraphBuilder
from rlgraph.graphs.graph_executor import GraphExecutor
from rlgraph.spaces import Space, ContainerSpace
from rlgraph.utils.input_parsing import parse_execution_spec
from rlgraph.utils.specifiable import Specifiable


# TODO: Make Agent also a Learner subclass at some point.
class Learner(ABC, Specifiable):
    """
    A Learner uses a SupervisedModel Component for neural-network based
    learning mechanisms.
    """
    def __init__(self, input_space, output_space, execution_spec=None,
                 name="Learner", **kwargs):
        """
        Builds a basic RLGraph learner.

        Args:
            input_space (Union[dict,Space]): Spec dict for the input Space or a direct Space object.
            output_space (Union[dict,Space]): Spec dict for the output Space or a direct Space object.
            network_spec (Optional[list,NeuralNetwork]): Spec list for a NeuralNetwork Component or the NeuralNetwork
                object itself.
            execution_spec (Union[dict,Execution]): The spec-dict specifying execution settings.
            optimizer_spec (Union[dict,Optimizer]): The spec-dict to create the Optimizer for this Learner.
        """
        super(Learner, self).__init__(**kwargs)
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.input_space = Space.from_spec(input_space).with_batch_rank(False)
        self.flat_input_space = self.input_space.flatten() if isinstance(self.input_space, ContainerSpace) else None
        self.logger.info("Parsed input space definition: {}".format(self.input_space))
        self.output_space = Space.from_spec(output_space).with_batch_rank(False)
        self.flat_output_space = self.output_space.flatten() if isinstance(self.output_space, ContainerSpace) else None
        self.logger.info("Parsed output space definition: {}".format(self.output_space))

        # The Learner's root-Component.
        self.root_component = None

        self.input_spaces = None  # To be specified by child class before build.

        self.execution_spec = parse_execution_spec(execution_spec)

        self.graph_builder = GraphBuilder(action_space=self.output_space)
        self.graph_executor = GraphExecutor.from_spec(
            get_backend(),
            graph_builder=self.graph_builder,
            execution_spec=self.execution_spec
        )
        self.graph_built = False

    def _build_graph(self, root_components, input_spaces, **kwargs):
        """
        Builds the internal graph from the RLGraph meta-graph via the graph executor..
        """
        return self.graph_executor.build(root_components, input_spaces, **kwargs)

    def build(self, build_options=None):
        """
        Builds this agent. This method call only be called if the agent parameter "auto_build"
        was set to False.

        Args:
            build_options (Optional[dict]): Optional build options, see build doc.
        """
        assert not self.graph_built, \
            "ERROR: Attempting to build learner which has already been built. Ensure `auto_build` c'tor arg is set " \
            "to `False` and the `Learner.build` method has not been called twice."

        build_stats = self.graph_executor.build(
            [self.root_component], self.input_spaces, optimizer=self.root_component.optimizer,
            build_options=build_options, batch_size=self.root_component.memory_batch_size
        )

        self.graph_built = True
        return build_stats

    @abstractmethod
    def predict(self, prediction_input):
        """
        Runs an input through the trained network and returns the output sampled
        from some distribution.

        Args:
            prediction_input (Union[np.ndarray, list]): Input.

        Returns:
            any: The prediction outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def get_distribution_parameters(self, prediction_input):
        """
        Runs an input through the trained network and returns the pure distribution-parameter output,
        without the sampling step after that.

        Args:
            prediction_input (Union[np.ndarray, list]): Input.

        Returns:
            any: The distribution parameters returned by the NN.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, batch=None, time_percentage=None):
        """
        Performs a learning update on the given batch (or from a memory) using our
        SupervisedModel.

        Args:
            batch (Optional[any]): An external bath to learn from.

            time_percentage (Optional[float]): The time_percentage (between 0.0 and 1.0) used  to calculate
            decaying/over-time-changing parameters such as learning rates.

        Returns:
            (Union[float, dict]): Loss metrics.
        """
        raise NotImplementedError

    def export_graph(self, filename=None):
        """
        Any algorithm defined as a full-graph, as opposed to mixed (mixed Python and graph control flow)
        should be able to export its graph for deployment.

        Args:
            filename (str): Export path. Depending on the backend, different filetypes may be required.
        """
        self.graph_executor.export_graph_definition(filename)

    def store_model(self, path=None, add_timestep=True):
        """
        Store model using the backend's check-pointing mechanism.

        Args:
            path (str): Path to model directory.

            add_timestep (bool): Indicates if current training step should be appended to exported model.
                If false, may override previous checkpoints.
        """
        self.graph_executor.store_model(path=path, add_timestep=add_timestep)

    def load_model(self, checkpoint_directory=None, checkpoint_path=None):
        """
        Loads model from specified path location using the following semantics:

        If checkpoint directory and checkpoint path are given, attempts to find `checkpoint_path` as relative path from
        `checkpoint_directory`.

        If a checkpoint directory is given but no path (e.g. because timestep of checkpoint is not known in advance),
        attempts to fetch latest check-point.

        If no directory is given, attempts to fetch checkpoint from the full absolute path `checkpoint_path'.

        Args:
            checkpoint_directory (str): Optional path to directory containing checkpoint(s).
            checkpoint_path (str): Path to specific model checkpoint.
        """
        self.graph_executor.load_model(checkpoint_directory=checkpoint_directory, checkpoint_path=checkpoint_path)

    def reset(self):
        """
        Must be implemented to define some reset behavior.
        """
        pass  # optional

    def terminate(self):
        self.graph_executor.terminate()

    def __repr__(self):
        """
        Returns:
            str: A short, but informative description for this Agent.
        """
        raise NotImplementedError
