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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from rlgraph import get_backend
from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.components.component import Component
from rlgraph.components.distributions import Normal, Categorical, Beta
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.components.action_adapters.action_adapter import ActionAdapter
from rlgraph.spaces.space import Space
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.ops import FlattenedDataOp, DataOpDict

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class Policy(Component):
    """
    A Policy is a wrapper Component that contains a NeuralNetwork, an ActionAdapter and a Distribution Component.
    """
    def __init__(self, network_spec, action_space=None, action_adapter_spec=None,
                 deterministic=True, scope="policy", **kwargs):
        """
        Args:
            network_spec (Union[NeuralNetwork,dict]): The NeuralNetwork Component or a specification dict to build
                one.

            action_space (Space): The action Space within which this Component will create actions.

            action_adapter_spec (Optional[dict]): A spec-dict to create an ActionAdapter. Use None for the default
                ActionAdapter object.

            deterministic (bool): Whether to pick actions according to the max-likelihood value or via sampling.
                Default: True.

            batch_apply (bool): Whether to wrap both the NN and the ActionAdapter with a BatchApply Component in order
                to fold time rank into batch rank before a forward pass.
        """
        super(Policy, self).__init__(scope=scope, **kwargs)

        self.neural_network = NeuralNetwork.from_spec(network_spec)  # type: NeuralNetwork

        # Create the necessary action adapters for this Policy. One for each action space component.
        self.action_adapters = {}
        if action_space is None:
            self.action_adapters[""] = ActionAdapter.from_spec(action_adapter_spec)
            self.action_space = self.action_adapters[""].action_space
            # Assert single component action space.
            assert len(self.action_space.flatten()) == 1,\
                "ERROR: Action space must not be ContainerSpace if no `action_space` is given in Policy c'tor!"
        else:
            self.action_space = Space.from_spec(action_space)
            for i, (flat_key, action_component) in enumerate(self.action_space.flatten().items()):
                if isinstance(action_adapter_spec, dict):
                    aa_spec = action_adapter_spec.get(flat_key, action_adapter_spec)
                    aa_spec["action_space"] = action_component
                elif not isinstance(action_adapter_spec, ActionAdapter):
                    aa_spec = dict(action_space=action_component)
                else:
                    aa_spec = action_adapter_spec
                self.action_adapters[flat_key] = ActionAdapter.from_spec(aa_spec, scope="action-adapter-{}".format(i))

        self.deterministic = deterministic

        # Check for bounded FloatBoxes.
        self.bounded_action_space = {}
        for i, (flat_key, action_component) in enumerate(self.action_space.flatten().items()):
            if isinstance(action_component, FloatBox):
                # Unbounded.
                if action_component.low == float("-inf") and action_component.high == float("inf"):
                    self.bounded_action_space[flat_key] = False
                # Bounded.
                elif action_component.low != float("-inf") and action_component.high != float("inf"):
                    self.bounded_action_space[flat_key] = True
                # TODO: Semi-bounded -> Exponential distribution.
                else:
                    raise RLGraphError(
                        "Semi-bounded action spaces are not supported yet! You passed in low={} high={}.".\
                        format(action_component.low, action_component.high)
                    )
            else:
                self.bounded_action_space[flat_key] = None

        # Figure out our Distributions.
        self.distributions = {}
        for i, (flat_key, action_component) in enumerate(self.action_space.flatten().items()):
            if isinstance(action_component, IntBox):
                self.distributions[flat_key] = Categorical(scope="categorical-{}".format(i))
            # Continuous action space -> Normal distribution (each action needs mean and variance from network).
            elif isinstance(action_component, FloatBox):
                # Unbounded -> Normal distribution.
                if self.bounded_action_space is False:
                    self.distributions[flat_key] = Normal(scope="normal-{}".format(i))
                # Bounded -> Beta distribution.
                else:
                    self.distributions[flat_key] = Beta(scope="beta-{}".format(i))
            else:
                raise RLGraphError("ERROR: `action_component` is of type {} and not allowed in {} Component!".
                                   format(type(action_space).__name__, self.name))

        self.add_components(
            *[self.neural_network] + list(self.action_adapters.values()) + list(self.distributions.values())
        )

    # Define our interface.
    @rlgraph_api
    def get_nn_output(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: The raw output of the neural network (before it's cleaned-up and passed through the ActionAdapter).
        """
        out = self.neural_network.apply(nn_input, internal_states)
        return dict(output=out["output"], last_internal_states=out.get("last_internal_states"))

    @rlgraph_api
    def get_action_layer_output(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: The raw output of the action layer of the ActionAdapter (including possibly the last internal states
                of a RNN-based NN).
        """
        nn_output = self.get_nn_output(nn_input, internal_states)
        action_layer_outputs = self._graph_fn_get_action_layer_outputs(nn_output["output"], nn_input)
        # Add last internal states to return value.
        return dict(output=action_layer_outputs, last_internal_states=nn_output["last_internal_states"])

    @rlgraph_api
    def get_logits_parameters_log_probs(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                logits: The (reshaped) logits from the ActionAdapter.
                parameters: The parameters for the distribution (gained from the softmaxed logits or interpreting
                    logits as mean and stddev for a normal distribution).
                log_probs: The log(probabilities) values.
        """
        nn_output = self.get_nn_output(nn_input, internal_states)
        logits, parameters, log_probs = self._graph_fn_get_action_adapter_logits_parameters_log_probs(
            nn_output["output"], nn_input
        )

        return dict(
            logits=logits, parameters=parameters, log_probs=log_probs,
            last_internal_states=nn_output["last_internal_states"]
        )

    @rlgraph_api
    def get_logits_probabilities_log_probs(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                logits: The (reshaped) logits from the ActionAdapter.
                probabilities: The probabilities gained from the softmaxed logits.
                log_probs: The log(probabilities) values.
        """
        self.logger.warn("Deprecated API method `get_logits_probabilities_log_probs` used!"
                         "Use `get_logits_parameters_log_probs` instead.")
        nn_output = self.get_nn_output(nn_input, internal_states)
        logits, parameters, log_probs = self._graph_fn_get_action_adapter_logits_parameters_log_probs(
            nn_output["output"], nn_input
        )

        return dict(
            logits=logits, probabilities=parameters, parameters=parameters, log_probs=log_probs,
            last_internal_states=nn_output["last_internal_states"]
        )

    @rlgraph_api
    def get_action(self, nn_input, internal_states=None, deterministic=None):
        """
        Returns an action based on NN output, action adapter output and distribution sampling.

        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.
            deterministic (Optional[bool]): If not None, use this to determine whether actions should be drawn
                from the distribution in max-likelihood (deterministic) or stochastic fashion.

        Returns:
            dict:
                `action`: The drawn action.
                `last_internal_states`: The last internal states (if NN is RNN based, otherwise: None).
        """
        deterministic = self.deterministic if deterministic is None else deterministic

        out = self.get_logits_parameters_log_probs(nn_input, internal_states)
        action = self._graph_fn_get_action_components(out["logits"], out["parameters"], deterministic)

        return dict(action=action, last_internal_states=out["last_internal_states"], logits=out["logits"],
                    parameters=out["parameters"], log_probs=out["log_probs"])

    @rlgraph_api
    def get_action_log_probs(self, nn_input, actions, internal_states=None):
        """
        Computes the log-likelihood for a given set of actions under the distribution induced by a set of states.

        Args:
            nn_input (any): The input to our neural network.
            actions (any): The actions for which to get log-probs returned.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Log-probs of actions under current policy
        """
        out = self.get_logits_parameters_log_probs(nn_input, internal_states)

        # Probabilities under current action.
        action_log_probs = self._graph_fn_get_distribution_log_probs(out["parameters"], actions)

        return dict(action_log_probs=action_log_probs, logits=out["logits"],
                    last_internal_states=out["last_internal_states"])

    @rlgraph_api
    def get_deterministic_action(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: See `get_action`, but with deterministic force set to True.
        """
        out = self.get_logits_parameters_log_probs(nn_input, internal_states)
        action = self._graph_fn_get_action_components(out["logits"], out["parameters"], True)

        return dict(action=action, last_internal_states=out["last_internal_states"])

    @rlgraph_api
    def get_stochastic_action(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: See `get_action`, but with deterministic force set to False.
        """
        out = self.get_logits_parameters_log_probs(nn_input, internal_states)
        action = self._graph_fn_get_action_components(out["logits"], out["parameters"], False)

        return dict(action=action, last_internal_states=out["last_internal_states"])

    @rlgraph_api
    def get_entropy(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: See Distribution component.
        """
        out = self.get_logits_parameters_log_probs(nn_input, internal_states)
        entropy = self._graph_fn_get_distribution_entropies(out["parameters"])

        return dict(entropy=entropy, last_internal_states=out["last_internal_states"])

    @graph_fn(flatten_ops={1})
    def _graph_fn_get_action_layer_outputs(self, nn_output, nn_input):
        """
        Pushes the given nn_output through all our action adapters and returns a DataOpDict with the keys corresponding
        to our `action_space`.

        Args:
            nn_output (DataOp): The output of our neural network.

        Returns:
            FlattenedDataOp: A DataOpDict with the different action adapter outputs (keys correspond to
                structure of `self.action_space`).
        """
        if isinstance(nn_input, FlattenedDataOp):
            nn_input = next(iter(nn_input.values()))

        ret = FlattenedDataOp()
        for flat_key, action_adapter in self.action_adapters.items():
            ret[flat_key] = action_adapter.get_logits(nn_output, nn_input)

        return ret

    @graph_fn(flatten_ops={1})
    def _graph_fn_get_action_adapter_logits_parameters_log_probs(self, nn_output, nn_input):
        """
        Pushes the given nn_output through all our action adapters' get_logits_parameters_log_probs API's and
        returns a DataOpDict with the keys corresponding to our `action_space`.

        Args:
            nn_output (DataOp): The output of our neural network.

        Returns:
            tuple:
                - FlattenedDataOp: A DataOpDict with the different action adapters' logits outputs.
                - FlattenedDataOp: A DataOpDict with the different action adapters' parameters outputs.
                - FlattenedDataOp: A DataOpDict with the different action adapters' log_probs outputs.
            Note: Keys always correspond to structure of `self.action_space`.
        """
        logits = FlattenedDataOp()
        parameters = FlattenedDataOp()
        log_probs = FlattenedDataOp()

        if isinstance(nn_input, dict):
            nn_input = next(iter(nn_input.values()))

        for flat_key, action_adapter in self.action_adapters.items():
            out = action_adapter.get_logits_parameters_log_probs(nn_output, nn_input)
            logits[flat_key], parameters[flat_key], log_probs[flat_key] = \
                out["logits"], out["parameters"], out["log_probs"]

        return logits, parameters, log_probs

    @graph_fn
    def _graph_fn_get_distribution_entropies(self, parameters):
        """
        Pushes the given `probabilities` through all our distributions' `entropy` API-methods and returns a
        DataOpDict with the keys corresponding to our `action_space`.

        Args:
            parameters (DataOp): The parameters to define a distribution. This could be a ContainerDataOp, which
                container the parameter pieces for each action component.

        Returns:
            FlattenedDataOp: A DataOpDict with the different distributions' `entropy` outputs. Keys always correspond to
                structure of `self.action_space`.
        """
        ret = FlattenedDataOp()
        for flat_key, d in self.distributions.items():
            if flat_key == "":
                if isinstance(parameters, FlattenedDataOp):
                    return d.entropy(parameters[flat_key])
                else:
                    return d.entropy(parameters)
            else:
                ret[flat_key] = d.entropy(parameters.flat_key_lookup(flat_key))
        return ret

    @graph_fn
    def _graph_fn_get_distribution_log_probs(self, parameters, actions):
        """
        Pushes the given `probabilities` and actions through all our distributions' `log_prob` API-methods and returns a
        DataOpDict with the keys corresponding to our `action_space`.

        Args:
            parameters (DataOp): The parameters to define a distribution.
            actions (DataOp): The actions for which to return the log-probs.

        Returns:
            FlattenedDataOp: A DataOpDict with the different distributions' `log_prob` outputs. Keys always correspond
                to structure of `self.action_space`.
        """
        ret = FlattenedDataOp()
        for flat_key, action_space_component in self.action_space.flatten().items():
            low, high = action_space_component.tensor_backed_bounds()
            if flat_key == "":
                # For bounded continuous action spaces, need to unscale (0.0 to 1.0 for beta distribution).
                if self.bounded_action_space[flat_key] is True:
                    actions = (actions - low) / (high - low)
                if isinstance(parameters, FlattenedDataOp):
                    return self.distributions[flat_key].log_prob(parameters[flat_key], actions)
                else:
                    return self.distributions[flat_key].log_prob(parameters, actions)
            else:
                # For bounded continuous action spaces, need to unscale (0.0 to 1.0 for beta distribution).
                action_value = actions.flat_key_lookup(flat_key)
                if self.bounded_action_space[flat_key] is True:
                    action_value = (action_value - low) / \
                               (high - low)
                ret[flat_key] = self.distributions[flat_key].log_prob(
                    parameters.flat_key_lookup(flat_key), action_value
                )
        return ret

    @graph_fn
    def _graph_fn_get_action_components(self, logits, parameters, deterministic):
        ret = FlattenedDataOp()
        for flat_key, action_space_component in self.action_space.flatten().items():
            # Skip our distribution, iff discrete action-space and deterministic acting (greedy).
            # In that case, one does not need to create a distribution in the graph each act (only to get the argmax
            # over the logits, which is the same as the argmax over the probabilities (or log-probabilities)).
            if isinstance(action_space_component, IntBox) and \
                    (deterministic is True or (isinstance(deterministic, np.ndarray) and deterministic)):
                if flat_key == "":
                    return self._graph_fn_get_deterministic_action_wo_distribution(logits)
                else:
                    ret[flat_key] = self._graph_fn_get_deterministic_action_wo_distribution(
                        logits.flat_key_lookup(flat_key)
                    )
            else:
                if flat_key == "":
                    # Still wrapped as FlattenedDataOp.
                    if isinstance(parameters, FlattenedDataOp):
                        return self.distributions[flat_key].draw(parameters[flat_key], deterministic)
                    else:
                        return self.distributions[flat_key].draw(parameters, deterministic)

                actions = self.distributions[flat_key].draw(parameters.flat_key_lookup(flat_key), deterministic)

                # If a bounded space (Beta distribution output between 0.0 and 1.0) -> scale correctly.
                if self.bounded_action_space[flat_key] is True:
                    actions = actions * (self.action_space.high - self.action_space.low) + self.action_space.low

                ret[flat_key] = actions

        return ret

    @graph_fn(flatten_ops=True, split_ops=True)
    def _graph_fn_get_deterministic_action_wo_distribution(self, logits):
        """
        Use this function only for discrete action spaces to circumvent using a full-blown
        backend-specific distribution object (e.g. tf.distribution.Multinomial).

        Args:
            logits (SingleDataOp): Logits over which to pick the argmax (greedy action).

        Returns:
            SingleDataOp: The argmax over the last rank of the input logits.
        """
        if get_backend() == "tf":
            return tf.argmax(logits, axis=-1, output_type=tf.int32)
        elif get_backend() == "pytorch":
            return torch.argmax(logits, dim=-1).int()
