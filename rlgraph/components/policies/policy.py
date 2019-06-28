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

import numpy as np
from rlgraph import get_backend
from rlgraph.components.action_adapters.action_adapter import ActionAdapter
from rlgraph.components.action_adapters.action_adapter_utils import get_action_adapter_type_from_distribution_type, \
    get_distribution_spec_from_action_adapter
from rlgraph.components.component import Component
from rlgraph.components.distributions import Distribution
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.spaces import Space, BoolBox, IntBox, ContainerSpace
from rlgraph.spaces.space_utils import get_default_distribution_from_space
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.ops import FlattenedDataOp, flat_key_lookup
from rlgraph.utils.rlgraph_errors import RLGraphError, RLGraphObsoletedError

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class Policy(Component):
    """
    A Policy is a wrapper Component that contains a NeuralNetwork, an ActionAdapter and a Distribution Component.
    """
    def __init__(self, network_spec, action_space=None, action_adapter_spec=None,
                 deterministic=True, scope="policy", distributions_spec=None, **kwargs):
        """
        Args:
            network_spec (Union[NeuralNetwork,dict]): The NeuralNetwork Component or a specification dict to build
                one.

            action_space (Union[dict,Space]): A specification dict to create the Space within which this Component
                will create actions or the action Space object directly.

            action_adapter_spec (Optional[dict]): A spec-dict to create an ActionAdapter. Use None for the default
                ActionAdapter object.

            deterministic (bool): Whether to pick actions according to the max-likelihood value or via sampling.
                Default: True.

            distributions_spec (dict): Specifies bounded and discrete distribution types, and optionally additional
                configuration parameters such as temperature.

            batch_apply (bool): Whether to wrap both the NN and the ActionAdapter with a BatchApply Component in order
                to fold time rank into batch rank before a forward pass.
        """
        super(Policy, self).__init__(scope=scope, **kwargs)

        self.neural_network = NeuralNetwork.from_spec(network_spec)  # type: NeuralNetwork
        self.deterministic = deterministic
        self.action_adapters = {}
        self.distributions = {}

        self.distributions_spec = distributions_spec if distributions_spec is not None else {}
        self.bounded_distribution_type = self.distributions_spec.get("bounded_distribution_type", "beta")
        self.discrete_distribution_type = self.distributions_spec.get("discrete_distribution_type", "categorical")
        # For discrete approximations.
        self.gumbel_softmax_temperature = self.distributions_spec.get("gumbel_softmax_temperature", 1.0)

        self.action_space = None
        self.flat_action_space = None
        self._create_action_adapters_and_distributions(
            action_space=action_space, action_adapter_spec=action_adapter_spec
        )

        self.add_components(
            *[self.neural_network] + list(self.action_adapters.values()) + list(self.distributions.values())
        )

    def _create_action_adapters_and_distributions(self, action_space, action_adapter_spec):
        if action_space is None:
            adapter = ActionAdapter.from_spec(action_adapter_spec)
            self.action_space = adapter.action_space
            # Assert single component action space.
            assert len(self.action_space.flatten()) == 1, \
                "ERROR: Action space must not be ContainerSpace if no `action_space` is given in Policy constructor!"
        else:
            self.action_space = Space.from_spec(action_space)

        self.flat_action_space = self.action_space.flatten()

        # Figure out our Distributions.
        for i, (flat_key, action_component) in enumerate(self.flat_action_space.items()):
            # Spec dict.
            if isinstance(action_adapter_spec, dict):
                aa_spec = flat_key_lookup(action_adapter_spec, flat_key, action_adapter_spec)
                aa_spec["action_space"] = action_component
            # Simple type spec.
            elif not isinstance(action_adapter_spec, ActionAdapter):
                aa_spec = dict(action_space=action_component)
            # Direct object.
            else:
                aa_spec = action_adapter_spec

            if isinstance(aa_spec, dict) and "type" not in aa_spec:
                dist_spec = get_default_distribution_from_space(
                    action_component, self.bounded_distribution_type, self.discrete_distribution_type,
                    self.gumbel_softmax_temperature
                )

                self.distributions[flat_key] = Distribution.from_spec(
                    dist_spec, scope="{}-{}".format(dist_spec["type"], i)
                )
                if self.distributions[flat_key] is None:
                    raise RLGraphError(
                        "ERROR: `action_component` is of type {} and not allowed in {} Component!".
                            format(type(action_space).__name__, self.name)
                    )
                aa_spec["type"] = get_action_adapter_type_from_distribution_type(
                    type(self.distributions[flat_key]).__name__
                )
                self.action_adapters[flat_key] = ActionAdapter.from_spec(aa_spec, scope="action-adapter-{}".format(i))
            else:
                self.action_adapters[flat_key] = ActionAdapter.from_spec(aa_spec, scope="action-adapter-{}".format(i))
                dist_spec = get_distribution_spec_from_action_adapter(self.action_adapters[flat_key])
                self.distributions[flat_key] = Distribution.from_spec(
                    dist_spec, scope="{}-{}".format(dist_spec["type"], i)
                )

    # Define our interface.
    @rlgraph_api
    def get_nn_outputs(self, nn_inputs):
        """
        Args:
            nn_inputs (any): The inputs to our neural network.

        Returns:
            any: The raw outputs of the neural network (before it's cleaned-up and passed through the ActionAdapter).
        """
        return self.neural_network.call(nn_inputs)

    @rlgraph_api
    def get_adapter_outputs(self, nn_inputs):
        """
        Args:
            nn_inputs (any): The inputs to our neural network.

        Returns:
            any: The (reshaped) outputs of the action layer of the ActionAdapter.
        """
        nn_outputs = self.get_nn_outputs(nn_inputs)
        nn_main_outputs = nn_outputs
        if self.neural_network.num_outputs > 1:
            nn_main_outputs = nn_outputs[0]
        action_layer_outputs = self._graph_fn_get_adapter_outputs(nn_main_outputs)
        # Add last internal states to return value.
        return dict(adapter_outputs=action_layer_outputs, nn_outputs=nn_outputs)

    @rlgraph_api
    def get_adapter_outputs_and_parameters(self, nn_inputs):
        """
        Args:
            nn_inputs (any): The input to our neural network.
            #internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                logits: The (reshaped) logits from the ActionAdapter.
                parameters: The parameters for the distribution.
                log_probs (Optional): The log(probabilities) values for discrete distributions.
        """
        nn_outputs = self.get_nn_outputs(nn_inputs)
        nn_main_outputs = nn_outputs
        if self.neural_network.num_outputs > 1:
            nn_main_outputs = nn_outputs[0]
        out = self._graph_fn_get_adapter_outputs_and_parameters(nn_main_outputs)
        return dict(
            nn_outputs=nn_outputs, adapter_outputs=out[0], parameters=out[1],
            action_probabilities=out[2], log_probs=out[3]
        )

    @rlgraph_api
    def get_action(self, nn_inputs, deterministic=None):  # other_nn_inputs=None,
        """
        Returns an action based on NN output, action adapter output and distribution sampling.

        Args:
            nn_inputs (any): The input to our neural network.
            #other_nn_inputs (DataOp): Inputs to the NN that don't have to be pushed through the preprocessor.
            deterministic (Optional[bool]): If not None, use this to determine whether actions should be drawn
                from the distribution in max-likelihood (deterministic) or stochastic fashion.

        Returns:
            dict:
                `action`: The drawn action.
                #`last_internal_states`: The last internal states (if NN is RNN based, otherwise: None).
        """
        deterministic = self.deterministic if deterministic is None else deterministic

        out = self.get_adapter_outputs_and_parameters(nn_inputs)  #, other_nn_inputs)
        action = self._graph_fn_get_action_components(out["adapter_outputs"], out["parameters"], deterministic)

        return dict(
            action=action,
            nn_outputs=out["nn_outputs"],
            adapter_outputs=out["adapter_outputs"],
            parameters=out["parameters"],
            log_probs=out.get("log_probs"),
            action_probabilities=out.get("action_probabilities")
        )

    @rlgraph_api
    def get_action_and_log_likelihood(self, nn_inputs, deterministic=None):
        """
        Args:
            nn_inputs (any): The input to our neural network.
            #internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.
            deterministic (Optional[bool]): If not None, use this to determine whether actions should be drawn
                from the distribution in max-likelihood (deterministic) or stochastic fashion.
        Returns:
            dict:
                `nn_outputs`: The raw output of the neural network.
                `adapter_outputs`: The (reshaped) raw action adapter output.
                `action`: The drawn action.
                `log_likelihood`: The log probability/log likelihood of the drawn action.
        """
        deterministic = self.deterministic if deterministic is None else deterministic
        out = self.get_adapter_outputs_and_parameters(nn_inputs)
        action, log_likelihood = self._graph_fn_get_action_and_log_likelihood(out["parameters"], deterministic)
        log_likelihood = self._graph_fn_combine_log_likelihood_over_container_keys(log_likelihood)

        return dict(
            nn_outputs=out["nn_outputs"],
            adapter_outputs=out["adapter_outputs"],
            action=action,
            # log-llh of the drawn action.
            log_likelihood=log_likelihood,
            # All action probabilities (discrete case).
            action_probabilities=out.get("action_probabilities")
        )

    @rlgraph_api(must_be_complete=False)
    def get_log_likelihood(self, nn_inputs, actions):
        """
        Computes the log-likelihood for a given set of actions under the distribution induced by a set of states.

        Args:
            nn_inputs (any): The input to our neural network.
            actions (any): The actions for which to get the log-likelihood.

        Returns:
            Log-probs of actions under current policy
        """
        out = self.get_adapter_outputs_and_parameters(nn_inputs)

        # Probabilities under current action.
        log_likelihood = self._graph_fn_get_distribution_log_likelihood(out["parameters"], actions)
        log_likelihood = self._graph_fn_combine_log_likelihood_over_container_keys(log_likelihood)

        return dict(log_likelihood=log_likelihood, adapter_outputs=out["adapter_outputs"])

    @rlgraph_api
    def get_deterministic_action(self, nn_inputs):
        """
        Args:
            nn_inputs (any): The input to our neural network.
            #internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: See `get_action`, but with deterministic force set to True.
        """
        out = self.get_adapter_outputs_and_parameters(nn_inputs)
        action = self._graph_fn_get_action_components(out["adapter_outputs"], out["parameters"], True)

        return dict(action=action, nn_outputs=out["nn_outputs"])

    @rlgraph_api
    def get_stochastic_action(self, nn_inputs): #, internal_states=None):
        """
        Args:
            nn_inputs (any): The input to our neural network.
            #internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: See `get_action`, but with deterministic force set to False.
        """
        out = self.get_adapter_outputs_and_parameters(nn_inputs)  # internal_states
        action = self._graph_fn_get_action_components(out["adapter_outputs"], out["parameters"], False)

        return dict(action=action, nn_outputs=out["nn_outputs"])

    @rlgraph_api
    def get_entropy(self, nn_inputs):
        """
        Args:
            nn_inputs (any): The inputs to our neural network.
            #internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: See Distribution component.
        """
        out = self.get_adapter_outputs_and_parameters(nn_inputs)  #, internal_states)
        entropy = self._graph_fn_get_distribution_entropies(out["parameters"])

        return dict(entropy=entropy, nn_outputs=out["nn_outputs"])

    @graph_fn(flatten_ops=True, split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_get_adapter_outputs(self, flat_key, nn_outputs):  #, nn_inputs):
        """
        Pushes the given nn_output through all our action adapters and returns a DataOpDict with the keys corresponding
        to our `action_space`.

        Args:
            nn_outputs (DataOp): The output of our neural network.
            #nn_inputs (DataOp): The original inputs of the NN (that produced the `nn_outputs`).

        Returns:
            FlattenedDataOp: A DataOpDict with the different action adapter outputs (keys correspond to
                structure of `self.action_space`).
        """
        # NN outputs are already split -> Feed flat-key NN output directly into its corresponding action_adapter.
        if flat_key in self.action_adapters:
            return self.action_adapters[flat_key].call(nn_outputs)
        # Many NN outputs, but no action adapters specified for this one -> return nn_outputs as is.
        elif flat_key != "":
            return nn_outputs, nn_outputs, None

        ret = FlattenedDataOp()
        for aa_flat_key, action_adapter in self.action_adapters.items():
            ret[aa_flat_key] = action_adapter.call(nn_outputs)

        return ret

    @graph_fn(flatten_ops=True, split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_get_adapter_outputs_and_parameters(self, flat_key, nn_outputs):
        """
        Pushes the given nn_output through all our action adapters' get_logits_parameters_log_probs API's and
        returns a DataOpDict with the keys corresponding to our `action_space`.

        Args:
            nn_outputs (DataOp): The output of our neural network.
            #nn_inputs (DataOp): The original inputs of the NN (that produced the `nn_outputs`).

        Returns:
            tuple:
                - FlattenedDataOp: A DataOpDict with the different action adapters' logits outputs.
                - FlattenedDataOp: A DataOpDict with the different action adapters' parameters outputs.
                - FlattenedDataOp: A DataOpDict with the different action adapters' log_probs outputs.
            Note: Keys always correspond to structure of `self.action_space`.
        """
        # NN outputs are already split -> Feed flat-key NN output directly into its corresponding action_adapter.
        if flat_key in self.action_adapters:
            adapter_outs = self.action_adapters[flat_key].call(nn_outputs)
            params = self.action_adapters[flat_key].get_parameters_from_adapter_outputs(adapter_outs)
            return adapter_outs, params["parameters"], params.get("probabilities"), params.get("log_probs")
        # Many NN outputs, but no action adapters specified for this one -> return nn_outputs as is.
        elif flat_key != "":
            return nn_outputs, nn_outputs, None, None

        # There is only a single NN-output, but many action adapters.
        adapter_outputs = FlattenedDataOp()
        parameters = FlattenedDataOp()
        probs = FlattenedDataOp()
        log_probs = FlattenedDataOp()
        for aa_flat_key, action_adapter in self.action_adapters.items():
            adapter_outs = action_adapter.call(nn_outputs)
            params = action_adapter.get_parameters_from_adapter_outputs(adapter_outs)
            #out = action_adapter.get_adapter_outputs_and_parameters(nn_outputs, nn_inputs)
            adapter_outputs[aa_flat_key], parameters[aa_flat_key], probs[aa_flat_key], log_probs[aa_flat_key] = \
                adapter_outs, params["parameters"], params.get("probabilities"), params.get("log_probs")

        return adapter_outputs, parameters, probs, log_probs

    @graph_fn(flatten_ops="flat_action_space", split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_get_distribution_entropies(self, flat_key, parameters):
        """
        Pushes `parameters` through the respective self.distributions' `entropy` API-methods and returns a
        DataOp with the entropy values.

        Args:
            parameters (DataOp): The parameters to define a distribution. This could be a ContainerDataOp, which
                container the parameter pieces for each action component.

        Returns:
            SingleDataOp: The DataOp with the `entropy` outputs for the given flat_key distribution.
        """
        return self.distributions[flat_key].entropy(parameters)

    @graph_fn(flatten_ops="flat_action_space", split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_get_distribution_log_likelihood(self, flat_key, parameters, actions):
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
        return self.distributions[flat_key].log_prob(parameters, actions)

    @graph_fn(flatten_ops=True)
    def _graph_fn_combine_log_likelihood_over_container_keys(self, log_likelihoods):
        """
        If action space is a container space, add log-likelihoods (assuming all distribution components are
        independent).
        """
        if isinstance(self.action_space, ContainerSpace):
            if get_backend() == "tf":
                log_likelihoods = tf.stack(list(log_likelihoods.values()))
                log_likelihoods = tf.reduce_sum(log_likelihoods, axis=0)
            elif get_backend() == "pytorch":
                log_likelihoods = torch.stack(list(log_likelihoods.values()))
                log_likelihoods = torch.sum(log_likelihoods, 0)

        return log_likelihoods

    @graph_fn(flatten_ops="flat_action_space", split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_get_action_components(self, flat_key, logits, parameters, deterministic):
        action_space_component = self.flat_action_space[flat_key]

        # Skip our distribution, iff discrete action-space and deterministic acting (greedy).
        # In that case, one does not need to create a distribution in the graph each act (only to get the argmax
        # over the logits, which is the same as the argmax over the probabilities (or log-probabilities)).
        if isinstance(action_space_component, IntBox) and \
                (deterministic is True or (isinstance(deterministic, np.ndarray) and deterministic)):
            return self._graph_fn_get_deterministic_action_wo_distribution(logits)
        # Bernoulli: Sigmoid derived p must be larger 0.5.
        elif isinstance(action_space_component, BoolBox) and \
                (deterministic is True or (isinstance(deterministic, np.ndarray) and deterministic)):
            # Note: Change 0.5 to 1.0, once parameters are logits, not probs anymore (so far, parameters for
            # Bernoulli distributions are still probs).
            if get_backend() == "tf":
                return tf.greater(parameters, 0.5)
            elif get_backend() == "pytorch":
                return torch.gt(parameters, 0.5)
        # Deterministic is tensor or False. Pass through graph.
        else:
            return self.distributions[flat_key].draw(parameters, deterministic)

    @graph_fn(returns=2, flatten_ops="flat_action_space", split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_get_action_and_log_likelihood(self, flat_key, parameters, deterministic):
        # TODO: Utilize same logic in _graph_fn_get_action_components.
        # TODO: Not working right now, because we would split twice (here and in _graph_fn_get_action_components).
        action = None
        log_prob_or_likelihood = None

        action_space_component = self.flat_action_space[flat_key]

        # Categorical: Argmax over raw logits.
        if isinstance(action_space_component, IntBox) and \
                (deterministic is True or (isinstance(deterministic, np.ndarray) and deterministic)):
            action = self._graph_fn_get_deterministic_action_wo_distribution(parameters)
            if get_backend() == "tf":
                log_prob_or_likelihood = tf.log(tf.reduce_max(tf.nn.softmax(parameters, axis=-1), axis=-1))
            elif get_backend() == "pytorch":
                log_prob_or_likelihood = torch.log(torch.max(torch.softmax(parameters, dim=-1), dim=-1)[0])
        # Bernoulli: Sigmoid derived p must be larger 0.5.
        elif isinstance(action_space_component, BoolBox) and \
                (deterministic is True or (isinstance(deterministic, np.ndarray) and deterministic)):
            # Note: Change 0.5 to 1.0, once parameters are logits, not probs anymore (so far, parameters for
            # Bernoulli distributions are still probs).
            if get_backend() == "tf":
                action = tf.greater(parameters, 0.5)
                log_prob_or_likelihood = tf.log(tf.where(parameters > 0.5, parameters, 1.0 - parameters))
            elif get_backend() == "pytorch":
                action = torch.gt(parameters, 0.5)
                log_prob_or_likelihood = torch.log(torch.where(parameters > 0.5, parameters, 1.0 - parameters))
        # Deterministic is tensor or False. Pass through graph.
        else:
            action, log_prob_or_likelihood = self.distributions[flat_key].sample_and_log_prob(
                parameters, deterministic
            )

        return action, log_prob_or_likelihood

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

    def get_logits_parameters_log_probs(self, nn_inputs, internal_states=None):
        raise RLGraphObsoletedError("API-method", "get_logits_parameters_log_probs",
                                    "get_adapter_outputs_and_parameters")

    def get_logits_probabilities_log_probs(self, nn_inputs, internal_states=None):
        raise RLGraphObsoletedError("API-method", "get_logits_probabilities_log_probs",
                                    "get_adapter_outputs_and_parameters")

    def get_action_and_log_params(self, nn_inputs, internal_states=None, deterministic=None):
        raise RLGraphObsoletedError("API-method", "get_action_and_log_params", "get_action_and_log_likelihood")

    def get_action_log_probs(self, nn_inputs, actions):
        raise RLGraphObsoletedError("API-method", "get_action_log_probs", "get_log_likelihood")

    def get_action_layer_output(self, nn_inputs):
        raise RLGraphObsoletedError("API-method", "get_action_layer_output", "get_adapter_outputs")

    def get_nn_output(self, nn_inputs):
        raise RLGraphObsoletedError("API-method", "get_nn_output", "get_nn_outputs")
