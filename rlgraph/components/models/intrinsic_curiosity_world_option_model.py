# Copyright 2018/2019 ducandu GmbH. All Rights Reserved.
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

from rlgraph.components.policies.supervised_predictor import SupervisedPredictor

from rlgraph import get_backend
from rlgraph.components.layers.nn.concat_layer import ConcatLayer
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.loss_functions.neg_log_likelihood_loss import NegativeLogLikelihoodLoss
from rlgraph.components.models.supervised_model import SupervisedModel
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.spaces import *
from rlgraph.spaces.space_utils import get_default_distribution_from_space
from rlgraph.utils.decorators import rlgraph_api, graph_fn

if get_backend() == "tf":
    import tensorflow as tf


class IntrinsicCuriosityWorldOptionModel(SupervisedModel):
    """
    Combines an inverse dynamics network, predicting actions that lead from state s to state s', with a forward
    model ("world option model"), predicting a distribution over a latent feature-vector for s', when given s and a.
    Uses a single loss function, combining the loss terms for both these prediction tasks, and one optimizer.

    Codename: LEM (Latent Space Exploration/Establishing Model).

    Based on:
    [1] Curiosity-driven Exploration by Self-supervised Prediction - Pathak et al. - UC Berkeley 2017
    [2] World Models - Ha, Schmidhuber - 2018
    """
    def __init__(
            self, action_space, world_option_model_network, encoder_network, num_features, num_mixtures, beta=0.2,
            post_phi_concat_network=None,
            reward_clipping=1.0,
            intrinsic_rewards_weight=0.1,
            concat_with_command_vector=False,
            optimizer=None, deterministic=False, scope="intrinsic-curiosity-world-option-model",
            **kwargs
    ):
        """
        Args:
            action_space (Space): The action Space to be fed into the model together with the latent feature vector
                for the states. Will be flattened automatically and then concatenated by this component.

            world_option_model_network (Union[NeuralNetwork,dict]): A specification dict (or NN object directly) to
                construct the world-option-model's neural network.

            encoder_network (Union[NeuralNetwork,dict]): A specification dict (or NN object directly) to
                construct the inverse dynamics model's encoder network leading from s to phi (feature vector).

            num_features (int): The size of the feature vectors phi.

            num_mixtures (int): The number of mixture Normals to use for the next-state distribution output.

            beta (float): The weight for the phi' loss (action loss is then 1.0 - beta).

            post_phi_concat_network

            reward_clipping (float): 0.0 for no clipping, some other value for +/- reward value clipping.
                Default: 1.0.

            concat_with_command_vector (bool): If True, this model needs an additional command vector (coming from the
                policy above) to concat it together with the latent state vector.

            optimizer (Optional[Optimizer]): The optimizer to use for supervised learning of the two networks
                (ICM and WOM).
        """
        self.num_features = num_features
        self.num_mixtures = num_mixtures
        self.deterministic = deterministic
        self.beta = beta
        assert 0.0 < self.beta < 1.0, "ERROR: `beta` must be between 0 and 1!"
        self.reward_clipping = reward_clipping
        self.intrinsic_rewards_weight = intrinsic_rewards_weight

        # Create the encoder network inside a SupervisedPredictor (so we get the adapter + distribution with it).
        self.state_encoder = SupervisedPredictor(
            network_spec=encoder_network, output_space=FloatBox(shape=(num_features,), add_batch_rank=True),
            scope="state-encoder"
        )

        # Create the container loss function for the two prediction tasks:
        # a) Action prediction and b) next-state prediction, each of them using a simple neg log likelihood loss
        # comparing the actual action and s' with their log-likelihood value vs the respective distributions.
        self.loss_functions = dict(
            # Action prediction loss (neg log likelihood of observed action vs the parameterized distribution).
            predicted_actions=NegativeLogLikelihoodLoss(
                distribution_spec=get_default_distribution_from_space(action_space),
                scope="action-loss"
            ),
            # s' prediction loss (neg log likelihood of observed s' vs the parameterized mixed normal distribution).
            predicted_phi_=NegativeLogLikelihoodLoss(distribution_spec=dict(type="mixture", _args=[
                "multi-variate-normal" for _ in range(num_mixtures)
            ]), scope="phi-loss")
        )

        # TODO: Support for command vector concatenation.
        #self.concat_with_command_vector = concat_with_command_vector

        # Define the Model's network's custom call method.
        def custom_call(self, inputs):
            phi = inputs["phi"]
            actions = inputs["actions"]
            phi_ = inputs["phi_"]
            actions_flat = self.get_sub_component_by_name("action-flattener").call(actions)
            concat_phis = self.get_sub_component_by_name("concat-phis").call(phi, phi_)
            # Predict the action that lead from s to s'.
            predicted_actions = self.get_sub_component_by_name("post-phi-concat-nn").call(concat_phis)

            # Concat phi with flattened actions.
            phi_and_actions = self.get_sub_component_by_name("concat-states-and-actions").call(
                phi, actions_flat
            )
            # Add stop-gradient to phi here before predicting phi'
            # (the phis should only be trained by the inverse dynamics model, not by the world option model).
            # NOT DONE IN ORIGINAL PAPER's CODE AND ALSO NOT IN MLAGENTS EQUIVALENT.
            # phi_and_actions = self.get_sub_component_by_name("stop-gradient").stop(phi_and_actions)
            # Predict phi' (through a mixture gaussian distribution).
            predicted_phi_ = self.get_sub_component_by_name("wom-nn").call(phi_and_actions)

            return dict(
                # Predictions (actions and next-state-features (mixture distribution)).
                predicted_actions=predicted_actions,
                predicted_phi_=predicted_phi_
                ## Also return the two feature vectors for s and s'.
                #phi=phi, phi_=phi_
            )

        # Create the SupervisedPredictor's neural network.
        predictor_network = NeuralNetwork(
            # The world option model network taking action-cat-phi and mapping them to the predicted phi'.
            NeuralNetwork.from_spec(world_option_model_network, scope="wom-nn"),
            # The concat component concatenating both latent state vectors (phi and phi').
            ConcatLayer(scope="concat-phis"),
            # The NN mapping from phi-cat-phi' to the action prediction.
            NeuralNetwork.from_spec(post_phi_concat_network, scope="post-phi-concat-nn"),
            # The ReShape component for flattening all actions in arbitrary action spaces.
            ReShape(flatten=True, flatten_categories=True, flatten_containers=True, scope="action-flattener"),
            # The concat component concatenating latent state feature vector and incoming (flattened) actions.
            ConcatLayer(scope="concat-states-and-actions"),
            # Set the `call` method.
            api_methods={("call", custom_call)}
        )

        if optimizer is None:
            optimizer = dict(type="adam", learning_rate=3e-4)

        super(IntrinsicCuriosityWorldOptionModel, self).__init__(
            predictor=dict(
                network_spec=predictor_network,
                output_space=Dict({
                    "predicted_actions": action_space,
                    "predicted_phi_": FloatBox(shape=(self.num_features,))
                }, add_batch_rank=action_space.has_batch_rank, add_time_rank=action_space.has_time_rank),
                distribution_adapter_spec=dict(
                    # for `predicted_actions`: use default adapter
                    # for predicted_phi': use normal-mixture adapter & distribution.
                    predicted_phi_={"type": "normal-mixture-adapter", "num_mixtures": num_mixtures}
                ),
                deterministic=deterministic
            ),
            loss_function=self.loss_functions["predicted_actions"],
            optimizer=optimizer, scope=scope, **kwargs
        )

        self.add_components(self.state_encoder, self.loss_functions["predicted_phi_"])

    @rlgraph_api
    def get_phi(self, states, deterministic=None):
        """
        Returns the (automatically learnt) feature vector given some state (s).

        Args:
            states (DataOpRec): The states to encode to phi (feature vector).
            deterministic (DataOpRec[bool]): Whether to sample from the distribution output of the encoder network
                deterministically (max-likelihood) or not.

        Returns:
            dict:
                - phi: The feature vector for s.
        """
        deterministic = self.deterministic if deterministic is None else deterministic
        phi = self.state_encoder.predict(states, deterministic=deterministic)
        phi["phi"] = phi["predictions"]
        return phi

    @rlgraph_api
    def get_phis_from_nn_inputs(self, nn_inputs, deterministic=None):
        """
        Returns the (automatically learnt) feature vectors given some state (s) and next state (s').
        Action inputs within `nn_inputs` are just passed through.

        Args:
            nn_inputs (DataOpRec[dict]): The NN input (as dict) with the keys: "states", "next_states", and "actions".

            deterministic (DataOpRec[bool]): Whether to sample from the distribution output of the encoder network
                deterministically (max-likelihood) or not.

        Returns:
            dict:
                - phi: The feature vector for s.
                - phi_: The feature vector for s'.
                - actions: The actions from `nn_inputs` unchanged.
        """
        states = nn_inputs["states"]
        next_states = nn_inputs["next_states"]
        actions = nn_inputs["actions"]
        phi = self.get_phi(states, deterministic=deterministic)["predictions"]
        phi_ = self.get_phi(next_states, deterministic=deterministic)["predictions"]
        return dict(phi=phi, actions=actions, phi_=phi_)

    @rlgraph_api
    def predict(self, nn_inputs, deterministic=None):
        """
        Returns:

        """
        nn_inputs = self.get_phis_from_nn_inputs(nn_inputs, deterministic)
        return self.predictor.predict(nn_inputs, deterministic=deterministic)

    @rlgraph_api
    def get_distribution_parameters(self, nn_inputs, deterministic=None):
        nn_inputs = self.get_phis_from_nn_inputs(nn_inputs, deterministic=deterministic)
        return self.predictor.get_distribution_parameters(nn_inputs)

    @rlgraph_api
    def update(self, nn_inputs, labels=None, time_percentage=None):
        """
        Returns:
            dict:
                see SupervisedModel outputs
                - intrinsic_rewards: The intrinsic rewards
        """
        # Update the model (w/o labels b/c labels are already included in the nn_inputs).
        assert labels is None, "ERROR: `labels` arg not needed for {}.`update()`!".format(type(self).__name__)
        nn_inputs = self.get_phis_from_nn_inputs(nn_inputs, deterministic=False)
        parameters = self.predictor.get_distribution_parameters(nn_inputs)
        # Construct labels from nn_inputs.
        labels = dict(predicted_actions=nn_inputs["actions"], predicted_phi_=nn_inputs["phi_"])

        # Get two losses from both parts of the loss function.
        action_loss, action_loss_per_item = self.loss_functions["predicted_actions"].loss(
            parameters["predicted_actions"], labels["predicted_actions"], time_percentage
        )
        phi_loss, phi_loss_per_item = self.loss_functions["predicted_phi_"].loss(
            parameters["predicted_phi_"], labels["predicted_phi_"], time_percentage
        )
        # Average all the losses.
        loss, loss_per_item, intrinsic_rewards = self._graph_fn_average_losses_and_calc_rewards(
            action_loss, action_loss_per_item, phi_loss, phi_loss_per_item
        )
        step_op = self.optimizer.step(self.variables(), loss, loss_per_item, time_percentage)

        return dict(
            step_op=step_op,
            loss=loss,
            loss_per_item=loss_per_item,
            intrinsic_rewards=intrinsic_rewards,
            parameters=parameters
        )

    # TODO: Move this to Component base class.
    @graph_fn
    def _graph_fn_stop_gradient(self, input_):
        if get_backend() == "tf":
            return tf.stop_gradient(input_)

    @graph_fn
    def _graph_fn_average_losses_and_calc_rewards(self, action_loss, action_loss_per_item, phi_loss, phi_loss_per_item):
        loss = (1.0 - self.beta) * action_loss + self.beta * phi_loss
        loss_per_item = (1.0 - self.beta) * action_loss_per_item + self.beta * phi_loss_per_item
        intrinsic_rewards = None
        if get_backend() == "tf":
            if self.reward_clipping > 0.0:
                intrinsic_rewards = tf.clip_by_value(
                    phi_loss_per_item, clip_value_min=-self.reward_clipping, clip_value_max=self.reward_clipping
                )
            else:
                intrinsic_rewards = phi_loss_per_item
            intrinsic_rewards *= self.intrinsic_rewards_weight
        return loss, loss_per_item, intrinsic_rewards

