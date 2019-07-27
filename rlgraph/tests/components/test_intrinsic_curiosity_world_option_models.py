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

from __future__ import absolute_import, division, print_function

from collections import Counter
import unittest

import numpy as np

from rlgraph.components.models.intrinsic_curiosity_world_option_model import IntrinsicCuriosityWorldOptionModel
from rlgraph.components.models.supervised_model import SupervisedModel
from rlgraph.environments.grid_world import GridWorld
from rlgraph.spaces import FloatBox, IntBox, Dict
from rlgraph.tests.component_test import ComponentTest
from rlgraph.utils.numpy import dense_layer, one_hot, softmax
from rlgraph.tests.test_util import recursive_assert_almost_equal


class TestIntrinsicCuriosityWorldOptionModels(unittest.TestCase):
    """
    Tests for the different Model classes.
    """
    def test_intrinsic_curiosity_world_option_model_deterministic(self):
        num_mixtures = 2
        num_features = 3

        state_space = FloatBox(shape=(5,), add_batch_rank=True)
        #state_feature_space = FloatBox(shape=(num_features,), add_batch_rank=True)
        action_space = IntBox(3, add_batch_rank=True)

        # Set the input dict for the Model's NN.
        nn_inputs_space = Dict(
            {"actions": action_space, "states": state_space, "next_states": state_space},
            add_batch_rank=True
        )
        # Set the labels Space (we are outputting actions and ).
        #labels_space = Dict({"predicted_actions": action_space, "predicted_phi_": state_feature_space},
        #                    add_batch_rank=True)

        icwom = IntrinsicCuriosityWorldOptionModel(action_space=action_space, world_option_model_network=[
            {"type": "dense", "units": 5, "activation": "linear"}
        ], encoder_network=[{"type": "dense", "units": 4, "activation": "linear"}],
            num_features=num_features,
            post_phi_concat_network=[{"type": "dense", "units": 2, "activation": "linear"}],
            num_mixtures=num_mixtures, optimizer={"type": "adam", "learning_rate": 3e-4}, deterministic=True)

        test = ComponentTest(component=icwom, input_spaces=dict(
            nn_inputs=nn_inputs_space,
            states=state_space
        ))

        variables = test.read_variable_values(icwom.variable_registry)
        encoder_nn_scope = "intrinsic-curiosity-world-option-model/state-encoder/policy/neural-network/"
        encoder_aa_scope = "intrinsic-curiosity-world-option-model/state-encoder/policy/action-adapter-0/action-network/action-layer/dense/"
        nn_scope = "intrinsic-curiosity-world-option-model/supervised-predictor/policy/neural-network/"
        aa_scope = "intrinsic-curiosity-world-option-model/supervised-predictor/policy/action-adapter-"

        # Test the `predict` API. Expect an action and a latent-space next-state output.
        nn_inputs = nn_inputs_space.sample(5)

        # Deterministic prediction (pick max-likelihood action and next-state feature vector).
        state_features_params = dense_layer(
            dense_layer(
                nn_inputs["states"], variables[encoder_nn_scope+"dense-layer/dense/kernel"],
                variables[encoder_nn_scope+"dense-layer/dense/bias"]
            ), variables[encoder_aa_scope+"kernel"], variables[encoder_aa_scope+"bias"]
        )
        next_state_features_params = dense_layer(
            dense_layer(
                nn_inputs["next_states"], variables[encoder_nn_scope+"dense-layer/dense/kernel"],
                variables[encoder_nn_scope+"dense-layer/dense/bias"]
            ), variables[encoder_aa_scope+"kernel"], variables[encoder_aa_scope+"bias"]
        )
        # Model is deterministic -> Get the mean for the state-feature vectors.
        state_features_mean = state_features_params[:, :num_features]  # mean values
        next_state_features_mean = next_state_features_params[:, :num_features]  # mean values

        concat_features = np.concatenate([state_features_mean, next_state_features_mean], axis=-1)

        action_parameters = dense_layer(dense_layer(
            concat_features, variables[nn_scope+"post-phi-concat-nn/dense-layer/dense/kernel"],
            variables[nn_scope+"post-phi-concat-nn/dense-layer/dense/bias"]
        ), variables[aa_scope+"0/action-network/action-layer/dense/kernel"],
            variables[aa_scope+"0/action-network/action-layer/dense/bias"])
        expected_actions = np.argmax(action_parameters, axis=-1)

        # Calculate next-state prediction.
        actions_flat = one_hot(nn_inputs["actions"], depth=action_space.num_categories)
        concat_state_features_and_actions = np.concatenate([state_features_mean, actions_flat], axis=-1)
        wom_output = dense_layer(concat_state_features_and_actions, variables[nn_scope+"wom-nn/dense-layer/dense/kernel"],
                                 variables[nn_scope+"wom-nn/dense-layer/dense/bias"])
        # Get the mixture Gaussian parameters.
        next_state_parameters = dense_layer(
            wom_output, variables[aa_scope + "1/action-network/action-layer/dense/kernel"],
            variables[aa_scope + "1/action-network/action-layer/dense/bias"]
        )
        next_state_parameters_softmaxed = np.copy(next_state_parameters)
        next_state_parameters_softmaxed[:, 0:num_mixtures] = softmax(next_state_parameters[:, 0:num_mixtures])
        expected_next_state_features = (
            next_state_parameters_softmaxed[:, 0:1] * next_state_parameters[:, num_mixtures:num_mixtures+num_features] + \
            next_state_parameters_softmaxed[:, 1:2] * next_state_parameters[:, num_mixtures+num_features:num_mixtures + num_features * 2]
        )

        # Test for 10x the same results that match what we think it should be (deterministic forward pass).
        for _ in range(10):
            out = test.test(("get_distribution_parameters", nn_inputs))
            params = out["parameters"]
            # Check all distribution parameters.
            recursive_assert_almost_equal(action_parameters, params["predicted_actions"])
            recursive_assert_almost_equal(next_state_parameters[:, 0:num_mixtures], params["predicted_phi_"]["categorical"])
            recursive_assert_almost_equal(next_state_parameters[:, num_mixtures:num_mixtures+num_features], params["predicted_phi_"]["parameters0"][0])
            recursive_assert_almost_equal(next_state_parameters[:, num_mixtures+num_features:num_mixtures+2*num_features], params["predicted_phi_"]["parameters1"][0])
            # Check the actual deterministic sampling step.
            out = test.test(("predict", nn_inputs, ["predictions", "parameters", "adapter_outputs"]))
            recursive_assert_almost_equal(action_parameters, out["adapter_outputs"]["predicted_actions"])
            recursive_assert_almost_equal(next_state_parameters, out["adapter_outputs"]["predicted_phi_"], decimals=5)
            recursive_assert_almost_equal(expected_actions, out["predictions"]["predicted_actions"], decimals=5)
            recursive_assert_almost_equal(expected_next_state_features, out["predictions"]["predicted_phi_"], decimals=5)

        test.terminate()

    def test_intrinsic_curiosity_world_option_model_stochastic(self):
        num_mixtures = 3
        num_features = 2

        state_space = FloatBox(shape=(4,), add_batch_rank=True)
        #state_feature_space = FloatBox(shape=(num_features,), add_batch_rank=True)
        action_space = IntBox(2, add_batch_rank=True)

        # Set the input dict for the Model's NN.
        nn_inputs_space = Dict(
            {"actions": action_space, "states": state_space, "next_states": state_space},
            add_batch_rank=True
        )
        # Set the labels Space (we are outputting actions and ).
        #labels_space = Dict(
        #    {"predicted_actions": action_space, "predicted_phi_": state_feature_space}, add_batch_rank=True
        #)

        icwom = IntrinsicCuriosityWorldOptionModel(action_space=action_space, world_option_model_network=[
            {"type": "dense", "units": 4, "activation": "linear"}
        ], encoder_network=[{"type": "dense", "units": 3, "activation": "linear"}], num_features=num_features,
            post_phi_concat_network=None,
            num_mixtures=num_mixtures, optimizer={"type": "adam", "learning_rate": 3e-4}, deterministic=False)

        test = ComponentTest(component=icwom, input_spaces=dict(
            nn_inputs=nn_inputs_space, #labels=labels_space,
            states=state_space
        ))  #, build_kwargs={"optimizer": icwom.optimizer})

        variables = test.read_variable_values(icwom.variable_registry)
        encoder_nn_scope = "intrinsic-curiosity-world-option-model/state-encoder/policy/neural-network/"
        encoder_aa_scope = "intrinsic-curiosity-world-option-model/state-encoder/policy/action-adapter-0/action-network/action-layer/dense/"
        nn_scope = "intrinsic-curiosity-world-option-model/supervised-predictor/policy/neural-network/"
        aa_scope = "intrinsic-curiosity-world-option-model/supervised-predictor/policy/action-adapter-"

        # Test the `predict` API. Expect an action and a latent-space next-state output.
        nn_inputs = nn_inputs_space.sample(4)

        # Deterministic prediction (pick max-likelihood action and next-state feature vector).
        state_features_params = dense_layer(
            dense_layer(
                nn_inputs["states"], variables[encoder_nn_scope+"dense-layer/dense/kernel"],
                variables[encoder_nn_scope+"dense-layer/dense/bias"]
            ), variables[encoder_aa_scope+"kernel"], variables[encoder_aa_scope+"bias"]
        )
        next_state_features_params = dense_layer(
            dense_layer(
                nn_inputs["next_states"], variables[encoder_nn_scope+"dense-layer/dense/kernel"],
                variables[encoder_nn_scope+"dense-layer/dense/bias"]
            ), variables[encoder_aa_scope+"kernel"], variables[encoder_aa_scope+"bias"]
        )
        # Model is stochastic -> Get the mean for the state-feature vectors.
        state_features_mean = state_features_params[:, :num_features]  # mean values
        # max stddev (NN output is exp'd to get the actual stddev)
        state_features_std_max = np.max(np.exp(state_features_params[:, num_features:]))
        next_state_features_mean = next_state_features_params[:, :num_features]  # mean values
        next_state_features_std_max = np.max(np.exp(next_state_features_params[:, num_features:]))  # max stddev

        # Sample n times to get the mean for phi and phi'.
        phis = []
        phi_s = []
        for _ in range(1000):
            out = test.test(("get_phis_from_nn_inputs", nn_inputs))
            # Make sure values are always different (stochastic sampling).
            if len(phis) > 0:
                self.assertTrue((phis[-1] != out["phi"]).all())
                self.assertTrue((phi_s[-1] != out["phi_"]).all())
            # Store for later averaging.
            phis.append(out["phi"])
            phi_s.append(out["phi_"])

        recursive_assert_almost_equal(state_features_mean, np.mean(np.array(phis), axis=0), atol=state_features_std_max / 10)
        recursive_assert_almost_equal(next_state_features_mean, np.mean(np.array(phi_s), axis=0), atol=next_state_features_std_max / 10)

        concat_features = np.concatenate([state_features_mean, next_state_features_mean], axis=-1)

        action_parameters = softmax(dense_layer(
            concat_features,
            variables[aa_scope+"0/action-network/action-layer/dense/kernel"],  # 0=action prediction output
            variables[aa_scope+"0/action-network/action-layer/dense/bias"]
        ))
        expected_mean_actions = np.sum(action_parameters * np.array([0, 1]), axis=-1)

        # Calculate next-state prediction.
        actions_flat = one_hot(nn_inputs["actions"], depth=action_space.num_categories)
        concat_state_features_and_actions = np.concatenate([state_features_mean, actions_flat], axis=-1)
        wom_output = dense_layer(
            concat_state_features_and_actions,
            variables[nn_scope+"wom-nn/dense-layer/dense/kernel"],
            variables[nn_scope+"wom-nn/dense-layer/dense/bias"]
        )
        # Get the mixture Gaussian parameters.
        next_state_parameters = dense_layer(
            wom_output,
            variables[aa_scope + "1/action-network/action-layer/dense/kernel"],  # 1=phi' prediction output
            variables[aa_scope + "1/action-network/action-layer/dense/bias"]
        )
        next_state_parameters_softmaxed = np.copy(next_state_parameters)
        next_state_parameters_softmaxed[:, 0:num_mixtures] = softmax(next_state_parameters[:, 0:num_mixtures])

        expected_next_state_features = (
            next_state_parameters_softmaxed[:, 0:1] * next_state_parameters[:, num_mixtures:num_mixtures+num_features] + \
            next_state_parameters_softmaxed[:, 1:2] * next_state_parameters[:, num_mixtures+num_features:num_mixtures+num_features*2] + \
            next_state_parameters_softmaxed[:, 2:3] * next_state_parameters[:, num_mixtures+num_features*2:num_mixtures+num_features*3]
        )
        expected_next_state_features_std = (
            next_state_parameters_softmaxed[:, 0:1] * np.exp(next_state_parameters[:, num_mixtures+num_features*3:num_mixtures+num_features*4]) + \
            next_state_parameters_softmaxed[:, 1:2] * np.exp(next_state_parameters[:, num_mixtures+num_features*4:num_mixtures+num_features*5]) + \
            next_state_parameters_softmaxed[:, 2:3] * np.exp(next_state_parameters[:, num_mixtures+num_features*5:num_mixtures+num_features*6])
        )

        # Test for n times different results whose mean matches what we think it should be (stochastic forward passes).
        predicted_actions = []
        predicted_next_states = []
        for _ in range(10000):
            # Check the actual deterministic sampling step.
            out = test.test(("predict", nn_inputs, ["predictions"]))
            predicted_actions.append(out["predictions"]["predicted_actions"])
            predicted_next_states.append(out["predictions"]["predicted_phi_"])
        predicted_next_states_mean = np.mean(np.array(predicted_next_states), axis=0)

        recursive_assert_almost_equal(expected_mean_actions, np.mean(np.array(predicted_actions), axis=0), atol=0.1)
        # Test by single item as they all have different stddevs.
        recursive_assert_almost_equal(expected_next_state_features[0][0], predicted_next_states_mean[0][0], atol=expected_next_state_features_std[0][0] / 2)
        recursive_assert_almost_equal(expected_next_state_features[0][1], predicted_next_states_mean[0][1], atol=expected_next_state_features_std[0][1] / 2)
        recursive_assert_almost_equal(expected_next_state_features[1][0], predicted_next_states_mean[1][0], atol=expected_next_state_features_std[1][0] / 2)
        recursive_assert_almost_equal(expected_next_state_features[1][1], predicted_next_states_mean[1][1], atol=expected_next_state_features_std[1][1] / 2)
        recursive_assert_almost_equal(expected_next_state_features[2][0], predicted_next_states_mean[2][0], atol=expected_next_state_features_std[2][0] / 2)
        recursive_assert_almost_equal(expected_next_state_features[2][1], predicted_next_states_mean[2][1], atol=expected_next_state_features_std[2][1] / 2)
        recursive_assert_almost_equal(expected_next_state_features[3][0], predicted_next_states_mean[2][0], atol=expected_next_state_features_std[3][0] / 2)
        recursive_assert_almost_equal(expected_next_state_features[3][1], predicted_next_states_mean[2][1], atol=expected_next_state_features_std[3][1] / 2)

        test.terminate()

    def test_intrinsic_curiosity_world_option_model_learning(self):
        env = GridWorld(world="4-room")
        num_mixtures = 3
        num_features = 4
        batch_size = 512

        state_space = env.state_space.with_batch_rank()
        state_feature_space = FloatBox(shape=(num_features,), add_batch_rank=True)
        action_space = env.action_space.with_batch_rank()

        # Set the input dict for the Model's NN.
        nn_inputs_space = Dict(
            {"actions": action_space, "states": state_space, "next_states": state_space},
            add_batch_rank=True
        )
        # Set the labels Space (we are outputting actions and phi (latent state space feature vectors)).
        #labels_space = Dict(
        #    {"predicted_actions": action_space, "predicted_phi_": state_feature_space}, add_batch_rank=True
        #)

        icwom = IntrinsicCuriosityWorldOptionModel(action_space=action_space, encoder_network=[
            {"type": "reshape", "flatten": True, "flatten_categories": True},  # flatten the int space
            {"type": "dense", "units": 64, "activation": "relu", "scope": "layerA"},
            {"type": "dense", "units": 64, "activation": "relu", "scope": "layerB"}
        ], world_option_model_network=[
            {"type": "dense", "units": 64, "activation": "relu", "scope": "layer-womA"},
            {"type": "dense", "units": 64, "activation": "relu", "scope": "layer-womB"}
        ], num_features=num_features, post_phi_concat_network=[
                {"type": "dense", "units": 128, "activation": "relu", "scope": "layerC"}
        ], num_mixtures=num_mixtures, beta=0.2, optimizer={"type": "adam", "learning_rate": 0.0003})

        test = ComponentTest(component=icwom, input_spaces=dict(
            nn_inputs=nn_inputs_space, #labels=labels_space,
            states=state_space
        ))  #, build_kwargs={"optimizer": icwom.optimizer})

        all_icwom_vars = test.read_variable_values(icwom.variable_registry)

        # Create a completely independent auto-decoder Model.
        auto_decoder = SupervisedModel(supervised_predictor_spec=dict(
            network_spec=[{"type": "dense", "units": 128, "activation": "relu", "scope": "layer-decA"},
                          {"type": "dense", "units": 128, "activation": "relu", "scope": "layer-decB"}],
            output_space=state_space,
            deterministic=False
        ), loss_function_spec=dict(
            type="neg-log-likelihood", distribution_spec=dict(type="categorical")
        ), optimizer_spec=dict(type="adam", learning_rate=0.0003)
        )

        test_auto_decoder = ComponentTest(
            component=auto_decoder, input_spaces=dict(
                nn_inputs=state_feature_space, labels=state_space
            )
            #build_kwargs={"optimizer": auto_decoder.optimizer}
        )

        phi_cache = {}  # Store latest phis per state in this dict.
        render_cache = {}  # Stores the rendering info of each state.
        losses = []  # Losses for the ICWOM.
        auto_decoder_losses = []  # Losses for the Decoder.
        state_counts = Counter()  # Counts the occurrence of each state for exploration statistics.

        s = env.reset()

        # Do n update rounds.
        for batch in range(200):
            # Buffers for trajectories.
            states = []
            actions = []
            terminals = []
            next_states = []

            for ts in range(batch_size):
                int_s = int(s)
                states.append(s)
                state_counts[int_s] += 1

                if int_s not in render_cache:
                    render_cache[int_s] = env.render_txt()

                # Act randomly.
                a = action_space.sample()
                actions.append(a)
                s, r, t, _ = env.step(a)
                terminals.append(t)
                next_states.append(s)
                if t:
                    s = env.reset()

            # Get all phi' for observed next-states.
            phi_s = test.test(("get_phi", np.array(next_states)))["predictions"]
            # Update our model.
            out = test.test(("update", [
                dict(states=np.array(states), actions=np.array(actions), next_states=np.array(next_states))  # NN-inputs
            ], ["loss", "step_op"]))
            # Keep track of losses.
            losses.append(out["loss"])

            # Make sure all of the Model's variables have changed all their values after the update.
            if batch % 10 == 0:
                all_icwom_vars_after_update = test.read_variable_values(icwom.variable_registry)
                for key, var_before_update in all_icwom_vars.items():
                    self.assertTrue((var_before_update != all_icwom_vars_after_update[key]).any())

            # Train our decoder with predicting the original state from the phis.
            out = test_auto_decoder.test(("update", [phi_s, np.array(next_states)], ["loss", "step_op"]))
            # Keep track of losses.
            auto_decoder_losses.append(out["loss"])

            if batch % 10 == 0:
                print("Batch {}: loss={} auto-decoder-loss={}".format(batch, losses[-1], auto_decoder_losses[-1]))
                for i, s in enumerate(next_states):
                    phi_cache[int(s)] = phi_s[i]

        #print()
        for s in sorted(phi_cache.keys()):
            phi = phi_cache[s]
            # Get the original auto-decoder state
            auto_decoded_state = test_auto_decoder.test(("predict", np.array([phi]), "predictions"))
            auto_decoded_state = auto_decoded_state["predictions"][0]
            #print("state={} phi={} auto-decoded-state={}\n{}\n".format(
            #    s, phi, auto_decoded_state, render_cache.get(s, "")
            #))
        #print()

        # Print softmaxed distribution over state visits:
        state_counts = np.array(list(state_counts.values()))
        state_distribution = softmax(state_counts / np.mean(state_counts))
        #print("Distribution over all states: {}".format(state_distribution))

        self.assertTrue(np.mean(losses[:10]) > np.mean(losses[-10:]))

        test.terminate()

