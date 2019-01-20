import unittest
import logging
import numpy as np

from rlgraph.utils import root_logger
from rlgraph.tests import ComponentTest
from rlgraph.agents.sac_agent import SACLossFunction, SACAgentComponent, SyncSpecification
from rlgraph.spaces import FloatBox, BoolBox, Tuple, IntBox
from rlgraph.components import Policy, NeuralNetwork, ValueFunction, PreprocessorStack, ReplayMemory, AdamOptimizer,\
    Synchronizable
from scipy import stats


class TestSACAgentFunctionality(unittest.TestCase):
    """
    Tests the SAC Agent's functionality.
    """
    root_logger.setLevel(level=logging.DEBUG)

    def test_sac_loss_function(self):
        loss_function = SACLossFunction(
            alpha=0.9, discount=0.8
        )

        test = ComponentTest(
            component=loss_function,
            input_spaces=dict(
                log_probs_next_sampled=FloatBox(shape=(1,), add_batch_rank=True),
                q_values_next_sampled=Tuple(FloatBox(shape=(1,)), FloatBox(shape=(1,)), add_batch_rank=True),
                q_values=Tuple(FloatBox(shape=(1,)), FloatBox(shape=(1,)), add_batch_rank=True),
                log_probs_sampled=FloatBox(shape=(1,), add_batch_rank=True),
                q_values_sampled=Tuple(FloatBox(shape=(1,)), FloatBox(shape=(1,)), add_batch_rank=True),
                rewards=FloatBox(add_batch_rank=True),
                terminals=BoolBox(add_batch_rank=True),
                loss_per_item=FloatBox(add_batch_rank=True)
            ),
            action_space=IntBox(2, shape=(), add_batch_rank=True)
        )
        batch_size = 10

        inputs = [
            [[0.5]] * batch_size,  # log_probs_next_sampled
            ([[0.0]] * batch_size, [[1.0]] * batch_size),  # q_values_next_sampled
            ([[0.0]] * batch_size, [[1.0]] * batch_size),  # q_values
            [[0.5]] * batch_size,  # log_probs_sampled
            ([[.1]] * batch_size, [[.2]] * batch_size),  # q_values_sampled
            [1.0] * batch_size,  # rewards
            [False] * batch_size  # terminals
        ]

        policy_loss_per_item = [.35] * batch_size
        values_loss_per_item = [.2696] * batch_size

        test.test(
            (loss_function.loss, inputs),
            expected_outputs=[
                np.mean(policy_loss_per_item),
                policy_loss_per_item,
                np.mean(values_loss_per_item),
                values_loss_per_item
            ],
            decimals=5
        )

        test.test(
            (loss_function.loss_per_item, inputs),
            expected_outputs=[
                policy_loss_per_item,
                values_loss_per_item
            ],
            decimals=5
        )

        test.test(
            (loss_function.loss_average, [policy_loss_per_item]),
            expected_outputs=[np.mean(policy_loss_per_item)],
            decimals=5
        )

    def test_sac_agent_component(self):
        continuous_action_space = FloatBox(low=-1.0, high=1.0)
        state_space = FloatBox(shape=(2, ))
        action_space = continuous_action_space
        terminal_space = BoolBox(add_batch_rank=True)
        policy = Policy(
            network_spec=NeuralNetwork.from_spec([
                {"type": "dense", "units": 8, "activation": "relu"},
                {"type": "dense", "units": 8, "activation": "relu", "scope": "h2"},
                {"type": "dense", "units": 8, "activation": "relu", "scope": "h3"}
            ]),
            action_space=action_space
        )
        policy.add_components(Synchronizable(), expose_apis="sync")
        q_functions = [
            ValueFunction(
                network_spec=[
                    {"type": "dense", "units": 8, "activation": "relu"},
                    {"type": "dense", "units": 8, "activation": "relu", "scope": "h2"},
                    {"type": "dense", "units": 8, "activation": "relu", "scope": "h3"}
                ],
                scope="q-function-1"
            ),
            ValueFunction(
                network_spec=[
                    {"type": "dense", "units": 8, "activation": "relu"},
                    {"type": "dense", "units": 8, "activation": "relu", "scope": "h2"},
                    {"type": "dense", "units": 8, "activation": "relu", "scope": "h3"}
                ],
                scope="q-function-2"
            )
        ]

        agent_component = SACAgentComponent(
            policy=policy,
            q_functions=q_functions,
            preprocessor=PreprocessorStack.from_spec([]),
            memory=ReplayMemory(),
            discount=0.8,
            alpha=0.01,
            optimizer=AdamOptimizer(learning_rate=1e-2, scope="policy-optimizer"),
            vf_optimizer=AdamOptimizer(learning_rate=1e-2, scope="vf-optimizer"),
            q_sync_spec=SyncSpecification(sync_interval=10)
        )

        test = ComponentTest(
            component=agent_component,
            input_spaces=dict(
                states=state_space.with_batch_rank(add_batch_rank=True),
                preprocessed_states=state_space.with_batch_rank(add_batch_rank=True),
                actions=action_space.with_batch_rank(add_batch_rank=True),
                rewards=FloatBox(add_batch_rank=True),
                next_states=state_space.with_batch_rank(add_batch_rank=True),
                terminals=terminal_space,
                batch_size=int,
                preprocessed_s_prime=state_space.with_batch_rank(add_batch_rank=True),
                importance_weights=FloatBox(add_batch_rank=True),
                preprocessed_next_states=state_space.with_batch_rank(add_batch_rank=True),
                deterministic=bool,
                weights="variables:{}".format(policy.scope)
            ),
            action_space=action_space,
            build_kwargs=dict(
                optimizer=agent_component._optimizer,
                build_options=dict(
                    vf_optimizer=agent_component.vf_optimizer,
                ),
            )
        )

        policy_loss = []
        values_loss = []
        target_dist = stats.norm(loc=0.5, scale=0.2)
        for _ in range(1000):
            batch_size = 10
            action_sample = action_space.sample(batch_size)
            rewards = target_dist.pdf(action_sample)
            result = test.graph_executor.execute(("update_from_external_batch", [
                #state_space.sample(batch_size, fill_value=1.0),
                state_space.sample(batch_size),
                action_sample,
                rewards,  # reward
                terminal_space.sample(batch_size),  # terminal
                #state_space.sample(batch_size, fill_value=0.0),
                state_space.sample(batch_size),
                [1.0] * batch_size  # importance
            ]))
            policy_loss.append(result[0])
            values_loss.append(result[2])
        print(policy_loss)
        print(values_loss)
