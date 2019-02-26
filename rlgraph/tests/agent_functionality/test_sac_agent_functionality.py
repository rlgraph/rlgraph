import unittest
import logging
import numpy as np
from scipy import stats

from rlgraph.agents import Agent
from rlgraph.agents.sac_agent import SACAgentComponent, SyncSpecification
from rlgraph.spaces import FloatBox, BoolBox
from rlgraph.components import Policy, NeuralNetwork, ValueFunction, PreprocessorStack, ReplayMemory, AdamOptimizer,\
    Synchronizable
from rlgraph.environments import GaussianDensityAsRewardEnvironment
from rlgraph.execution import SingleThreadedWorker
from rlgraph.tests import ComponentTest, config_from_path
from rlgraph.utils import root_logger


class TestSACAgentFunctionality(unittest.TestCase):
    """
    Tests the SAC Agent's functionality.
    """
    root_logger.setLevel(level=logging.DEBUG)

    def test_sac_agent_component(self):
        continuous_action_space = FloatBox(low=-1.0, high=1.0)
        state_space = FloatBox(shape=(1,))
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
        q_function = ValueFunction(
            network_spec=[
                {"type": "dense", "units": 8, "activation": "relu"},
                {"type": "dense", "units": 8, "activation": "relu", "scope": "h2"},
                {"type": "dense", "units": 8, "activation": "relu", "scope": "h3"}
            ]
        )

        agent_component = SACAgentComponent(
            policy=policy,
            q_function=q_function,
            preprocessor=PreprocessorStack.from_spec([]),
            memory=ReplayMemory(),
            discount=0.8,
            initial_alpha=0.01,
            target_entropy=None,
            optimizer=AdamOptimizer(learning_rate=1e-2, scope="policy-optimizer"),
            vf_optimizer=AdamOptimizer(learning_rate=1e-2, scope="vf-optimizer"),
            q_sync_spec=SyncSpecification(sync_interval=10, sync_tau=1.0),
            num_q_functions=2
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
                weights="variables:{}".format(policy.scope),
                # TODO: how to provide the space for multiple component variables?
                #q_weights=Dict(
                #    q_0="variables:{}".format(q_function.scope),
                #    q_1="variables:{}".format(agent_component._q_functions[1].scope),
                #)
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
        true_mean = 0.5
        target_dist = stats.norm(loc=true_mean, scale=0.2)
        batch_size = 100
        for _ in range(1000):
            action_sample = action_space.sample(batch_size)
            rewards = target_dist.pdf(action_sample)
            result = test.graph_executor.execute((agent_component.update_from_external_batch, [
                state_space.sample(batch_size),
                action_sample,
                rewards,
                [True] * batch_size,
                state_space.sample(batch_size),
                [1.0] * batch_size  # importance
            ]))
            policy_loss.append(result["actor_loss"])
            values_loss.append(result["critic_loss"])

        action_sample = np.linspace(-1, 1, batch_size)
        q_values = test.graph_executor.execute((agent_component.get_q_values, [state_space.sample(batch_size), action_sample]))
        for q_val in q_values:
            q_val = q_val.flatten()
            np.testing.assert_allclose(q_val, target_dist.pdf(action_sample), atol=0.2)

        action_sample, _ = test.graph_executor.execute((agent_component.action_from_preprocessed_state, [state_space.sample(batch_size), False]))
        action_sample = action_sample.flatten()
        np.testing.assert_allclose(np.mean(action_sample), true_mean, atol=0.1)
