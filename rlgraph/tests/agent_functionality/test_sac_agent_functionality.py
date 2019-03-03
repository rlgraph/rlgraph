import logging
import unittest

import numpy as np

from rlgraph.agents.sac_agent import SACAgentComponent, SyncSpecification
from rlgraph.components import Policy, ValueFunction, PreprocessorStack, ReplayMemory, AdamOptimizer, \
    Synchronizable
from rlgraph.spaces import FloatBox, BoolBox
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger


class TestSACAgentFunctionality(unittest.TestCase):
    """
    Tests the SAC Agent's functionality.
    """
    root_logger.setLevel(level=logging.DEBUG)

    def test_sac_agent_component_functionality(self):
        config = config_from_path("configs/sac_component_for_fake_env_test.json")

        # Arbitrary state space, state should not be used in this example.
        state_space = FloatBox(shape=(8,))
        continuous_action_space = FloatBox(shape=(1,), low=-2.0, high=2.0)
        terminal_space = BoolBox(add_batch_rank=True)
        rewards_space = FloatBox(add_batch_rank=True)
        policy = Policy.from_spec(config["policy"], action_space=continuous_action_space)
        policy.add_components(Synchronizable(), expose_apis="sync")
        q_function = ValueFunction.from_spec(config["value_function"])

        agent_component = SACAgentComponent(
            agent=None,
            policy=policy,
            q_function=q_function,
            preprocessor=PreprocessorStack.from_spec([]),
            memory=ReplayMemory.from_spec(config["memory"]),
            discount=config["discount"],
            initial_alpha=config["initial_alpha"],
            target_entropy=None,
            optimizer=AdamOptimizer.from_spec(config["optimizer"]),
            vf_optimizer=AdamOptimizer.from_spec(config["value_function_optimizer"], scope="vf-optimizer"),
            q_sync_spec=SyncSpecification(sync_interval=10, sync_tau=1.0),
            num_q_functions=2
        )

        test = ComponentTest(
            component=agent_component,
            input_spaces=dict(
                states=state_space.with_batch_rank(),
                preprocessed_states=state_space.with_batch_rank(),
                env_actions=continuous_action_space.with_batch_rank(),
                actions=continuous_action_space.with_batch_rank(),
                rewards=rewards_space,
                next_states=state_space.with_batch_rank(),
                terminals=terminal_space,
                batch_size=int,
                preprocessed_s_prime=state_space.with_batch_rank(),
                importance_weights=FloatBox(add_batch_rank=True),
                preprocessed_next_states=state_space.with_batch_rank(),
                deterministic=bool,
                weights="variables:{}".format(policy.scope),
                # TODO: how to provide the space for multiple component variables?
                #q_weights=Dict(
                #    q_0="variables:{}".format(q_function.scope),
                #    q_1="variables:{}".format(agent_component._q_functions[1].scope),
                #)
            ),
            action_space=continuous_action_space,
            build_kwargs=dict(
                optimizer=agent_component._optimizer,
                build_options=dict(
                    vf_optimizer=agent_component.vf_optimizer,
                ),
            )
        )

        batch_size = 10
        action_sample = continuous_action_space.with_batch_rank().sample(batch_size)
        rewards = rewards_space.sample(batch_size)
        # Check, whether an update runs ok.
        result = test.test(("update_from_external_batch", [
            state_space.sample(batch_size),
            action_sample,
            rewards,
            [True] * batch_size,
            state_space.sample(batch_size),
            [1.0] * batch_size  # importance
        ]))
        self.assertTrue(result["actor_loss"].dtype == np.float32)
        self.assertTrue(result["critic_loss"].dtype == np.float32)

        action_sample = np.linspace(-1, 1, batch_size).reshape((batch_size, 1))
        q_values = test.test(("get_q_values", [state_space.sample(batch_size), action_sample]))
        for q_val in q_values:
            self.assertTrue(q_val.dtype == np.float32)
            self.assertTrue(q_val.shape == (batch_size, 1))

        action_sample, _ = test.test(("action_from_preprocessed_state", [state_space.sample(batch_size), False]))
        self.assertTrue(action_sample.dtype == np.float32)
        self.assertTrue(action_sample.shape == (batch_size, 1))
