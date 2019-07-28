import logging
import unittest

import numpy as np

from rlgraph.agents.sac_agent import SACAlgorithmComponent, SACAgent
from rlgraph.components import Policy, ReplayMemory, AdamOptimizer, Synchronizable
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution.rules.sync_rules import SyncRules
from rlgraph.spaces import FloatBox, BoolBox, IntBox
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal
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
        action_space = FloatBox(shape=(1,), low=-2.0, high=2.0)
        float_action_space = action_space.with_batch_rank().map(
            mapping=lambda flat_key, space: space.as_one_hot_float_space() if isinstance(space, IntBox) else space
        )
        terminal_space = BoolBox(add_batch_rank=True)
        rewards_space = FloatBox(add_batch_rank=True)
        policy = Policy.from_spec(config["policy"], action_space=action_space)
        policy.add_components(Synchronizable(), expose_apis="sync")

        agent_component = SACAlgorithmComponent(
            agent=None,
            policy_spec=policy,
            q_function_spec=config["q_function"],
            preprocessing_spec=None,
            memory_spec=ReplayMemory.from_spec(config["memory"]),
            discount=config["discount"],
            initial_alpha=config["initial_alpha"],
            target_entropy=None,
            optimizer_spec=AdamOptimizer.from_spec(config["optimizer_spec"]),
            q_function_optimizer_spec=AdamOptimizer.from_spec(
                config["q_function_optimizer_spec"]
            ),
            #alpha_optimizer=None,
            q_function_sync_rules=SyncRules(sync_every_n_updates=10, sync_tau=1.0),
            num_q_functions=2,
            # Switch off this API as SAC's state value function takes 2 inputs (not 1).
            #switched_off_apis={"get_state_values"}
        )

        test = ComponentTest(
            component=agent_component,
            input_spaces=dict(
                states=state_space.with_batch_rank(),
                preprocessed_states=state_space.with_batch_rank(),
                env_actions=action_space.with_batch_rank(),
                actions=float_action_space,
                rewards=rewards_space,
                next_states=state_space.with_batch_rank(),
                terminals=terminal_space,
                importance_weights=FloatBox(add_batch_rank=True),
                deterministic=bool,
                policy_weights="variables:{}".format(policy.scope),
                time_percentage=float
                # TODO: how to provide the space for multiple component variables?
                #q_weights=Dict(
                #    q_0="variables:{}".format(q_function.scope),
                #    q_1="variables:{}".format(agent_component._q_functions[1].scope),
                #)
            ),
            action_space=action_space,
            #build_kwargs=dict(
            #    optimizer=agent_component.optimizer,
            #    build_options=dict(
            #        vf_optimizer=agent_component.q_function_optimizer,
            #    ),
            #)
        )

        batch_size = 10
        action_sample = action_space.with_batch_rank().sample(batch_size)
        rewards = rewards_space.sample(batch_size)
        # Check, whether an update runs ok.
        result = test.test(("update_from_external_batch", [
            state_space.sample(batch_size),
            action_sample,
            rewards,
            [True] * batch_size,
            state_space.sample(batch_size),
            [1.0] * batch_size,  # importance weights
            0.1  # time-percentage
        ]))
        self.assertTrue(result["actor_loss"].dtype == np.float32)
        self.assertTrue(result["critic_loss"].dtype == np.float32)

        action_sample = np.linspace(-1, 1, batch_size).reshape((batch_size, 1))
        q_values = test.test(("get_q_values", [state_space.sample(batch_size), action_sample]))
        for q_val in q_values:
            self.assertTrue(q_val.dtype == np.float32)
            self.assertTrue(q_val.shape == (batch_size, 1))

        action_sample = test.test(
            ("get_actions_from_preprocessed_states", [state_space.sample(batch_size), False, 0.1])
        )["actions"]
        self.assertTrue(action_sample.dtype == np.float32)
        self.assertTrue(action_sample.shape == (batch_size, 1))

    def test_policy_get_set_weights(self):
        """
        Tests weight syncing of policy (and only policy, not Q-functions).
        """
        env = OpenAIGymEnv("CartPole-v0")
        agent = SACAgent.from_spec(
            config_from_path("configs/sac_agent_for_cartpole.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        # update every n == 4

        weights = agent.get_weights()
        print("weights = ", weights.keys())

        new_weights = {}
        for key, value in weights["policy_weights"].items():
            new_weights[key] = value + 0.01

        agent.set_weights(policy_weights=new_weights, q_function_weights=None)

        updated_weights = agent.get_weights()["policy_weights"]
        recursive_assert_almost_equal(updated_weights, new_weights)

    def test_image_value_functions(self):
        """
        Tests if actions and states are successfully merged on image inputs to compute Q(s,a).
        """
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True)
        agent = SACAgent.from_spec(
            config_from_path("configs/sac_agent_for_pong.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        # Test updating from image batch.
        batch = dict(
            states=agent.preprocessed_state_space.sample(32),
            actions=env.action_space.sample(32),
            rewards=np.ones((32,)),
            terminals=np.zeros((32,)),
            next_states=agent.preprocessed_state_space.sample(32),
            importance_weights=np.ones((32,))
        )
        print(agent.update(batch))

    def test_apex_integration(self):
        try:
            from rlgraph.execution.ray import ApexExecutor
        except (NameError, ImportError):
            return

        env_spec = dict(
            type="openai",
            gym_env="PongNoFrameskip-v4",
            # The frameskip in the agent config will trigger worker skips, this
            # is used for internal env.
            frameskip=4,
            max_num_noops=30,
            episodic_life=False,
            fire_reset=True
        )

        # Not a learning config, just testing integration.
        executor = ApexExecutor(
            environment_spec=env_spec,
            agent_config=config_from_path("configs/ray_sac_pong_test.json"),
        )

        # Tests short execution.
        result = executor.execute_workload(workload=dict(
            num_timesteps=5000, report_interval=100, report_interval_min_seconds=1)
        )
        print(result)
