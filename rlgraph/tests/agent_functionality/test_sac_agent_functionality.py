import unittest
import logging
import numpy as np

from rlgraph.utils import root_logger
from rlgraph.tests import ComponentTest
from rlgraph.agents.sac_agent import SACLossFunction, SACAgentComponent, SyncSpecification, SACAgent
from rlgraph.spaces import FloatBox, BoolBox, Tuple, IntBox
from rlgraph.components import Policy, NeuralNetwork, ValueFunction, PreprocessorStack, ReplayMemory, AdamOptimizer,\
    Synchronizable
from rlgraph.environments import Environment
from rlgraph.execution import SingleThreadedWorker
from scipy import stats


class DummyEnvironment(Environment):
    """Dummy environment, the reward is density of the gaussian at the action."""
    def __init__(self, loc=-0.5, scale=0.2, episode_length=5):
        super(DummyEnvironment, self).__init__(state_space=FloatBox(shape=(1, )),
                                               action_space=FloatBox(shape=(1, ), low=-2.0, high=2.0))
        self.episode_length = episode_length
        self.episode_step = 0
        self.target_dist = stats.norm(loc=loc, scale=scale)

    def seed(self, seed=None):
        pass

    def reset(self):
        print("start")
        self.episode_step = 0
        return np.random.uniform(size=(1, ))

    def step(self, actions, **kwargs):
        reward = self.target_dist.pdf(actions)[0]
        print(actions, reward)
        self.episode_step += 1
        return np.random.uniform(size=(1, )), reward, self.episode_step >= self.episode_length, dict()

    def __str__(self):
        return self.__class__.__name__


class DummyEnvironment2(Environment):
    """Dummy environment, the reward is density of the gaussian at the action."""
    def __init__(self, episode_length=5):
        super(DummyEnvironment2, self).__init__(state_space=FloatBox(shape=(1, )),
                                                action_space=FloatBox(shape=(1, ), low=-2.0, high=2.0))
        self.episode_length = episode_length
        self.episode_step = 0
        self.loc = None

    def seed(self, seed=None):
        pass

    def reset(self):
        print("start")
        self.episode_step = 0
        self.loc = np.random.uniform(size=(1, )) * 2 - 1
        return self.loc

    def step(self, actions, **kwargs):
        reward = stats.norm(loc=self.loc, scale=0.1).pdf(actions)[0]
        print(self.loc[0], actions[0], reward)
        self.episode_step += 1
        self.loc = np.random.uniform(size=(1,)) * 2 - 1
        return self.loc, reward, self.episode_step >= self.episode_length, dict()

    def __str__(self):
        return self.__class__.__name__


class TestSACAgentFunctionality(unittest.TestCase):
    """
    Tests the SAC Agent's functionality.
    """
    root_logger.setLevel(level=logging.DEBUG)

    def _prepare_loss_function_test(self, loss_function):
        test = ComponentTest(
            component=loss_function,
            input_spaces=dict(
                alpha=float,
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
        return test

    def test_sac_loss_function(self):
        loss_function = SACLossFunction(
            target_entropy=0.1, discount=0.8
        )
        test = self._prepare_loss_function_test(loss_function)

        batch_size = 10
        inputs = [
            0.9,  # alpha
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
        alpha_loss_per_item = [-.54] * batch_size

        test.test(
            (loss_function.loss, inputs),
            expected_outputs=[
                np.mean(policy_loss_per_item),
                policy_loss_per_item,
                np.mean(values_loss_per_item),
                values_loss_per_item,
                np.mean(alpha_loss_per_item),
                alpha_loss_per_item
            ],
            decimals=5
        )

        test.test(
            (loss_function.loss_per_item, inputs),
            expected_outputs=[
                policy_loss_per_item,
                values_loss_per_item,
                alpha_loss_per_item
            ],
            decimals=5
        )

        test.test(
            (loss_function.loss_average, [policy_loss_per_item]),
            expected_outputs=[np.mean(policy_loss_per_item)],
            decimals=5
        )

    def test_sac_loss_function_no_target_entropy(self):
        loss_function = SACLossFunction(
            target_entropy=None, discount=0.8
        )
        test = self._prepare_loss_function_test(loss_function)

        batch_size = 10
        inputs = [
            0.9,  # alpha
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
        alpha_loss_per_item = [0.0] * batch_size

        test.test(
            (loss_function.loss, inputs),
            expected_outputs=[
                np.mean(policy_loss_per_item),
                policy_loss_per_item,
                np.mean(values_loss_per_item),
                values_loss_per_item,
                np.mean(alpha_loss_per_item),
                alpha_loss_per_item
            ],
            decimals=5
        )

        test.test(
            (loss_function.loss_per_item, inputs),
            expected_outputs=[
                policy_loss_per_item,
                values_loss_per_item,
                alpha_loss_per_item
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
        q_function = ValueFunction(
            network_spec=[
                {"type": "dense", "units": 8, "activation": "relu"},
                {"type": "dense", "units": 8, "activation": "relu", "scope": "h2"},
                {"type": "dense", "units": 8, "activation": "relu", "scope": "h3"}
            ],
            scope="q-function-1"
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
            policy_loss.append(result[4])
            values_loss.append(result[6])
        print(policy_loss)
        print(values_loss)

        action_sample = np.linspace(-1, 1, batch_size)
        q_values = test.graph_executor.execute((agent_component.get_q_values, [state_space.sample(batch_size), action_sample]))
        for q_val in q_values:
            q_val = q_val.flatten()
            np.testing.assert_allclose(q_val, target_dist.pdf(action_sample), atol=0.2)

        action_sample, _ = test.graph_executor.execute((agent_component.action_from_preprocessed_state, [state_space.sample(batch_size), False]))
        action_sample = action_sample.flatten()
        np.testing.assert_allclose(np.mean(action_sample), true_mean, atol=0.1)

    def test_sac_agent(self):
        env = DummyEnvironment2(episode_length=5)
        agent = SACAgent(
            discount=0.99,
            state_space=env.state_space,
            action_space=env.action_space,
            memory_spec=ReplayMemory(capacity=1000),
            update_spec={
                "update_mode": "time_steps",
                "do_updates": True,
                "update_interval": 1,
                "sync_interval": 1,
                "sync_tau": 0.05,
                "batch_size": 100,
                "num_iterations": 1
            },
            network_spec=[
                {
                    "type": "dense",
                    "units": 8,
                    "activation": "tanh",
                    "scope": "hidden1"
                },
                {
                    "type": "dense",
                    "units": 8,
                    "activation": "tanh",
                    "scope": "hidden2"
                }
            ],
            value_function_spec=[
                {
                    "type": "dense",
                    "units": 8,
                    "activation": "tanh",
                    "scope": "vf-hidden1"
                },
                {
                    "type": "dense",
                    "units": 8,
                    "activation": "tanh",
                    "scope": "vf-hidden2"
                }
            ],
            optimizer_spec={
                "type": "adam",
                "learning_rate": 3e-3
            },
            value_function_optimizer_spec={
                "type": "adam",
                "learning_rate": 3e-3
            },
            initial_alpha=.01
        )

        rewards = []

        def episode_finish_callback(reward, **kwargs):
            nonlocal rewards
            rewards.append(reward)

        worker = SingleThreadedWorker(env_spec=lambda: env, agent=agent, episode_finish_callback=episode_finish_callback)
        worker.execute_episodes(num_episodes=500)

        assert np.mean(rewards[:100]) < np.mean(rewards[-100:])
