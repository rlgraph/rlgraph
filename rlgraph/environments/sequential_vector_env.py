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

from queue import Queue
from threading import Thread

from six.moves import xrange as range_

from rlgraph.environments import VectorEnv, Environment


class SequentialVectorEnv(VectorEnv):
    """
    Sequential multi-environment class which iterates over a list of environments
    to step them.
    """
    def __init__(self, num_environments, env_spec, num_background_envs=1, async_reset=False):
        """
            num_background_envs (Optional([int]): Number of environments asynchronously
                reset in the background. Need to be calibrated depending on reset cost.
            async_reset (Optional[bool]): If true, resets envs asynchronously in another thread.
        """
        self.environments = []

        for _ in range_(num_environments):
            if isinstance(env_spec, dict):
                env = Environment.from_spec(env_spec)
            elif hasattr(env_spec, '__call__'):
                env = env_spec()
            else:
                raise ValueError("Env_spec must be either a dict containing an environment spec or a callable"
                                 "returning a new environment object.")
            self.environments.append(env)

        super(SequentialVectorEnv, self).__init__(
            num_environments=num_environments,
            state_space=self.environments[0].state_space, action_space=self.environments[0].action_space
        )

        self.async_reset = async_reset
        if self.async_reset:
            self.resetter = ThreadedResetter(env_spec, num_background_envs)
        else:
            self.resetter = Resetter()

    def seed(self, seed=None):
        return [env.seed(seed) for env in self.environments]

    def get_env(self, index=0):
        return self.environments[index]

    def reset(self, index=0):
        state, env = self.resetter.swap(self.environments[index])
        self.environments[index] = env
        return state

    def reset_all(self):
        states = []
        for i, env in enumerate(self.environments):
            state, env = self.resetter.swap(self.environments[i])
            states.append(state)
            self.environments[i] = env
        return states

    def step(self, actions, **kwargs):
        states, rewards, terminals, infos = [], [], [], []
        for i in range_(self.num_environments):
            state, reward, terminal, info = self.environments[i].step(actions[i])
            states.append(state)
            rewards.append(reward)
            terminals.append(terminal)
            infos.append(info)
        return states, rewards, terminals, infos

    def render(self, index=0):
        self.environments[index].render()

    def terminate(self, index=0):
        self.environments[index].terminate()

    def terminate_all(self):
        for env in self.environments:
            env.terminate()

    def __str__(self):
        return [str(env) for env in self.environments]


class Resetter(object):

    def swap(self, env):
        return env.reset(), env


class ThreadedResetter(Thread):
    """
    Keeps resetting environments in a queue,

    n.b. mechanism originally seen ins RLlib, since removed.
    """

    def __init__(self, env_spec, num_environments):
        super(ThreadedResetter, self).__init__()
        self.daemon = True
        self.in_need_reset = Queue()
        self.out_ready = Queue()

        # Create a set of environments ready to use.
        for _ in range_(num_environments):
            env = Environment.from_spec(env_spec)
            state = env.reset()
            self.out_ready.put((state, env))

        self.start()

    def swap(self, env):
        """
        Trade environment in need of reset for ready to use environment.
        Args:
            env (Environment): Environment object.

        Returns:
            any, Environment: State and ready to use environment.
        """
        self.in_need_reset.put(env)
        state, ready_to_use_env = self.out_ready.get(timeout=30)
        return state, ready_to_use_env

    def run(self):
        # Keeps resetting environments as they come in.
        while True:
            env = self.in_need_reset.get()
            state = env.reset()
            self.out_ready.put((state, env))
