# Copyright 2018 The RLgraph authors. All Rights Reserved.
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


from rlgraph.environments import Environment
from six.moves import xrange as range_


class VectorEnv(Environment):
    """
    Abstract multi-environment class to support stepping multiple environments
    at once.
    """
    def __init__(self, num_envs, env_spec):
        """
        Args:
            num_envs (int): Number of environments
            env_spec Union[callable, dict]: Environment spec dict.
        """
        self.num_envs = num_envs
        self.environments = list()

        for _ in range_(num_envs):
            if isinstance(env_spec, dict):
                env = Environment.from_spec(env_spec)
            elif hasattr(env_spec, '__call__'):
                env = env_spec()
            else:
                raise ValueError("Env_spec must be either a dict containing an environment spec or a callable"
                                 "returning a new environment object.")
            self.environments.append(env)
        super(VectorEnv, self).__init__(state_space=self.environments[0].state_space,
                                        action_space=self.environments[0].action_space)

    def get_env(self):
        """
        Returns an underlying environment instance (the first).
        Returns:
            Environment: Environment instance.
        """
        return self.environments[0]

    def render(self):
        self.environments[0].render()

    def seed(self, seed=None):
        return [env.seed(seed) for env in self.environments]

    def reset_all(self):
        """
        Resets all environments.

        Returns:
            any: New states for environments.
        """
        raise NotImplementedError

    def reset(self, index=0):
        """
        Resets specific environments
        Args:
            index (Optional[int]): Environment to reset, defaults to first.
        Returns:
            any: New state for environment.
        """
        raise NotImplementedError

    def step(self, **kwargs):
        """
        Executes steps on a vector of environments.
        Args:
            **kwargs: Step args.

        Returns:
            any: Step results for each environment.
        """
        raise NotImplementedError

    def __str__(self):
        return [str(env) for env in self.environments]
