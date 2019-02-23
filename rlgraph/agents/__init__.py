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

from rlgraph.agents.agent import Agent
from rlgraph.agents.dqn_agent import DQNAgent
from rlgraph.agents.dqfd_agent import DQFDAgent
from rlgraph.agents.apex_agent import ApexAgent
from rlgraph.agents.impala_agents import IMPALAAgent, SingleIMPALAAgent
from rlgraph.agents.ppo_agent import PPOAgent
from rlgraph.agents.actor_critic_agent import ActorCriticAgent
from rlgraph.agents.random_agent import RandomAgent
from rlgraph.agents.sac_agent import SACAgent


Agent.__lookup_classes__ = dict(
    apex=ApexAgent,
    apexagent=ApexAgent,
    actorcritic=ActorCriticAgent,
    dqn=DQNAgent,
    dqnagent=DQNAgent,
    dqfd=DQFDAgent,
    dqfdagent=DQFDAgent,
    impala=IMPALAAgent,  # TODO: Split non-single agents into Actor and Learner
    singleimpala=SingleIMPALAAgent,
    singleimpalaagent=SingleIMPALAAgent,
    ppo=PPOAgent,
    ppoagent=PPOAgent,
    random=RandomAgent,
    randomagent=RandomAgent,
    sac=SACAgent,
    sacagent=SACAgent
)

__all__ = ["Agent"] + \
          list(set(map(lambda x: x.__name__, Agent.__lookup_classes__.values())))

