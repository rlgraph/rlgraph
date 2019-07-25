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

from __future__ import absolute_import, division, print_function

import numpy as np

from rlgraph.agents.agent import Agent
from rlgraph.environments.environment import Environment
from rlgraph.learners.supervised_learner import SupervisedLearner
from rlgraph.execution.single_threaded_worker import SingleThreadedWorker
from rlgraph.spaces import FloatBox, Space


class DADSWorker(SingleThreadedWorker):
    """
    "Dynamics Aware Discovery of Skills" Worker implementation [1]
    Learns in two phases:
    1) Learns continuous skills in unsupervised fashion (without rewards).
    2) Uses a planner to apply these skills to a given optimization problem (with reward function).

    DADS is a wrapper around any RL-algorithm (SAC is used in the paper), only adding an extra skill-vector
    input to the state space. Skills are selected during training time at (uniform) random and then fed  as continuous
    or one-hot vectors into the chosen algorithm's policy.

    [1]: Dynamics-Aware Unsupervised Discovery of Skills - Sharma et al - Google Brain - 2019
    https://arxiv.org/pdf/1907.01657.pdf
    """

    def __init__(
            self,
            agent_spec,
            env_spec,
            skill_dynamics_model_spec,
            preprocessing_spec=None,
            worker_executes_preprocessing=None,
            num_skill_dimensions=4,
            use_discrete_skills=False,
            **kwargs
    ):
        """
        Args:
            skill_dynamics_model_spec (Union[dict,SupervisedModel]): A specification dict for the
                environment skill-dynamics-learning SupervisedModel or the model Component directly.

            num_skill_dimensions (int): The size of the skills vector.

            use_discrete_skills (bool): Whether to use discrete skills (one-hot vectors of size `num_skill_dimensions`).
                Default: False (continuous skills).
        """
        self.num_skill_dimensions = num_skill_dimensions
        self.use_discrete_skills = use_discrete_skills

        dummy_env = Environment.from_spec(env_spec)

        # Add the skill-vector space to the Agent's state-space.
        # The Agent should not know about the extra skill dimensions and treat the skill-vector
        # as a part of the environment.
        if isinstance(agent_spec, Agent):
            raise Exception(
                "`agent_spec` must not be a built Agent object due to necessary skill-vector/state-space "
                "manipulation for DADS learning!"
            )
        # Add skill-Space to the Agent's state_space.
        env_state_space = Space.from_spec(agent_spec.get("state_space", dummy_env.state_space))
        # TODO: Make this work for any kind of state Space.
        assert isinstance(env_state_space, FloatBox) and env_state_space.rank == 1
        new_dim = env_state_space.shape[0] + self.num_skill_dimensions
        state_space = FloatBox(shape=(new_dim,))
        # Add state and action spaces to Agent config and create Worker base.
        agent_spec["state_space"] = state_space
        agent_spec["action_space"] = dummy_env.action_space

        ## Change the network spec to a multi-input-stream NN, one for the skill vector, the other for the rest.
        ## TODO: Make this work for any kind of network spec. Right now it only works for simple dense NNs.
        #network_spec = agent_spec["network_spec"]
        #network_spec = dict(type="multi-input-stream-nn", input_network_specs=dict(
        #    skill_vector=[], state=[]
        #), post_concat_network_spec=network_spec)
        ## TODO: Un-simplify: SAC Q-network same as policy network.
        #agent_spec["network_spec"] = agent_spec["q_function_spec"] = network_spec

        # Force worker-preprocessing off.
        # We will add our own preprocessing for now (adding the skill-vector to the states vector).
        assert not worker_executes_preprocessing and preprocessing_spec is None, \
            "ERROR: DADS does not allow worker-preprocessing! Send `preprocessing_spec`=None and " \
            "`worker_executes_preprocessing`=None"
        worker_executes_preprocessing = True

        # Define our simple preprocessor concatenating the skill-vector with the state.
        def preprocessing_spec(state, in_space, env_id):
            return np.concatenate([np.array([self.skill_vectors[env_id]]), state], axis=-1)

        super(DADSWorker, self).__init__(
            agent_spec=agent_spec, env_spec=env_spec,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=worker_executes_preprocessing,
            **kwargs
        )

        # Reserve skill-vector (with batch rank==num envs in vector-env) to sample into.
        self.skill_vectors = {env_id: np.ndarray(shape=(self.num_skill_dimensions,), dtype=np.float32) for env_id in self.env_ids}

        # Create supervised learner that learns a SupervisedModel on the skill-dependent-transition-function
        # q(s'|s,z), where z=skill vector
        # Convenience: Add output_space if not already done in config (determined by env).
        if isinstance(skill_dynamics_model_spec, dict) and \
                isinstance(skill_dynamics_model_spec["supervised_predictor_spec"], dict) and \
                "output_space" not in skill_dynamics_model_spec["supervised_predictor_spec"]:
            skill_dynamics_model_spec["supervised_predictor_spec"]["output_space"] = env_state_space
        self.transition_function_learner = SupervisedLearner.from_spec(
            input_space=state_space,
            output_space=env_state_space,
            supervised_model_spec=skill_dynamics_model_spec
        )
        print()

    def _observe(self, env_ids, states, actions, rewards, next_states, terminals, **other_data):
        """
        Args:
            env_ids ():
            states ():
            actions ():
            rewards ():
            next_states ():
            terminals ():
            **other_data ():
        """
        super(DADSWorker, self)._observe(env_ids, states, actions, rewards, next_states, terminals, **other_data)

    def pre_reset(self, env_id):
        # Sets the skill vector for this env to random uniform [-1, 1] for the next episode.
        self.skill_vectors[env_id] = np.random.uniform(-1.0, 1.0, size=self.skill_vectors[env_id].shape)
