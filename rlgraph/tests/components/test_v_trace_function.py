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

import numpy as np
import unittest

from rlgraph.components.helpers.v_trace_function import VTraceFunction
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.utils.numpy import one_hot, softmax


class TestVTraceFunctions(unittest.TestCase):

    def test_v_trace_function(self):
        v_trace_function = VTraceFunction()

        #time_x_batch_space = FloatBox(add_batch_rank=True, add_time_rank=True)
        time_x_batch_x_3_space = FloatBox(shape=(3,), add_batch_rank=True, add_time_rank=True)
        time_x_batch_x_1_space = FloatBox(shape=(1,), add_batch_rank=True, add_time_rank=True)
        action_space = IntBox(3, add_batch_rank=True, add_time_rank=True)
        action_space_flat = FloatBox(shape=(3,), add_batch_rank=True, add_time_rank=True)
        input_spaces = dict(
            # Log rhos, discounts, rewards, values, bootstrapped_value.
            logits_actions_pi=time_x_batch_x_3_space,
            log_probs_actions_mu=time_x_batch_x_3_space,
            actions=action_space,
            actions_flat=action_space_flat,
            discounts=time_x_batch_x_1_space,
            rewards=time_x_batch_x_1_space,
            values=time_x_batch_x_1_space,
            bootstrapped_values=time_x_batch_x_1_space
        )

        test = ComponentTest(component=v_trace_function, input_spaces=input_spaces)

        actions = np.array([[1, 0], [1, 2], [2, 2]])
        # log_is_weights = log(prob_pi(a) / prob_mu(a)) =  - log(prob_mu(a))
        # => log(prob_pi(a)) = log_is_weights + log(prob_mu(a))
        # => prob_pi(a) = exp(log_is_weights + log(prob_mu(a)))
        # => logit_pi(a) = exp(log_is_weights + log(prob_mu(a)))
        log_is_weights = np.array([
            [[0.1, 1.0, 0.1], [0.65, 0.1, 0.1]],
            [[0.1, -0.4, 0.1], [0.1, 0.1, 0.2]],
            [[0.1, 0.1, 1.5], [0.1, 0.1, -1.0]]
        ])
        log_probs_actions_mu = np.array([
            [[-0.1, -4.5, -0.1], [-0.3, -0.1, -0.1]],
            [[-0.1, -2.0, -0.1], [-0.1, -0.1, -0.5]],
            [[-0.1, -0.1, -0.01], [-0.1, -0.1, -0.23]]
        ])
        logits_actions_pi = np.exp(log_is_weights + log_probs_actions_mu)
        # Batch of size=2, time-steps=3.
        # time_major=True (time axis is 0th, batch 1st).
        input_ = [
            # logits actions pi
            logits_actions_pi,
            # log-probs actions mu
            log_probs_actions_mu,
            # actions
            actions,
            # actions flat
            one_hot(actions, depth=3),
            # discounts
            np.array([[[0.99], [0.98]], [[0.97], [0.96]], [[0.5], [0.4]]]),
            # rewards
            np.array([[[1.0], [0.0]], [[2.0], [1.0]], [[-5.0], [2.0]]]),
            # values
            np.array([[[2.3], [-1.1]], [[1.563], [-2.0]], [[0.9], [-0.3]]]),
            # bootstrapped value
            np.array([[[2.3], [-1.0]]])
        ]

        """
        Calculation:
        vs = V(xs) + SUM[t=s to s+N-1]( gamma^t-s * ( PROD[i=s to t-1](ci) ) * dt_V )
        with:
            dt_V = rho_t * (rt + gamma V(xt+1) - V(xt))
            rho_t and ci being the clipped IS weights
        
        # Extract rho_t from log(rho_t):
        is_weights = exp(log_is_weights) = [2.71828183, 1.91554083], [0.67032005, 1.22140276], [4.48168907, 0.36787944]
        
        rho_t = min(rho_bar, is_weights) = [1.0, 1.0], [0.67032005, 1.0], [1.0, 0.36787944]
        
        # Calculate ci terms for all timesteps:
        ci = min(c_bar, is_weights) = [1.0, 1.0], [0.67032005, 1.0], [1.0, 0.36787944]
        
        # Calculate all terms dt_V (for each time step and each batch item):
        - shift values by 1 and add bootstrapped value at end:
        v_t_plus_1 = [1.563, -2.0], [0.9, -0.3], [2.3, -1.0]
        - calc rho_t * (rt + V(xt+1) - V(xt)):
        dt_V = rho_t * (rt + discounts * v_t_plus_1 - values) =
            [0.24737, -0.86], [0.87811947, 2.712], [-4.75,  0.698971]
        
        # Recursively calculate v-traces minus V (vs - V(xs)) for each timestep starting from end according to:
        # vs - V(xs) = dsV + γ * cs * (vs+1 − V(xs+1)) <- this last term will start with 0 for the first iteration
        item 1 in batch:
        s=t+2: vs - Vxs = -4.75 + 0.0 = -4.75  # start with initial vs-V(xs) == 0
        s=t+1: vs - Vxs = 0.87811926 + 0.97 * 0.67032005 * -4.75 = -2.210380370375
        s=t+0: vs - Vxs = 0.24737 + 0.99 * 1.0 * -2.210380370375 = -1.9409065666712495 
        vs = [-1.9409065666712495, -2.210380370375, -4.75] + values ([2.3, 1.563, 0.9]) =
            [0.35909343, -0.64738037, -3.85]
        
        item 2 in batch:
        s=t+2: vs - Vxs = 0.698971 + 0.0 = 0.698971
        s=t+1: vs - Vxs = 2.712 + 0.96 * 1.0 * 0.698971 = 3.3830121600000003
        s=t+0: vs - Vxs = -0.86 + 0.98 * 1.0 * 3.3830121600000003 = 2.4553519168
        vs = [2.4553519168, 3.3830121600000003, 0.698971] + values ([-1.1, -2.0, -0.3]) =
            [1.35535192, 1.38301216, 0.398971]
        
        # PG-Advantages
        vs_t_plus_1 = [-0.64738037, 1.38301216], [-3.85, 0.398971], [2.3, -1.0]
        At = rho_t_pg * (rt + discounts * vs_t_plus_1 - values) =
            [-1.94090657, 2.45535192], [-2.21038035, 3.38301216], [-4.75, 0.69897094]
        """
        expected_vs = np.array([[[0.35909343], [1.35535192]], [[-0.64738037], [1.38301216]], [[-3.85], [0.398971]]])
        expected_advantages = np.array([[-1.94090657, 2.45535192], [-2.21038035, 3.38301216], [-4.75, 0.69897094]])

        test.test(("calc_v_trace_values", input_), expected_outputs=[expected_vs, expected_advantages], decimals=4)
