# Copyright 2018/2019 ducandu GmbH. All Rights Reserved.
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

from rlgraph.utils.specifiable import Specifiable


class UpdateRules(Specifiable):
    """
    Simple settings class specifying how Workers should schedule and perform
    Agent.update() calls.
    """
    def __init__(
            self, do_update=True, unit="time_step", update_every_n_units=1,
            update_repeats=1, first_update_after_n_units=0
    ):
        """
        Args:
            do_update (bool): Whether to make calls to Agent.update(). Default: True.
            unit (str): One of "time_step" or "episode". What to count when determining update
                intervals, by time-step (single act = 1 time step) or episode.
            update_every_n_units (int): Every how many units (ts or episodes) should one call `Agent.update()`?
            update_repeats (int): How many update calls to make in a single step. Default: 1.
            first_update_after_n_units (int): If > 0, wait for n units (ts or episodes) before making any
                calls to `Agent.update()`.
        """
        super(UpdateRules, self).__init__()

        self.do_update = do_update
        assert unit == "time_step" or unit == "episode"
        self.unit = unit
        self.update_every_n_units = update_every_n_units
        self.update_repeats = update_repeats
        self.first_update_after_n_units = first_update_after_n_units
