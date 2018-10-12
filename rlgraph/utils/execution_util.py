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


def print_call_chain(profile_data, sort=True, filter_threshold=None):
    """
    Prints a component call chain stdout. Useful to analyze define by run performance.

    Args:
        profile_data (list): Component file data.
        sort (bool): If true, sorts call sorted by call duration.
        filter_threshold (Optional[float]): Optionally specify an execution threshold in seconds (e.g. 0.01).
            All call entries below the threshold be dropped from the printout.
    """
    original_length = len(profile_data)
    if filter_threshold is not None:
        assert isinstance(filter_threshold, float), "ERROR: Filter threshold must be float but is {}.".format(
            type(filter_threshold))
        profile_data = [data for data in profile_data if data[2] > filter_threshold]
    if sort:
        res = sorted(profile_data, key=lambda v: v[2], reverse=True)
        print("Call chain sorted by runtime ({} calls, {} before filter):".
              format(len(profile_data), original_length))
        for v in res:
            print("{}.{}: {} s".format(v[0], v[1], v[2]))
    else:
        print("Directed call chain ({} calls, {} before filter):".format(len(profile_data), original_length))
        for i in range(len(profile_data) - 1):
            v = profile_data[i]
            print("({}.{}: {} s) ->".format(v[0], v[1], v[2]))
        v = profile_data[-1]
        print("({}.{}: {} s)".format(v[0], v[1], v[2]))
