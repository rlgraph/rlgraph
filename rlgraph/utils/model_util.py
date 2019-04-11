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
import logging

from rlgraph.components import ValueFunction


def register_value_function(name, cls):
    """
    Registers a custom value function by name and type.

    After registration, the custom network can be used by passing to agents
    a value function spec with type=name.

    Args:
        name (str): Name of custom value function. Should usually correspond to the type name, e.g. if the class
            is named MyCustomVF, the name should be my_custom_vf or similar.
        cls (ValueFunction): Custom value function inheriting from ValueFuction.
    """
    if name in ValueFunction.__lookup_classes__[name]:
        raise ValueError("Name {} is already defined in ValueFunctions. All names are: {}".format(
            name, ValueFunction.__lookup_classes__.keys()
        ))
    else:
        ValueFunction.__lookup_classes__[name] = cls
        logging.info("Registered custom value function {} under name: {}".format(cls, name))