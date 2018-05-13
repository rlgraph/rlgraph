# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from yarl.components import Component
from yarl.spaces import Space, Dict, Tuple


class Memory(Component):
    """
    Abstract memory component.
    """

    def __init__(
        self,
        record_space,
        capacity=1000,
        name="",
        scope="memory",
        sub_indexes=None,
    ):
        """
        Args:
            record_space (Space): The space of the records to be stored.
            capacity (int): The number of records that can be stored in this memory.
            sub_indexes (Union[None,List[str]]): A list of strings of sub-indexes to add to the main (enumeration)
                index. For example: Each record could belong to a certain RL-episode. Each sub-index gets its own
                vector variable with the size of the number of records as well as a counter that holds the highest
                value of that sub-index currently in the memory.
        """
        super(Memory, self).__init__(name=name, scope=scope)

        # Variables.
        self.record_registry = None

        self.capacity = capacity
        self.record_space = record_space
        self.sub_indexes_spec = sub_indexes

    def create_variables(self):
        # Main memory.
        buffer_variables = self.get_variable(name="replay-buffer", trainable=False,
                                             from_space=self.record_space, flatten=True)
        self.record_registry = self.record_space.flatten(mapping=lambda key, primitive: buffer_variables[key])

    #def build_record_variable_registry(self, variable_or_dict):
    #    """
    #    Builds a flat variable registry from a recursively defined variable Space.
    #    Args:
    #        variable_or_dict Union[Dict, tf.Variable]: Dict containing variables or variable.
    #    """
    #    for key, value in variable_or_dict:
    #        if isinstance(value, Dict):
    #            self.build_record_variable_registry(value)
    #        elif isinstance(value, Tuple):
    #            self.build_record_variable_registry(value[0])
    #            self.build_record_variable_registry(value[1])
    #        else:
    #            self.record_registry[key] = value

    #def scatter_update_records(self, records, indices, updates):
    #    """
    #    Updates record variables using the variable registry.
    #
    #    Args:
    #        records (dict): Dict containing record data. Keys must match keys in record space,
    #            values must be tensors.
    #        indices (array): Indices to update.
    #        updates (list): Assignments to update.
    #    """
    #    for name, value in records:
    #        if isinstance(value, Dict):
    #            self.scatter_update_records(value, indices, updates)
    #        elif isinstance(value, Tuple):
    #            self.scatter_update_records(value[0], indices, updates)
    #            self.scatter_update_records(value[1], indices, updates)
    #        else:
    #            updates.append(self.scatter_update_variable(
    #                variable=self.record_registry[name],
    #                indices=indices,
    #                updates=value
    #            ))

    def _computation_insert(self, records):
        """
        Inserts one or more complex records.

        Args:
            records (OrderedDict): OrderedDict containing record data. Keys must match keys in flattened record
                space, values must be tensors. Use the Component's flatten options to .
        """
        raise NotImplementedError

    def _computation_get_records(self, num_records):
        """
        Returns a number of records according to the retrieval strategy implemented by
        the memory.

        Args:
            num_records (int): Number of records to return.

        Returns: The retrieved records.
        """
        raise NotImplementedError

    def _computation_get_sequences(self, num_sequences, sequence_length):
        """
        Retrieves a given number of temporally consistent time-step sequences from the stored
        records (in the order they were inserted).

        Args:
            num_sequences (int): Number of sequences to retrieve.
            sequence_length (int): Length of sequences.

        Returns: The retrieved sequences.
        """
        raise NotImplementedError

    def _computation_clear(self):
        """
        Removes all entries from memory.
        """
        # Optional?
        pass

    def _computation_update_records(self, update):
        """
        Optionally ipdates memory records using information such as losses, e.g. to
        compute priorities.

        Args:
            update (dict): Any information relevant to update records, e.g. losses
                of most recently read batch of records.
        """
        pass


