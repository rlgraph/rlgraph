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

from collections import defaultdict
import numpy as np
import pandas as pd
import time
import os


class PerformanceTimer(object):
    def __init__(self, filename, columns):
        self.start = 0
        self.last = 0

        self.data = defaultdict(lambda: np.nan)
        self.subtimers = dict()

        self.filename = filename
        self.columns = columns

        self.complete_data = list()

    def __enter__(self):
        self.start = time.time()
        self.last = time.time()
        return self

    def __call__(self, column):
        assert column in self.columns

        delta = time.time() - self.last
        self.data[column] = delta

        self.last = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_delta = time.time() - self.start

        all_columns = ['start']
        data = dict(start=self.start)
        for column in self.columns:
            if column in self.subtimers:
                subdata = self.subtimers[column].get_results(prefix=column)
                all_columns += subdata.keys()
                data.update(subdata)
            else:
                all_columns.append(column)
                data[column] = self.data[column]

        all_columns.append('total_time')
        data['total_time'] = end_delta

        self.complete_data.append(data)

    def get_results(self, prefix=None):
        df = pd.DataFrame(self.complete_data)
        return_data = dict()
        column_names = list()
        for column in self.columns:
            if not prefix:
                column_name = column
            else:
                column_name = '{}_{}'.format(prefix, column)
            column_data = np.asarray(df[column])
            return_data[column_name] = np.mean(column_data)
            column_names.append(column_name)

        return return_data

    def write(self):
        df = pd.DataFrame(self.complete_data)

        write_header = False
        if not os.path.exists(self.filename):
            write_header = True

        df.to_csv(self.filename, mode='at', index=write_header)

    def sub(self, column, subcolumns):
        if not column in self.subtimers:
            subtimer = PerformanceTimer(filename=None, columns=subcolumns)
            self.subtimers[column] = subtimer
        else:
            subtimer = self.subtimers[column]

        return subtimer
