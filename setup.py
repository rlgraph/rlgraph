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
from __future__ import print_function
from __future__ import division

import os
from setuptools import setup, find_packages

# Read __version__ avoiding imports that might be in install_requires
version_vars = dict()
with open(os.path.join(os.path.dirname(__file__), 'rlgraph', 'version.py')) as fp:
    exec(fp.read(), version_vars)

install_requires = [
    'opencv-python',
    'numpy',
    'pyyaml',
    'pytest',
    'six',
    'requests',
    'absl-py'
]

setup_requires = []

extras_require = {
    'tf': ['tensorflow'],
    'tf-gpu': ['tensorflow-gpu'],
    'pytorch': ['torch', 'torchvision'],  # TODO platform dependent.
    'gym': ['gym', 'atari-py'],
    'horovod': 'horovod',
    'ray': ['ray', 'lz4', 'pyarrow']
}

setup(
    name='rlgraph',
    version=version_vars['__version__'],
    description='A Framework for Flexible Deep Reinforcement Learning Graphs',
    url='https://rlgraph.org',
    author='The RLgraph development team',
    author_email='rlgraph@rlgraph.org',
    license='Apache 2.0',
    packages=[package for package in find_packages() if package.startswith('rlgraph')],
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False
)
