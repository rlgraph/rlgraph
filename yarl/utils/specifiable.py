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

import importlib
import json
import os
import yaml

from .yarl_error import YARLError


class Specifiable(object):
    """
    Members of this class support the methods from_file, from_spec and from_mixed.
    """

    # An optional python dict with supported str-to-ctor mappings for this class.
    __lookup_classes__ = None

    @classmethod
    def from_file(cls, filename):
        """
        Create object from spec saved in filename. Expects json or yaml format.

        Args:
            filename: file containing the spec (json or yaml)

        Returns: object

        """
        path = os.path.join(os.getcwd(), filename)
        if not os.path.isfile(path):
            raise YARLError('No such file: {}'.format(filename))

        with open(path, 'rt') as fp:
            if path.endswith('.yaml') or path.endswith('.yml'):
                spec = yaml.load(fp)
            else:
                spec = json.load(fp)

        return cls.from_spec(spec=spec)

    @classmethod
    def from_spec(cls, spec=None, **kwargs):
        """
        Uses the given spec to create an object. The `type` key can be used to instantiate a different (sub-)class.
        The following inputs are valid as types: a) a python callable, b) a string of a python callable,
        c) an identifier for a class, as specified in cls.__lookup_classes__. Per default, the base class will be
        instantiated.

        Args:
            spec (Optional[dict]): The specification dict.

        Keyword Args:
            kwargs (any): Optional possibility to pass the c'tor arguments in here and use spec as the type-only info.
                Then we can call this like: from_spec([type]?, [**kwargs for ctor])
                If `spec` is already a dict, then `kwargs` will be merged with spec (overwriting keys in `spec`) after
                "type" has been popped out of `spec`.
                If a constructor of a Specifiable needs an *args list of items, the special key `_args` can be passed
                inside `kwargs` with a list type value (e.g. kwargs={"_args": [arg1, arg2, arg3]}).

        Returns:
            The object generated from the spec.
        """
        # `type_`: Indicator for the Specifiable's constructor.
        # `ctor_args`: *args arguments for the constructor.
        # `ctor_kwargs`: **kwargs arguments for the constructor.
        if isinstance(spec, dict):
            if "type" in spec:
                type_ = spec.pop("type", None)
            else:
                type_ = None
            ctor_kwargs = spec
            ctor_kwargs.update(kwargs)  # give kwargs priority
        else:
            type_ = spec
            ctor_kwargs = kwargs
        # Special `_args` field in kwargs for *args-utilizing constructors.
        ctor_args = kwargs.pop("_args", [])

        # Figure out the actual constructor (class) from `type_`.
        constructor = None
        # Default case: same class
        if type_ is None:
            constructor = cls
        # type_ is already a created object of this class -> Take it as is.
        elif isinstance(type_, cls):
            return type_
        # Case c) identifier in cls.__lookup_classes__
        elif cls.__lookup_classes__ is not None and isinstance(cls.__lookup_classes__, dict) and \
                type_ in cls.__lookup_classes__:
            constructor = cls.__lookup_classes__[type_]
        # Case a) python callable
        elif callable(type_):
            constructor = type_
        # Case b) a string of a python callable
        elif isinstance(type_, str) and type_.find('.') != -1:
            module_name, function_name = type_.rsplit(".", 1)
            module = importlib.import_module(module_name)
            constructor = getattr(module, function_name)

        if not constructor:
            raise YARLError("Invalid type: {}".format(type_))

        obj = constructor(*ctor_args, **ctor_kwargs)
        assert isinstance(obj, constructor)

        return obj

    @classmethod
    def from_mixed(cls, mixed):
        """
        Create object from mixed input. Input might be a) a dict object (pass to `from_spec`), or b) a filename
        (pass to `from_file`)

        Args:
            mixed: dict or filename.

        Returns: Object

        """
        if isinstance(mixed, dict):
            return cls.from_spec(spec=mixed)
        elif isinstance(mixed, str):
            return cls.from_file(filename=mixed)
        else:
            raise YARLError('Invalid input to `from_mixed`: {}'.format(mixed))
