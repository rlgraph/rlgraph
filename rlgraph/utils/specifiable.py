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

from functools import partial
import importlib
import json
import multiprocessing
import os
import re
import yaml
import logging

from rlgraph.backend_system import get_backend
from rlgraph.utils.rlgraph_error import RLGraphError
from rlgraph.utils.util import default_dict

if get_backend() == "tf":
    import tensorflow as tf


class Specifiable(object):
    """
    Members of this class support the methods from_file, from_spec and from_mixed.
    """

    # An optional python dict with supported str-to-ctor mappings for this class.
    __lookup_classes__ = None
    # An optional default constructor to use without any arguments in case `spec` is None
    # and args/kwargs are both empty. This may be a functools.partial object.
    __default_constructor__ = None

    logger = logging.getLogger(__name__)

    @classmethod
    def from_spec(cls, spec=None, **kwargs):
        """
        Uses the given spec to create an object.
        If `spec` is a dict, an optional "type" key can be used as a "constructor hint" to specify a certain class
        of the object.
        If `spec` is not a dict, `spec`'s value is used directly as the "constructor hint".

        The rest of `spec` (if it's a dict) will be used as kwargs for the (to-be-determined) constructor.
        Additional keys in **kwargs will always have precedence (overwrite keys in `spec` (if a dict)).
        Also, if the spec-dict or **kwargs contains the special key "_args", it will be popped from the dict
        and used as *args list to be passed separately to the constructor.

        The following constructor hints are valid:
        - None: Use `cls` as constructor.
        - An already instantiated object: Will be returned as is; no constructor call.
        - A string or an object that is a key in `cls`'s `__lookup_classes__` dict: The value in `__lookup_classes__`
            for that key will be used as the constructor.
        - A python callable: Use that as constructor.
        - A string: Either a json filename or the name of a python module+class (e.g. "rlgraph.components.Component")
            to be Will be used to

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
        ctor_args = ctor_kwargs.pop("_args", list())

        # Figure out the actual constructor (class) from `type_`.
        constructor = None
        # None: Try __default__object (if no args/kwargs), only then constructor of cls (using args/kwargs).
        if type_ is None:
            # We have a default constructor that was defined directly by cls (not by its children).
            if cls.__default_constructor__ is not None and ctor_args == list() and \
                    (not hasattr(cls.__bases__[0], "__default_constructor__") or
                     cls.__bases__[0].__default_constructor__ is None or
                     cls.__bases__[0].__default_constructor__ is not cls.__default_constructor__
                    ):
                constructor = cls.__default_constructor__
                # Default partial's keywords into ctor_kwargs.
                if isinstance(constructor, partial):
                    kwargs = default_dict(ctor_kwargs, constructor.keywords)
                    constructor = partial(constructor.func, **kwargs)
                    ctor_kwargs = dict()  # erase to avoid duplicate kwarg error
            # Try our luck with this class itself.
            else:
                constructor = cls
        # type_ is already a created object of this class -> Take it as is.
        elif isinstance(type_, cls):
            return type_
        # Valid key of cls.__lookup_classes__.
        elif isinstance(cls.__lookup_classes__, dict) and (type_ in cls.__lookup_classes__ or
                 (isinstance(type_, str) and re.sub(r'[\W_]', '', type_.lower()) in cls.__lookup_classes__)):
            constructor = cls.__lookup_classes__.get(type_)
            if constructor is None:
                constructor = cls.__lookup_classes__[re.sub(r'[\W_]', '', type_.lower())]
        # Python callable.
        elif callable(type_):
            constructor = type_
        # A string: Filename or a python module+class.
        elif isinstance(type_, str):
            if re.search(r'\.(yaml|yml|json)$', type_):
                return cls.from_file(type_, *ctor_args, **ctor_kwargs)
            elif type_.find('.') != -1:
                module_name, function_name = type_.rsplit(".", 1)
                module = importlib.import_module(module_name)
                constructor = getattr(module, function_name)
            else:
                raise RLGraphError("ERROR: String specifier ({}) in from_spec must be a filename, a module+class, or "
                                "a key into {}.__lookup_classes__!".format(type_, cls.__name__))

        if not constructor:
            raise RLGraphError("Invalid type: {}".format(type_))

        obj = constructor(*ctor_args, **ctor_kwargs)
        assert isinstance(obj, constructor.func if isinstance(constructor, partial) else constructor)

        return obj

    @classmethod
    def from_file(cls, filename, *args, **kwargs):
        """
        Create object from spec saved in filename. Expects json or yaml format.

        Args:
            filename: file containing the spec (json or yaml)

        Keyword Args:
            Used as additional parameters for call to constructor.

        Returns:
            object
        """
        path = os.path.join(os.getcwd(), filename)
        if not os.path.isfile(path):
            raise RLGraphError('No such file: {}'.format(filename))

        with open(path, 'rt') as fp:
            if path.endswith('.yaml') or path.endswith('.yml'):
                spec = yaml.load(fp)
            else:
                spec = json.load(fp)

        # Add possible *args.
        spec["_args"] = args
        return cls.from_spec(spec=spec, **kwargs)

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
            raise RLGraphError('Invalid input to `from_mixed`: {}'.format(mixed))


class SpaceInfoCarrier(object):
    """
    A mix-in placeholder class whose children must implement the get_space_spec method.
    """
    def get_space_spec(self, method_name):
        """
        Returns a tuple of Space objects for a given method_name. This method, when called, must return values
        that are members of the specified Spaces.

        Args:
            method_name (str): The method name (member of `self`) for which we would like output Space
                information.

        Returns:
            Tuple[Space]: A tuple of Space objects matching the return values when calling the given method.
        """
        raise NotImplementedError


class SpecifiableServer(object):
    """
    A class that creates a separate python process ("server") which runs an arbitrary Specifiable object
    (wrapped as a `SpaceInfoCarrier` object to ascertain an API to get Space- and dtype-specs for
    the Specifiable).

    This is useful - for example - to run RLgraph Environments (which are Specifiables) in a highly parallelized and
    in-graph fashion for faster Agent-Environment stepping.
    """

    COLLECTION = "rlgraph_specifiable_server"

    def __init__(self, spec, shutdown_method=None):
        """
        Args:
            spec (dict): The specification dict that will be used to construct the Specifiable.
            shutdown_method (Optional[str]): An optional name of a shutdown method that will be called on the
                Specifiable object before "server" shutdown to give the Specifiable a chance to clean up.
                The Specifiable must implement this method.
        """
        self.spec = spec
        self.shutdown_method = shutdown_method

        self.class_ = None  # TODO: get class from spec already here

        # The process in which the Specifiable will run.
        self.process = None
        # The out-pipe to send commands (method calls) to the server process.
        self.out_pipe = None
        # The in-pipe to receive "ready" signal from the server process.
        self.in_pipe = None

        # Register this process in a special collection so we can shut it down once the tf.Session ends
        # (via session hook).
        if get_backend() == "tf":
            tf.add_to_collection(SpecifiableServer.COLLECTION, self)

    def __getattr__(self, method_name):
        """
        Returns a function that will create a server-call (given method_name must be one of the Specifiable object)
        from within the backend-specific graph.

        Args:
            method_name (str): The method to call on the Specifiable.

        Returns:

        """
        def call(*args):
            specs = self.class_.get_space_spec(method_name)

            if specs is None:
                raise RLGraphError("No Space information received for method '{}:{}'".format(self.class_.__name__,
                                                                                             method_name))

            dtypes = [space.dtype for space in specs]
            shapes = [space.shape for space in specs]

            if get_backend() == "tf":
                # This function will send the method-call-comment via the out-pipe to the remote (server) Specifiable
                # object - all in-graph - and return the results to be used further by other graph ops.
                def py_call(*args_):
                    try:
                        self.out_pipe.send(args_)
                        result_ = self.out_pipe.recv()
                        if isinstance(result_, Exception):
                            raise result_
                        elif result_ is not None:
                            return result_
                    except Exception as e:
                        if isinstance(e, IOError):
                            raise StopIteration()  # Clean exit.
                        else:
                            raise
                result = tf.py_func(py_call, (method_name,) + tuple(args), dtypes, name=method_name)
                # If we returned a tf op: Return it here.
                # TODO: this should be supportive of a mix of many different op types including tensors.
                if isinstance(result, tf.Operation):
                    return result

                # If tensors: Force shapes that we already know.
                for tensor, shape in zip(result, shapes):
                    tensor.set_shape(shape)

            else:
                raise NotImplementedError

            return result

        return call

    def start(self):
        # Create the in- and out- pipes to communicate with the proxy-Specifiable.
        self.out_pipe, self.in_pipe = multiprocessing.Pipe()
        # Create and start the process passing it the spec to construct the desired Specifiable object..
        self.process = multiprocessing.Process(self.run, args=(self.spec, self.in_pipe))
        self.process.start()

        # Wait for the "ready" signal.
        result = self.out_pipe.recv()

        # Check whether there were construction errors.
        if isinstance(result, Exception):
            raise result

    def close(self):  #, session):
        try:
            self.out_pipe.send(None)
            self.out_pipe.close()
        except IOError:
            pass
        self.process.join()

    @staticmethod
    def run(spec, in_pipe, shutdown_method=None):
        proxy_object = None
        try:
            # Construct the Specifiable object.
            proxy_object = Specifiable.from_spec(spec)

            # Send the ready signal (no errors).
            in_pipe.send(None)

            # Start a server-loop waiting for method call requests.
            while True:
                command = in_pipe.recv()

                # "close" signal (None) -> End this process.
                if command is None:
                    # Give the proxy_object a chance to clean up via some `shutdown_method`.
                    if shutdown_method is not None and hasattr(proxy_object, shutdown_method):
                        getattr(proxy_object, shutdown_method)()
                    in_pipe.close()
                    return

                # Call the method with the given args.
                method_name = str(command[0])
                inputs = command[1:]
                results = getattr(proxy_object, method_name)(*inputs)

                # Send return values back to caller.
                in_pipe.send(results)

        # If something happens during the construction and proxy run phase, pass the exception back through our pipe.
        except Exception as e:
            # Try to clean up.
            if proxy_object is not None and shutdown_method is not None and hasattr(proxy_object, shutdown_method):
                try:
                    getattr(proxy_object, shutdown_method)()
                except:
                    pass
            # Send the exception back so the main process knows what's going on.
            in_pipe.send(e)


class SpecifiableServerHook(tf.train.SessionRunHook):
    """
    A hook for a tf.MonitoredSession that takes care of automatically starting and stopping
    SpecifiableServer objects.
    """
    def begin(self):
        """
        Starts all registered RLGraphProxyProcess processes.
        """
        tp = multiprocessing.pool.ThreadPool()
        tp.map(lambda rlgraph_proxy_process: rlgraph_proxy_process.start(),
               tf.get_collection(SpecifiableServer.COLLECTION))
        tp.close()
        tp.join()

    def end(self, session):
        tp = multiprocessing.pool.ThreadPool()
        tp.map(lambda rlgraph_proxy_process: rlgraph_proxy_process.close(),
               tf.get_collection(SpecifiableServer.COLLECTION))
        tp.close()
        tp.join()

