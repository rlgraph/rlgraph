# Copyright 2018 The RLgraph authors, All Rights Reserved.
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

import multiprocessing

from rlgraph.utils.rlgraph_error import RLGraphError
from rlgraph import get_backend
from rlgraph.spaces.space import Space
from rlgraph.spaces.containers import ContainerSpace
from rlgraph.utils.specifiable import Specifiable
from rlgraph.utils.util import force_list, dtype

if get_backend() == "tf":
    import tensorflow as tf


class SpecifiableServer(Specifiable):
    """
    A class that creates a separate python process ("server") which runs an arbitrary Specifiable object.

    This is useful - for example - to run RLgraph Environments (which are Specifiables) in a highly parallelized and
    in-graph fashion for faster Agent-Environment stepping.
    """

    # Class instances get registered/deregistered here.
    INSTANCES = list()

    def __init__(self, class_, spec, output_spaces, shutdown_method=None):
        """
        Args:
            class_ (type): The class to use for constructing the Specifiable from spec. This class needs to be
                a child class of Specifiable (with a __lookup_classes__ property).
            spec (dict): The specification dict that will be used to construct the Specifiable.
            output_spaces (Union[callable,Dict[str,Space]]): A callable that takes a method_name as argument
                and returns the Space(s) that this method (on the Specifiable object) would return. Alternatively:
                A dict with key=method name and value=Space(s).
            shutdown_method (Optional[str]): An optional name of a shutdown method that will be called on the
                Specifiable object before "server" shutdown to give the Specifiable a chance to clean up.
                The Specifiable must implement this method.
            #flatten_output_dicts (bool): Whether output dictionaries should be flattened to tuples and then
            #    returned.
        """
        super(SpecifiableServer, self).__init__()

        self.class_ = class_
        self.spec = spec
        # If dict: Process possible specs so we don't have to do this during calls.
        if isinstance(output_spaces, dict):
            self.output_spaces = dict()
            for method_name, space_spec in output_spaces.items():
                if isinstance(space_spec, (tuple, list)):
                    self.output_spaces[method_name] = [Space.from_spec(spec) if spec is not None else
                                                       None for spec in space_spec]
                else:
                    self.output_spaces[method_name] = Space.from_spec(space_spec) if space_spec is not None else None
        else:
            self.output_spaces = output_spaces
        self.shutdown_method = shutdown_method

        # The process in which the Specifiable will run.
        self.process = None
        # The out-pipe to send commands (method calls) to the server process.
        self.out_pipe = None
        # The in-pipe to receive "ready" signal from the server process.
        self.in_pipe = None

        # Register this object with the class.
        self.INSTANCES.append(self)

    def __getattr__(self, method_name):
        """
        Returns a function that will create a server-call (given method_name must be one of the Specifiable object)
        from within the backend-specific graph.

        Args:
            method_name (str): The method to call on the Specifiable.
            #return_slots (Optional[List[int]]): An optional list of return slots to use. None for using all return
            #    values.

        Returns:
            callable: The callable to be executed when getting the given method name (of the Specifiable object
                (running inside the SpecifiableServer).
        """
        def call(*args):
            if isinstance(self.output_spaces, dict):
                assert method_name in self.output_spaces, "ERROR: Method '{}' not specified in output_spaces: {}!".\
                    format(method_name, self.output_spaces)
                specs = self.output_spaces[method_name]
            else:
                specs = self.output_spaces(method_name)

            if specs is None:
                raise RLGraphError(
                    "No Space information received for method '{}:{}'".format(self.class_.__name__, method_name)
                )

            dtypes = list()
            shapes = list()
            return_slots = list()
            for i, space in enumerate(force_list(specs)):
                assert not isinstance(space, ContainerSpace)
                # Expecting an op (space 0).
                if space == 0:
                    dtypes.append(0)
                    shapes.append(0)
                    return_slots.append(i)
                # Expecting a tensor.
                elif space is not None:
                    dtypes.append(dtype(space.dtype))
                    shapes.append(space.shape)
                    return_slots.append(i)

            if get_backend() == "tf":
                # This function will send the method-call-comment via the out-pipe to the remote (server) Specifiable
                # object - all in-graph - and return the results to be used further by other graph ops.
                def py_call(*args_):

                    def quick_flatten_dict(dict_, start_slot):
                        ret = tuple(res for slot, (key, res) in enumerate(sorted(dict_.items())) if slot+start_slot in
                                    return_slots)
                        #print("after quick flatten dict: returning {}".format(ret))
                        return ret

                    try:
                        #print("Sending args '{}' through pipe!".format(args))
                        self.out_pipe.send(args_)
                        result_ = self.out_pipe.recv()
                        #print("Received results_ '{}'.".format(result_))

                        # If an error occurred, it'll be passed back through the pipe.
                        if isinstance(result_, Exception):
                            raise result_
                        # Dict output. Flatten to tuple, then filter out the return
                        # values according to return_slots.
                        elif isinstance(result_, dict):
                            return quick_flatten_dict(result_, 0)
                        # Regular result. Filter out the return values according to return_slots.
                        elif isinstance(result_, tuple):
                            #return tuple((np.asarray(r, dtype=np.float32, order="C") if type(r) == float else r) for slot, r in enumerate(result_) if slot in return_slots)
                            #print("return slots={}".format(return_slots))
                            ret = list()
                            slot = 0
                            for i, r in enumerate(result_):
                                if slot in return_slots:
                                    if not isinstance(r, dict):
                                        #print("no dict, append {}".format(r))
                                        ret.append(r)
                                        slot += 1
                                    else:
                                        #print("QUICK flattening. {}".format(r))
                                        flat_dict = quick_flatten_dict(r, slot)
                                        ret.extend(flat_dict)
                                        slot += len(flat_dict)
                                else:
                                    #print("skipped slot={}".format(slot))
                                    slot += 1
                            #print("returning {}".format(ret))
                            return tuple(ret)
                        else:
                            #return np.asarray(result_, dtype=np.float32, order="C") if type(result_) == float else result_
                            return result_
                    except Exception as e:
                        if isinstance(e, IOError):
                            raise StopIteration()  # Clean exit.
                        else:
                            raise
                results = tf.py_func(py_call, (method_name,) + tuple(args), dtypes, name=method_name)

                # Force known shapes on the returned tensors.
                for i, (result, shape) in enumerate(zip(results, shapes)):
                    # Not an op (which have shape=0).
                    if shape != 0:
                        result.set_shape(shape)
            else:
                raise NotImplementedError

            return results[0] if len(dtypes) == 1 else tuple(results)

        return call

    def start(self):
        # Create the in- and out- pipes to communicate with the proxy-Specifiable.
        self.out_pipe, self.in_pipe = multiprocessing.Pipe()
        # Create and start the process passing it the spec to construct the desired Specifiable object..
        self.process = multiprocessing.Process(
            target=self.run, args=(self.class_, self.spec, self.in_pipe, self.shutdown_method)
        )
        self.process.start()

        # Wait for the "ready" signal (which is None).
        result = self.out_pipe.recv()

        # Check whether there were construction errors.
        if isinstance(result, Exception):
            raise result

    def stop(self):  #, session):
        try:
            self.out_pipe.send(None)
            self.out_pipe.close()
        except IOError:
            pass
        self.process.join()

    @staticmethod
    def run(class_, spec, in_pipe, shutdown_method=None):
        proxy_object = None
        try:
            # Construct the Specifiable object.
            print("SpecifiableServer: Constructing Specifiable object. ...")
            proxy_object = class_.from_spec(spec)
            print("SpecifiableServer: Done constructing Specifiable object '{}'. Sending 'ready' signal "
                  "...".format(proxy_object))

            # Send the ready signal (no errors).
            in_pipe.send(None)

            # Start a server-loop waiting for method call requests.
            while True:
                command = in_pipe.recv()

                # "close" signal (None) -> End this process.
                if command is None:
                    # Give the proxy_object a chance to clean up via some `shutdown_method`.
                    if shutdown_method is not None and hasattr(proxy_object, shutdown_method):
                        print("SpecifiableServer: Calling shutdown method '{}'. ...".format(shutdown_method))
                        getattr(proxy_object, shutdown_method)()
                    print("SpecifiableServer: Closing pipe.")
                    in_pipe.close()
                    return

                # Call the method with the given args.
                method_name = command[0].decode()  # must decode here as method_name comes in as bytes
                # print(method_name + " is called on Server.")
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


if get_backend() == "tf":
    class SpecifiableServerHook(tf.train.SessionRunHook):
        """
        A hook for a tf.MonitoredSession that takes care of automatically starting and stopping
        SpecifiableServer objects.
        """
        def __init__(self):
            self.specifiable_buffer = list()

        def begin(self):
            """
            Starts all registered RLGraphProxyProcess processes.
            """
            #tp = multiprocessing.pool.ThreadPool()
            #tp.map(lambda server: server.start(), SpecifiableServer.INSTANCES)

            for server in SpecifiableServer.INSTANCES:
                server.start()

            # Erase all SpecifiableServers as we open the Session (after having started all of them),
            # so new ones can get registered.
            self.specifiable_buffer = SpecifiableServer.INSTANCES[:]  # deepcopy items
            SpecifiableServer.INSTANCES.clear()

            #tp.close()
            #tp.join()

        def end(self, session):
            #tp = multiprocessing.pool.ThreadPool()
            #tp.map(lambda server: server.stop(), self.specifiable_buffer)
            #tp.close()
            #tp.join()
            for server in SpecifiableServer.INSTANCES:
                server.close()



