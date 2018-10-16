.. Copyright 2018 The RLgraph authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   ============================================================================

.. image:: images/rlcore-logo-full.png
   :scale: 25%
   :alt:

The Environment Classes
=======================

What is an environment?
-----------------------

The generic reinforcement learning problem can be broken down into the following ever repeating sequence of steps:

- An agent that "lives" in some environment observes the current state of that environment. The state could be \
   anything from something simple like the x/y position of the agent to an image of a camera (that the agent has \
   access to) or a line of text from the agent's chat partner (think: the agent is a chat bot). The nature of this \
   state signal (its data type and shape) is called the "state space".

- Based on that state observation, the agent now picks an action. The environment dictates from which space this action may be chosen. For example, one could think of a computer game, in which an agent has to move through a 2D maze and can pick the actions up, down, left, and right at each time step. This space from which to chose is called the "action space". In RLgraph, both state- and action space are usually given by the environment.
- The chosen action is now executed on the environment (e.g. the agent decided to move left) and the environment changes because of that action. This change is described by the transition function :math:`P(s'|s,a)`, which outputs the probability for ending up in next state `s'` given that the agent chose action `a` after having observed state `s`.
-

RLgraph's environment adapters.
-------------------------------

RLGraph supports many popular environment types and standards and offers a common interface into all these.
The base class is the `Environment` and its most important API-methods are `reset` (to reset the environment) and `step`
(to execute an action).
In the following, we will briefly describe the different supported environment types. If you are interested in
writing your own environments for your own RL research, we will be very happy to receive your pull request.
For more information on our environments, see the
`environment reference documentation <reference/environments/>`_.

OpenAI Gym
++++++++++

The `OpenAI Gym <https://gym.openai.com/envs/>`_ standard is the most widely used type of environment in reinforcement
learning research. It contains the famous set of Atari 2600 games (each game has a RAM state- and a 2D image version),
simple text-rendered grid-worlds, a set of robotics tasks, continuous control tasks (via the MuJoCO physics simulator),
and many others.

.. image:: images/mujoco_environment.png
    :alt: The "Ant-v2" environment of the many MuJoCo-simulator tasks of the OpenAI Gym.

RLgraph's OpenAIGymEnv class serves as an adapter between RLgraph code and any of these openAI Gym
environments. For example, in order to have your agent learn how to play Breakout from image pixels, you would create
the environment under RLgraph via:

.. code-block:: python

    from rlgraph.environments import OpenAIGymEnv
    # Create the env.
    breakout_env = OpenAIGymEnv("Breakout-v0", visualize=True)
    # Reset the env.
    breakout_env.reset()
    # Execute 100 random actions in the env.
    for _ in range(100):
        state, reward, is_terminal, info = breakout_env.step(breakout_env.action_space.sample())
        # Reset if terminal state was reached.
        if is_terminal:
            breakout_env.reset()



Deepmind Lab
++++++++++++

`Deepmind Lab <http://https://github.com/deepmind/lab>`_ is Google DeepMind's environment of choice for their advanced
RL research. It's a fully customizable suite of 3D environments (including mazes and other interesting worlds),
which are usually navigated by the agent through a 1st person's perspective.

.. image:: images/dm_lab_environment.png
    :alt: The "Nav Maze Arena" environment of the DM Lab.

Different state observation items can be configured as needed at environment construction time, e.g. an image
capturing the 1st person view from inside the
maze or a textual input offering instructions on where to go next (e.g. "blue ladder").
When using more than one state observation items, the Rlgraph state space will be a Dict with the keys describing the
nature of the different observation items (e.g. "RGB_INTERLEAVED" for an RGB image, "INSTR" for the instruction string).

DM Lab itself (and hence also its RLgraph adapter) is somewhat hard to install and only runs on Linux and Mac.
For details, you can take a look at our
`Docker file <https://github.com/rlgraph/rlgraph/blob/master/docker/Dockerfile>`_ to see which steps are required in
order to get it up and running.
