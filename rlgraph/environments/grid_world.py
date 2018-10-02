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

import math
import numpy as np
import random
from six.moves import xrange as range_
import time

from rlgraph.environments import Environment
import rlgraph.spaces as spaces


class GridWorld(Environment):
    """
    A classic grid world where the action space is up,down,left,right and the
    field types are:
    'S' : starting point
    ' ' : free space
    'W' : wall (blocks)
    'H' : hole (terminates episode) (to be replaced by W in save-mode)
    'F' : fire (usually causing negative reward)
    'G' : goal state (terminates episode)
    TODO: Create an option to introduce a continuous action space.
    """
    # Some built-in maps.
    MAPS = {
        "chain": [
            "G    S  F G"
        ],
        "2x2": [
            "SH",
            " G"
        ],
        "4x4": [
            "S   ",
            " H H",
            "   H",
            "H  G"
        ],
        "8x8": [
            "S       ",
            "        ",
            "   H    ",
            "     H  ",
            "   H    ",
            " HH   H ",
            " H  H H ",
            "   H   G"
        ],
        "8x16": [
            "S      H        ",
            "   H       HH   ",
            "    FF   WWWWWWW",
            "  H      W      ",
            "    FF   W  H   ",
            "         W      ",
            "    FF   W      ",
            "  H          H G"
        ],
        "16x16": [
            "S      H        ",
            "           HH   ",
            "    FF   W     W",
            "         W      ",
            "WWW FF      H   ",
            "         W      ",
            " FFFF    W      ",
            "  H          H  ",
            "       H        ",
            "   H       HH   ",
            "WWWW     WWWWWWW",
            "  H      W    W ",
            "    FF   W  H W ",
            "WWWW    WW    W ",
            "    FF   W      ",
            "  H          H G"
        ]
    }

    def __init__(self, world="4x4", save_mode=False, reward_function="sparse", state_representation="discr"):
        """
        Args:
            world (Union[str,List[str]]): Either a string to map into `MAPS` or a list of strings describing the rows
                of the world (e.g. ["S ", " G"] for a two-row/two-column world with start and goal state).

            save_mode (bool): Whether to replace holes (H) with walls (W). Default: False.

            reward_function (str): One of
                sparse: hole=-1, fire=-1, goal=50, all other steps=-1
                rich: hole=-100, fire=-10, goal=50

            state_representation (str): One of "discr_pos", "xy_pos", "cam"
        """
        # Build our map.
        if isinstance(world, str):
            self.description = world
            world = self.MAPS[world]
        else:
            self.description = "custom-map"

        world = np.array(list(map(list, world)))
        # Apply safety switch.
        world[world == 'H'] = ("H" if not save_mode else "F")

        # `world` is a list of lists that needs to be indexed using y/x pairs (first row, then column).
        self.world = world
        self.n_row, self.n_col = self.world.shape
        (start_x,), (start_y,) = np.nonzero(self.world == "S")

        # Figure out our state space.
        assert state_representation in ["discr", "xy", "cam"]
        self.state_representation = state_representation
        # Discrete states (single int from 0 to n).
        if self.state_representation == "discr":
            state_space = spaces.IntBox(self.n_row * self.n_col)
        # x/y position (2 ints).
        elif self.state_representation == "xy_pos":
            state_space = spaces.IntBox(low=(0, 0), high=(self.n_col, self.n_row), shape=(2,))
        # Camera outputting a 2D color image of the world.
        else:
            state_space = spaces.IntBox(0, 255, shape=(self.n_row, self.n_col, 3))

        self.default_start_pos = self.get_discrete_pos(start_x, start_y)
        self.discrete_pos = self.default_start_pos

        assert reward_function in ["sparse", "rich"]  # TODO: "potential"-based reward
        self.reward_function = reward_function

        # Store the goal position for proximity calculations (for "potential" reward function).
        (self.goal_x,), (self.goal_y,) = np.nonzero(self.world == "G")

        # Call the super's constructor.
        super(GridWorld, self).__init__(state_space=state_space, action_space=spaces.IntBox(4))

        # Reset ourselves.
        self.state = None
        self.camera_pixels = None  # only used, if state_representation=='cam'
        self.reward = None
        self.is_terminal = None
        self.reset(randomize=False)

    def seed(self, seed=None):
        if seed is None:
            seed = time.time()
        np.random.seed(seed)
        return seed

    def reset(self, randomize=False):
        """
        Args:
            randomize (bool): Whether to start the new episode in a random position (instead of "S").
                This could be an empty space (" "), the default start ("S") or a fire field ("F").
        """
        if randomize is False:
            self.discrete_pos = self.default_start_pos
        else:
            # Move to a random first position (" ", "S", or "F" (ouch!) are all ok to start in).
            while True:
                self.discrete_pos = random.choice(range(self.n_row * self.n_col))
                if self.world[self.y, self.x] in [" ", "S", "F"]:
                    break

        self.reward = 0.0
        self.is_terminal = False
        self.refresh_state()
        return self.state

    def step(self, actions, set_discrete_pos=None):
        """
        Action map:
        0: up
        1: right
        2: down
        3: left

        Args:
            actions (int): An integer 0-3 that describes the next action.
            set_discrete_pos (Optional[int]): An integer to set the current discrete position to before acting.

        Returns:
            tuple: State Space (Space), reward (float), is_terminal (bool), info (usually None).
        """
        # Process possible manual setter instruction.
        if set_discrete_pos is not None:
            assert isinstance(set_discrete_pos, int) and 0 <= set_discrete_pos < self.state_space.flat_dim
            self.discrete_pos = set_discrete_pos

        # then perform an action
        possible_next_positions = self.get_possible_next_positions(self.discrete_pos, actions)
        # determine the next state based on the transition function
        probs = [x[1] for x in possible_next_positions]
        next_state_idx = np.random.choice(len(probs), p=probs)
        self.discrete_pos = possible_next_positions[next_state_idx][0]

        next_x = self.discrete_pos // self.n_col
        next_y = self.discrete_pos % self.n_col

        # determine reward and done flag
        next_state_type = self.world[next_y, next_x]
        if next_state_type == "H":
            self.is_terminal = True
            self.reward = -5 if self.reward_function == "sparse" else -10
        elif next_state_type == "F":
            self.is_terminal = False
            self.reward = -3 if self.reward_function == "sparse" else -10
        elif next_state_type in [" ", "S"]:
            self.is_terminal = False
            self.reward = -1
        elif next_state_type == "G":
            self.is_terminal = True
            self.reward = 1 if self.reward_function == "sparse" else 50
        else:
            raise NotImplementedError

        self.refresh_state()

        return self.state, self.reward, self.is_terminal, None

    def render(self):
        # paints itself
        for row in range_(len(self.world)):
            for col, val in enumerate(self.world[row]):
                if self.x == col and self.y == row:
                    print("X", end="")
                else:
                    print(val, end="")
            print()
        print()

    def __str__(self):
        return "GridWorld({})".format(self.description)

    def refresh_state(self):
        if self.state_representation == "discr":
            self.state = self.discrete_pos
        elif self.state_representation == "xy_pos":
            self.state = (self.x, self.y)
        # Camera.
        else:
            self.update_cam_pixels()
            self.state = self.camera_pixels

    def get_possible_next_positions(self, discrete_pos, action):
        """
        Given a discrete position value and an action, returns a list of possible next states and
        their probabilities. Only next states with non-zero probabilities will be returned.
        For now: Implemented as a deterministic MDP.

        Args:
            discrete_pos (int): The discrete position to return possible next states for.
            action (int): The action choice.

        Returns:
            List[Tuple[int,float]]: A list of tuples (s', p(s'\|s,a)). Where s' is the next discrete position and
                p(s'\|s,a) is the probability of ending up in that position when in state s and taking action a.
        """
        x = discrete_pos // self.n_col
        y = discrete_pos % self.n_col
        coords = np.array([x, y])

        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_coords = np.clip(
            coords + increments[action],
            [0, 0],
            [self.n_row - 1, self.n_col - 1]
        )
        next_pos = self.get_discrete_pos(next_coords[0], next_coords[1])
        pos_type = self.world[y, x]
        next_pos_type = self.world[next_coords[1], next_coords[0]]
        # TODO: Allow stochasticity in this env. Right now, all probs are 1.0.
        # Next field is a wall or we are already terminal. Stay where we are.
        if next_pos_type == "W" or pos_type in ["H", "G"]:
            return [(discrete_pos, 1.)]
        # Move to next field.
        else:
            return [(next_pos, 1.)]

    def update_cam_pixels(self):
        # Init camera?
        if self.camera_pixels is None:
            self.camera_pixels = np.zeros(shape=(self.n_row, self.n_col, 3), dtype=float)
        self.camera_pixels[:, :, :] = 0  # reset everything

        # 1st channel -> walls (127) and goal (255)
        # 2nd channel -> dangers (fire=127, holes=255)
        # 3rd channel -> pawn position (255)
        for row in range_(self.n_row):
            for col in range_(self.n_col):
                field = self.world[row, col]
                if field == "F":
                    self.camera_pixels[row, col, 0] = 127
                elif field == "H":
                    self.camera_pixels[row, col, 0] = 255
                elif field == "W":
                    self.camera_pixels[row, col, 1] = 127
                elif field == "G":
                    self.camera_pixels[row, col, 1] = 255  # will this work (goal==2x wall)?
        # Overwrite player's position.
        self.camera_pixels[self.y, self.x, 2] = 255

    def get_dist_to_goal(self):
        return math.sqrt((self.x - self.goal_x) ** 2 + (self.y - self.goal_y) ** 2)

    def get_discrete_pos(self, x, y):
        """
        Returns a single, discrete int-value.
        Calculated by walking down the rows of the grid first (starting in upper left corner),
        then along the col-axis.

        Args:
            x (int): The x-coordinate.
            y (int): The y-coordinate.

        Returns:
            int: The discrete pos value corresponding to the given x and y.
        """
        return x * self.n_col + y

    @property
    def x(self):
        return self.discrete_pos // self.n_col

    @property
    def y(self):
        return self.discrete_pos % self.n_col

