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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import time

import numpy as np
from six.moves import xrange as range_

import rlgraph.spaces as spaces
from rlgraph.environments import Environment


class GridWorld(Environment):
    """
    A classic grid world.

    Possible action spaces are:
    - up, down, left, right
    - forward/halt/backward + turn left/right/no-turn + jump (or not)

    The state space is discrete.

    Field types are:
    'S' : starting point
    ' ' : free space
    'W' : wall (blocks, but can be jumped)
    'H' : hole (terminates episode) (to be replaced by W in save-mode)
    'F' : fire (usually causing negative reward, but can be jumped)
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
        ],
        "4-room": [  # 30=start state, 79=goal state
            "     W     ",
            "     W     ",
            "        G  ",
            "     W     ",
            "     W     ",
            "W WWWW     ",
            "     WWW WW",
            "     W     ",
            "  S  W     ",
            "           ",
            "     W     "
        ]
    }

    # Some useful class vars.
    grid_world_2x2_preprocessing_spec = [dict(type="reshape", flatten=True, flatten_categories=4)]
    grid_world_4x4_preprocessing_spec = [dict(type="reshape", flatten=True, flatten_categories=16)]
    # Preprocessed state spaces.
    grid_world_2x2_flattened_state_space = spaces.FloatBox(shape=(4,), add_batch_rank=True)
    grid_world_4x4_flattened_state_space = spaces.FloatBox(shape=(16,), add_batch_rank=True)

    def __init__(self, world="4x4", save_mode=False, action_type="udlr",
                 reward_function="sparse", state_representation="discrete"):
        """
        Args:
            world (Union[str,List[str]]): Either a string to map into `MAPS` or a list of strings describing the rows
                of the world (e.g. ["S ", " G"] for a two-row/two-column world with start and goal state).

            save_mode (bool): Whether to replace holes (H) with walls (W). Default: False.

            action_type (str): Which action space to use. Chose between "udlr" (up, down, left, right), which is a
                discrete action space and "ftj" (forward + turn + jump), which is a container multi-discrete
                action space.

            reward_function (str): One of
                sparse: hole=-5, fire=-3, goal=1, all other steps=-0.1
                rich: hole=-100, fire=-10, goal=50, all other steps=-0.1

            state_representation (str):
                - "discrete": An int representing the field on the grid, 0 meaning the upper left field, 1 the one
                    below, etc..
                - "xy": The x and y grid position tuple.
                - "xy+orientation": The x and y grid position tuple plus the orientation (if any) as tuple of 2 values
                    of the actor.
                - "camera": A 3-channel image where each field in the grid-world is one pixel and the 3 channels are
                    used to indicate different items in the scene (walls, holes, the actor, etc..).
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
        (start_y,), (start_x,) = np.nonzero(self.world == "S")

        # Figure out our state space.
        assert state_representation in ["discrete", "xy", "xy+orientation", "camera"]
        self.state_representation = state_representation
        # Discrete states (single int from 0 to n).
        if self.state_representation == "discrete":
            state_space = spaces.IntBox(self.n_row * self.n_col)
        # x/y position (2 ints).
        elif self.state_representation == "xy":
            state_space = spaces.IntBox(low=(0, 0), high=(self.n_col, self.n_row), shape=(2,))
        # x/y position + orientation (3 ints).
        elif self.state_representation == "xy+orientation":
            state_space = spaces.IntBox(low=(0, 0, 0, 0), high=(self.n_col, self.n_row, 1, 1))
        # Camera outputting a 2D color image of the world.
        else:
            state_space = spaces.IntBox(0, 255, shape=(self.n_row, self.n_col, 3))

        self.default_start_pos = self.get_discrete_pos(start_x, start_y)
        self.discrete_pos = self.default_start_pos

        assert reward_function in ["sparse", "rich"]  # TODO: "potential"-based reward
        self.reward_function = reward_function

        # Store the goal position for proximity calculations (for "potential" reward function).
        (self.goal_y,), (self.goal_x,) = np.nonzero(self.world == "G")

        # Specify the actual action spaces.
        self.action_type = action_type
        action_space = spaces.IntBox(4) if self.action_type == "udlr" else spaces.Dict(dict(
            forward=spaces.IntBox(3), turn=spaces.IntBox(3), jump=spaces.IntBox(2)
        ))

        # Call the super's constructor.
        super(GridWorld, self).__init__(state_space=state_space, action_space=action_space)

        # Reset ourselves.
        self.state = None
        self.orientation = None  # int: 0, 90, 180, 270
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
        self.orientation = 0
        self.refresh_state()
        return self.state

    def reset_flow(self, randomize=False):
        return self.reset(randomize=randomize)

    def step(self, actions, set_discrete_pos=None):
        """
        Action map:
        0: up
        1: right
        2: down
        3: left

        Args:
            actions (Optional[int,Dict[str,int]]):
                For "udlr": An integer 0-3 that describes the next action.
                For "ftj": A dict with keys: "turn" (0 (turn left), 1 (no turn), 2 (turn right)), "forward"
                    (0 (backward), 1(stay), 2 (forward)) and "jump" (0 (no jump) and 1 (jump)).

            set_discrete_pos (Optional[int]): An integer to set the current discrete position to before acting.

        Returns:
            tuple: State Space (Space), reward (float), is_terminal (bool), info (usually None).
        """
        # Process possible manual setter instruction.
        if set_discrete_pos is not None:
            assert isinstance(set_discrete_pos, int) and 0 <= set_discrete_pos < self.state_space.flat_dim
            self.discrete_pos = set_discrete_pos

        # Forward, turn, jump container action.
        move = None
        if self.action_type == "ftj":
            actions = self._translate_action(actions)
            # Turn around (0 (left turn), 1 (no turn), 2 (right turn)).
            if "turn" in actions:
                self.orientation += (actions["turn"] - 1) * 90
                self.orientation %= 360  # re-normalize orientation

            # Forward (0=move back, 1=don't move, 2=move forward).
            if "forward" in actions:
                forward = actions["forward"]
                # Translate into classic grid world action (0=up, 1=right, 2=down, 3=left).
                # We are actually moving in some direction.
                if actions["forward"] != 1:
                    if self.orientation == 0 and forward == 2 or self.orientation == 180 and forward == 0:
                        move = 0  # up
                    elif self.orientation == 90 and forward == 2 or self.orientation == 270 and forward == 0:
                        move = 1  # right
                    elif self.orientation == 180 and forward == 2 or self.orientation == 0 and forward == 0:
                        move = 2  # down
                    else:
                        move = 3  # left
            # Up, down, left, right actions.
        else:
            move = actions

        if move is not None:
            # determine the next state based on the transition function
            next_positions = self.get_possible_next_positions(self.discrete_pos, move)
            next_state_idx = np.random.choice(len(next_positions), p=[x[1] for x in next_positions])
            # Update our pos.
            self.discrete_pos = next_positions[next_state_idx][0]

        # Jump? -> Move two fields forward (over walls/fires/holes w/o any damage).
        if self.action_type == "ftj" and "jump" in actions:
            assert actions["jump"] == 0 or actions["jump"] == 1
            if actions["jump"] == 1:
                # Translate into "classic" grid world action (0=up, ..., 3=left) and execute that action twice.
                action = int(self.orientation / 90)
                for i in range(2):
                    # determine the next state based on the transition function
                    next_positions = self.get_possible_next_positions(self.discrete_pos, action, in_air=(i==1))
                    next_state_idx = np.random.choice(len(next_positions), p=[x[1] for x in next_positions])
                    # Update our pos.
                    self.discrete_pos = next_positions[next_state_idx][0]

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
            self.reward = -0.1
        elif next_state_type == "G":
            self.is_terminal = True
            self.reward = 1 if self.reward_function == "sparse" else 50
        else:
            raise NotImplementedError

        self.refresh_state()

        return self.state, np.array(self.reward, dtype=np.float32), np.array(self.is_terminal), None

    def step_flow(self, actions):
        state, reward, terminal, _ = self.step(actions)
        # Flow Env logic.
        if terminal:
            state = self.reset()

        return state, reward, terminal

    def render(self):
        print(self.render_as_txt())

    def render_as_txt(self):
        actor = "X"
        if self.action_type == "ftj":
            actor = "^" if self.orientation == 0 else ">" if self.orientation == 90 else "v" if \
                self.orientation == 180 else "<"

        # paints itself
        txt = ""
        for row in range_(len(self.world)):
            for col, val in enumerate(self.world[row]):
                if self.x == col and self.y == row:
                    txt += actor
                else:
                    txt += val
            txt += "\n"
        txt += "\n"
        return txt

    def __str__(self):
        return "GridWorld({})".format(self.description)

    def refresh_state(self):
        # Discrete state.
        if self.state_representation == "discrete":
            # TODO: If ftj-actions, maybe multiply discrete states with orientation (will lead to x4 state space size).
            self.state = np.array(self.discrete_pos, dtype=np.int32)
        # xy position.
        elif self.state_representation == "xy":
            self.state = np.array([self.x, self.y], dtype=np.int32)
        # xy + orientation (only if `self.action_type` supports turns).
        elif self.state_representation == "xy+orientation":
            orient = [0, 1] if self.orientation == 0 else [1, 0] if self.orientation == 90 else [0, -1] \
                if self.orientation == 180 else [-1, 0]
            self.state = np.array([self.x, self.y] + orient, dtype=np.int32)
        # Camera.
        else:
            self.update_cam_pixels()
            self.state = self.camera_pixels

    def get_possible_next_positions(self, discrete_pos, action, in_air=False):
        """
        Given a discrete position value and an action, returns a list of possible next states and
        their probabilities. Only next states with non-zero probabilities will be returned.
        For now: Implemented as a deterministic MDP.

        Args:
            discrete_pos (int): The discrete position to return possible next states for.
            action (int): The action choice.
            in_air (bool): Whether we are actually in the air right now (ignore if we come from "H" or "W").

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
        if next_pos_type == "W" or (in_air is False and pos_type in ["H", "G"]):
            return [(discrete_pos, 1.)]
        # Move to next field.
        else:
            return [(next_pos, 1.)]

    def update_cam_pixels(self):
        # Init camera?
        if self.camera_pixels is None:
            self.camera_pixels = np.zeros(shape=(self.n_row, self.n_col, 3), dtype=np.int32)
        self.camera_pixels[:, :, :] = 0  # reset everything

        # 1st channel -> Walls (127) and goal (255).
        # 2nd channel -> Dangers (fire=127, holes=255)
        # 3rd channel -> Actor position (255).
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

    def _translate_action(self, actions):
        """
        Maps a single integer action to dict actions. This allows us to compare how
        container actions perform when instead using a large range on a single discrete action by enumerating
        all combinations.

        Args:
            actions Union(int, dict): Maps single integer to different actions.

        Returns:
            dict: Actions dict.
        """
        # If already dict, do nothing.
        if isinstance(actions, dict):
            return actions
        else:
            # Unpack
            if isinstance(actions, (np.ndarray, list)):
                actions = actions[0]
            # 3 x 3 x 2 = 18 actions
            assert 18 > actions >= 0
            # For "ftj": A dict with keys: "turn" (0 (turn left), 1 (no turn), 2 (turn right)), "forward"
            # (0 (backward), 1(stay), 2 (forward)) and "jump" (0 (no jump) and 1 (jump)).
            converted_actions = {}

            # Mapping:
            # 0 = 0 0 0
            # 1 = 0 0 1
            # 2 = 0 1 0
            # 3 = 0 1 1
            # 4 = 0 2 0
            # 5 = 0 2 1
            # 6 = 1 0 0
            # 7 = 1 0 1
            # 8 = 1 1 0
            # 9 = 1 1 1
            # 10 = 1 2 0
            # 11 = 1 2 1
            # 12 = 2 0 0
            # 13 = 2 0 1
            # 14 = 2 1 0
            # 15 = 2 1 1
            # 16 = 2 2 0
            # 17 = 2 2 1

            # Set turn via range:
            if 6 > actions >= 0:
                converted_actions["turn"] = 0
            elif 12 > actions >= 6:
                converted_actions["turn"] = 1
            elif 18 > actions >= 12:
                converted_actions["turn"] = 2

            if actions in [0, 1, 6, 7, 12, 13]:
                converted_actions["forward"] = 0
            elif actions in [2, 3, 8, 9, 14, 15]:
                converted_actions["forward"] = 1
            elif actions in [4, 5, 10, 11, 16, 17]:
                converted_actions["forward"] = 2

            if actions % 2 == 0:
                converted_actions["jump"] = 0
            else:
                converted_actions["jump"] = 1
            return converted_actions
