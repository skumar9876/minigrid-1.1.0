from __future__ import annotations

from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, MultiColorGoal, Floor, Key, Ball, Box
import numpy as np


class ChainedMultiroomEnv(MiniGridEnv):

    """
    ## Description

    Reinforcement learning environment with multiple rooms, 
    each containing a different terminal goal state. Goal states 
    that are in rooms further to the right have larger reward.

    ## Mission Space

    "reach the goals"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of 'i / num_goals' is given for the ith goal. Step cost is -0.01.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches one of the goals.
    2. Timeout (see `max_steps`).

    """

    def __init__(self, max_steps=100, **kwargs):
        self._agent_default_pos = (1, 1)

        # self.size = 10
        self.num_goals = 3
        self.num_rooms = 3
        self.room_width = 7
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=self.num_rooms * self.room_width + self.num_rooms + 1,
            height=self.room_width + 2,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # for i in range(1, self.room_width + 1):
        #     for j in range(0, height):
        #         self.grid.set(i, j, Floor("yellow"))
        
        # for i in range(self.room_width + 2, 2 * self.room_width + 2):
        #     for j in range(0, height):
        #         self.grid.set(i, j, Floor("green"))
        
        # for i in range(2 * self.room_width + 3,  3 * self.room_width + 3):
        #     for j in range(0, height):
        #         self.grid.set(i, j, Floor("red"))
        
        objects = [Key, Ball, Box]
        
        object_colors = {0: ["red", "grey", "blue", "grey"], 
                         1: ["grey", "purple", "green", "purple"], 
                         2: ["blue", "red", "yellow", "red"]}
        for i in range(3):
            x = (i+1) + int(self.room_width / 2) + i * int(self.room_width)
            y = 1 + int(self.room_width / 2)

            object = objects[i]
            self.grid.set(x - 1, y - 1, object(object_colors[i][0]))
            self.grid.set(x - 1, y + 2, object(object_colors[i][1]))
            self.grid.set(x + 2, y + 2, object(object_colors[i][2]))
            self.grid.set(x + 2, y - 1, object(object_colors[i][3]))

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Generate the corridor walls
        self.grid.vert_wall(self.room_width + 1, 2, self.room_width)
        self.grid.vert_wall(2 * self.room_width + 2, 2, self.room_width)


        # Randomize the player start orientation
        self.agent_pos = self._agent_default_pos
        self.grid.set(*self._agent_default_pos, None)
        # assuming random start direction
        self.agent_dir = self._rand_int(0, 4)

        # Create and set the goals.
        goals = [MultiColorGoal("goal1", "red"),
                 MultiColorGoal("goal2", "green"), 
                 MultiColorGoal("goal3", "blue")]
        
        goal_positions = [(self.room_width, self.room_width),
                          (2 * self.room_width + 1, self.room_width),
                          (3 * self.room_width + 2, self.room_width)]

        
        for i in range(len(goals)):
            goal = goals[i]
            goal_pos = goal_positions[i]
            self.put_obj(goal, *goal_pos)
            goal.init_pos, goal.cur_pos = goal_pos

    def step(self, action):
        self.step_count += 1

        reward = -0.01  # Step cost.
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell is not None and "goal" in fwd_cell.type:
                done = True
                if fwd_cell.type == "goal1":
                    reward = 0.1
                elif fwd_cell.type == "goal2":
                    reward = 0.5
                elif fwd_cell.type == "goal3":
                    reward = 1.
            if fwd_cell is not None and fwd_cell.type == "lava":
                done = True
        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}