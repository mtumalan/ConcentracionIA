from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgPartialObsWrapper
import numpy as np
import random


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=19,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size
        self.key_positions = []
        self.lava_positions = []

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Adding walls
        for y in range(1, height - 1):
            self.put_obj(Wall(), width // 2, y)
        
        for x in range(1, width - 1):
            self.put_obj(Wall(), x, height // 2)

        # Create openings in the walls
        openings = [(width // 2, 3), (width // 2, 4), (width // 2, 5), (width // 2, 13), (width // 2, 14), (width // 2, 15), (3, height // 2), (4, height // 2), (5, height // 2), (13, height // 2), (14, height // 2), (15, height // 2)]
        for x, y in openings:
            self.grid.set(x, y, None)

        # Set goal position
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)

        self._place_agent()

        self.mission = 'Reach the goal'

    def _place_agent(self):
        """Place agent in a random location, ensuring it does not overlap with the goal."""
        while True:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            pos = (x, y)

            if self.grid.get(*pos) is None and pos != self.goal_pos:
                self.agent_pos = pos
                self.agent_dir = random.randint(0, 3)
                break
                        
    def reset(self, **kwargs):
        """Reset the environment and place the agent."""
        obs = super().reset(**kwargs)
        self._place_agent()  # Aseguramos colocar el agente después de resetear la grilla
        return obs

    def step(self, action):

        """Perform an action and update the environment."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Penalty for hitting a wall
        if self.grid.get(*self.agent_pos) is not None and not isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = -0.1  # Minor penalty for hitting walls

        # Small penalty for each step to encourage faster goal-seeking
        reward += -0.01

        # Check if the agent reached the goal
        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = 1  # Reward for reaching the goal
            terminated = True

        return obs, reward, terminated, truncated, info

    def calc_state(self, agent_pos, width, height):
        """Calculate a unique state based on the agent's position."""
        return (agent_pos[1] - 1) * (height - 2) + (agent_pos[0] - 1)

    def get_action(self, steps, state, q_table, env):
        """Select an action using epsilon-greedy strategy."""
        eps_end = 1.0
        eps_start = 0.1
        eps_decay = 0.1
        eps_threshold = eps_end - (eps_end - eps_start) * np.exp(-1 * steps / eps_decay)

        # Choose action with exploration/exploitation tradeoff
        if random.random() > eps_threshold:
            return q_table[state].argmax()
        else:
            return random.randint(0, 2)


def main():
    lr = 0.01  # Learning rate
    df = 0.9  # Discount factor
    width = 19
    height = 19
    env = SimpleEnv(render_mode="human")
    env = RGBImgPartialObsWrapper(env)

    env.reset()  # Reset the environment

    prev_state = env.calc_state(env.unwrapped.agent_pos, width, height)

    action_space = 3  # Number of possible actions
    total_states = (width - 2) * (height - 2)

    # Initialize Q-table
    q_table = np.zeros((total_states, action_space))

    terminated = False
    truncated = False

    step = 0  # Step counter

    # Main loop
    while not truncated and not terminated:
        step += 1
        action2take = env.get_action(step, prev_state, q_table, env)
        obs, reward, terminated, truncated, info = env.step(action2take)

        current_state = env.calc_state(env.unwrapped.agent_pos, width, height)

        # Q-learning update rule
        q_table[prev_state, action2take] += lr * (reward + df * q_table[current_state].max() - q_table[prev_state, action2take])
        
        # Print the Q-table after the update
        print(f"Q-table for state {current_state} after update:\n", q_table[current_state])

        prev_state = current_state  # Update the previous state

    env.close()  # Close the environment


if __name__ == "__main__":
    main()