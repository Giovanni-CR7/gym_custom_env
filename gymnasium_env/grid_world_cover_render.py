import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

class GridWorldCoverRenderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, size=5, render_mode=None, num_obstacles=3, max_steps=None, seed=None):
        super().__init__()
        self.size = int(size)
        self.window_size = 512
        self.num_obstacles = int(num_obstacles)
        self._obstacle_locations = set()
        self.visited_cells = set()
        self.max_steps = max_steps if max_steps is not None else 200

        # Actions: 0 down, 1 right, 2 up, 3 left
        self.action_space = spaces.Discrete(4)
        # Observation: channels = [free, obstacles, agent, visited] -> shape (4, H, W)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4, self.size, self.size), dtype=np.float32)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # RNG / seeding
        self._seed = seed
        self.np_random = np.random.RandomState(seed)

        # internal state
        self._agent_location = None
        self.steps = 0

    def seed(self, seed=None):
        self._seed = seed
        self.np_random = np.random.RandomState(seed)
        random.seed(seed)
        return [seed]

    def _get_obs(self):
        obs = np.zeros((4, self.size, self.size), dtype=np.float32)
        # channel 0: free (1 if free)
        free = np.ones((self.size, self.size), dtype=np.float32)
        for (i, j) in self._obstacle_locations:
            free[i, j] = 0.0
        obs[0] = free
        # channel 1: obstacles mask
        obs[1] = np.zeros((self.size, self.size), dtype=np.float32)
        for (i, j) in self._obstacle_locations:
            obs[1, i, j] = 1.0
        # channel 2: agent one-hot
        ai, aj = tuple(self._agent_location)
        obs[2, ai, aj] = 1.0
        # channel 3: visited mask
        vis = np.zeros((self.size, self.size), dtype=np.float32)
        for (i, j) in self.visited_cells:
            vis[i, j] = 1.0
        obs[3] = vis
        return obs

    def _get_info(self):
        return {"size": self.size, "obstacles": set(self._obstacle_locations)}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.visited_cells = set()
        self._obstacle_locations = set()
        self.steps = 0

        # sample obstacles from RNG
        while len(self._obstacle_locations) < self.num_obstacles:
            pos = (int(self.np_random.randint(0, self.size)), int(self.np_random.randint(0, self.size)))
            self._obstacle_locations.add(pos)

        # pick a start that's not an obstacle
        while True:
            start = (int(self.np_random.randint(0, self.size)), int(self.np_random.randint(0, self.size)))
            if start not in self._obstacle_locations:
                break

        self._agent_location = np.array(start, dtype=int)
        self.visited_cells.add(tuple(self._agent_location))

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action):
        action = int(action)
        direction = self._action_to_direction[action]
        potential = self._agent_location + direction
        i, j = int(potential[0]), int(potential[1])
        info = self._get_info()
        info["hit_obstacle"] = False

        # is move within bounds and not obstacle?
        valid = (0 <= i < self.size) and (0 <= j < self.size) and ((i, j) not in self._obstacle_locations)
        if valid:
            self._agent_location = np.array([i, j], dtype=int)
            self.visited_cells.add((i, j))
        else:
            # invalid move (either out of bounds or obstacle) -> agent stays
            if 0 <= i < self.size and 0 <= j < self.size and (i, j) in self._obstacle_locations:
                info["hit_obstacle"] = True
            # else out-of-bounds -> just stay

        self.steps += 1

        obs = self._get_obs()
        terminated = False
        truncated = False
        reward = 0.0  # neutral at env-level; wrapper will compute reward shaping

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def set_visited_cells(self, visited):
        self.visited_cells = set(visited)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # draw visited cells 
        for pos in self.visited_cells:
            pygame_pos = np.array([pos[1], pos[0]])
            pygame.draw.rect(
                canvas,
                (200, 200, 200),
                pygame.Rect(pix_square_size * pygame_pos, (pix_square_size, pix_square_size)),
            )

        # draw obstacles (black)
        for obstacle in self._obstacle_locations:
            pygame_pos = np.array([obstacle[1], obstacle[0]])
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(pix_square_size * pygame_pos, (pix_square_size, pix_square_size)),
            )

        # draw agent (blue)
        agent_pygame_pos = np.array([self._agent_location[1], self._agent_location[0]])
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (agent_pygame_pos + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # grid lines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

