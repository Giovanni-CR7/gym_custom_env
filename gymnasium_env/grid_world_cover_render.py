# gymnasium_env/grid_world_cover_render.py

import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import collections 

class GridWorldCoverRenderEnv(gym.Env):
    """
    Ambiente de Grid World que gera um mapa aleatório e calcula a área acessível
    a cada reset, garantindo que a tarefa seja sempre solucionável.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, size=20, render_mode=None, num_obstacles=40, max_steps=None, seed=None):
        super().__init__()
        self.size = int(size)
        self.window_size = 512
        self.num_obstacles = int(num_obstacles)
        if self.num_obstacles >= self.size * self.size:
            raise ValueError(f"Número de obstáculos ({self.num_obstacles}) excede o tamanho do grid ({self.size*self.size}).")
        
        self._obstacle_locations = set()
        self.visited_cells = set() # Usado primariamente para renderização
        
        # Limite padrão generoso para exploração inicial
        self.max_steps = max_steps if max_steps is not None else int(size * size * 4) # 1600

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4, self.size, self.size), dtype=np.float32)

        self._action_to_direction = {
            0: np.array([1, 0]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([0, -1]),
        }

        self.render_mode = render_mode
        self.window, self.clock = None, None
        self._seed = seed
        self.np_random = np.random.RandomState(seed)
        self._agent_location = None
        self.steps = 0

    def seed(self, seed=None):
        super().seed(seed)
        self._seed = seed
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _get_obs(self):
        obs = np.zeros((4, self.size, self.size), dtype=np.float32)
        obs[0] = np.ones((self.size, self.size), dtype=np.float32)
        for r, c in self._obstacle_locations:
            if 0 <= r < self.size and 0 <= c < self.size:
                obs[0, r, c] = 0.0
                obs[1, r, c] = 1.0
        r_agent, c_agent = self._agent_location
        if 0 <= r_agent < self.size and 0 <= c_agent < self.size:
             obs[2, r_agent, c_agent] = 1.0
        for r, c in self.visited_cells:
            if 0 <= r < self.size and 0 <= c < self.size:
                 obs[3, r, c] = 1.0
        return obs

    def _get_info(self):
        return {
            "size": self.size,
            "obstacles": self._obstacle_locations,
            "agent_location": tuple(self._agent_location)
        }

    # --- MÉTODO RESET  ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Gera o mapa (obstáculos e agente) uma única vez
        self._obstacle_locations = set()
        self.steps = 0
        while len(self._obstacle_locations) < self.num_obstacles:
            pos = (self.np_random.randint(0, self.size), self.np_random.randint(0, self.size))
            self._obstacle_locations.add(pos)
        
        start_pos = None
        while start_pos is None:
            potential_start = (self.np_random.randint(0, self.size), self.np_random.randint(0, self.size))
            if potential_start not in self._obstacle_locations:
                self._agent_location = np.array(potential_start, dtype=int)
                start_pos = tuple(self._agent_location)

        # 2. Executa BFS uma única vez para encontrar todas as células alcançáveis
        q = collections.deque([start_pos])
        reachable_cells = {start_pos}
        while q:
            r, c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if (0 <= nr < self.size and 0 <= nc < self.size and
                        neighbor not in self._obstacle_locations and
                        neighbor not in reachable_cells):
                    reachable_cells.add(neighbor)
                    q.append(neighbor)
        
        # 3. Prepara o estado final e o dicionário de info
        self.visited_cells = {start_pos}
        observation = self._get_obs()
        info = self._get_info()
        
        # --- INFORMA A ÁREA REALMENTE ACESSÍVEL ---
        info['reachable_cells_count'] = len(reachable_cells)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    # --- FIM DO RESET  ---

    def step(self, action):
        action = int(action)
        direction = self._action_to_direction[action]
        potential_location = self._agent_location + direction
        info = self._get_info()
        info["hit_obstacle"] = False
        target_r, target_c = potential_location
        is_valid_move = (0 <= target_r < self.size and 0 <= target_c < self.size and (target_r, target_c) not in self._obstacle_locations)

        if is_valid_move:
            self._agent_location = potential_location
            self.visited_cells.add(tuple(self._agent_location))
        else:
            if (0 <= target_r < self.size and 0 <= target_c < self.size and (target_r, target_c) in self._obstacle_locations):
                info["hit_obstacle"] = True
        
        self.steps += 1
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        # O Wrapper lida com recompensa, terminação e truncamento
        return observation, 0.0, False, False, info

    def set_visited_cells(self, visited_set):
        self.visited_cells = set(visited_set)

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode is None: return
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption(f"Grid World Coverage {self.size}x{self.size}")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        for r, c in self.visited_cells:
            pygame.draw.rect(canvas, (220, 220, 220), pygame.Rect(c * pix_square_size, r * pix_square_size, pix_square_size, pix_square_size))
        for r, c in self._obstacle_locations:
            pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(c * pix_square_size, r * pix_square_size, pix_square_size, pix_square_size))
        
        if self._agent_location is not None:
            agent_r, agent_c = self._agent_location
            center_x = (agent_c + 0.5) * pix_square_size
            center_y = (agent_r + 0.5) * pix_square_size
            pygame.draw.circle(canvas, (0, 0, 255), (center_x, center_y), pix_square_size / 3)

        for i in range(self.size + 1):
            pygame.draw.line(canvas, (100, 100, 100), (0, i * pix_square_size), (self.window_size, i * pix_square_size), width=1)
            pygame.draw.line(canvas, (100, 100, 100), (i * pix_square_size, 0), (i * pix_square_size, self.window_size), width=1)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

