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

    FIX: self.np_random era inicializado como np.random.RandomState, mas
    gymnasium.Env.reset(seed=...) o substitui por um numpy.random.Generator
    moderno, que usa .integers() em vez de .randint().
    Solução: removemos o RandomState manual e usamos apenas a API do Generator.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, size=20, render_mode=None, num_obstacles=40,
                 max_steps=None, seed=None):
        super().__init__()
        self.size = int(size)
        self.window_size = 512
        self.num_obstacles = int(num_obstacles)

        if self.num_obstacles >= self.size * self.size:
            raise ValueError(
                f"Número de obstáculos ({self.num_obstacles}) excede o "
                f"tamanho do grid ({self.size * self.size})."
            )

        self._obstacle_locations: set = set()
        self.visited_cells: set = set()

        self.max_steps = max_steps if max_steps is not None else int(size * size * 4)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(4, self.size, self.size), dtype=np.float32
        )

        self._action_to_direction = {
            0: np.array([1,  0]),
            1: np.array([0,  1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self.render_mode = render_mode
        self.window, self.clock = None, None
        self._seed = seed

        # FIX: não criamos RandomState aqui; deixamos o Gymnasium inicializar
        # self.np_random via super().reset(seed=...) como um Generator moderno.
        # Fazemos um seed inicial para que o env seja utilizável antes do reset.
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._agent_location = None
        self.steps = 0

    # ── helpers de RNG ────────────────────────────────────────────────────────

    def _randint(self, low: int, high: int) -> int:
        """Wrapper unificado: funciona com Generator (novo) e RandomState (legado)."""
        if hasattr(self.np_random, "integers"):
            # numpy.random.Generator  (Gymnasium >= 0.26)
            return int(self.np_random.integers(low, high))
        else:
            # numpy.random.RandomState (fallback legado)
            return int(self.np_random.randint(low, high))

    # ── obs / info ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((4, self.size, self.size), dtype=np.float32)
        obs[0] = 1.0  # canal de espaço livre
        for r, c in self._obstacle_locations:
            if 0 <= r < self.size and 0 <= c < self.size:
                obs[0, r, c] = 0.0
                obs[1, r, c] = 1.0
        r_a, c_a = self._agent_location
        if 0 <= r_a < self.size and 0 <= c_a < self.size:
            obs[2, r_a, c_a] = 1.0
        for r, c in self.visited_cells:
            if 0 <= r < self.size and 0 <= c < self.size:
                obs[3, r, c] = 1.0
        return obs

    def _get_info(self) -> dict:
        return {
            "size": self.size,
            "obstacles": self._obstacle_locations,
            "agent_location": tuple(self._agent_location),
        }

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        # FIX: super().reset() configura self.np_random como Generator moderno
        super().reset(seed=seed)

        self._obstacle_locations = set()
        self.steps = 0

        # Gera obstáculos
        while len(self._obstacle_locations) < self.num_obstacles:
            pos = (self._randint(0, self.size), self._randint(0, self.size))
            self._obstacle_locations.add(pos)

        # Posição inicial livre
        start_pos = None
        while start_pos is None:
            candidate = (self._randint(0, self.size), self._randint(0, self.size))
            if candidate not in self._obstacle_locations:
                self._agent_location = np.array(candidate, dtype=int)
                start_pos = candidate

        # BFS para calcular células alcançáveis
        q = collections.deque([start_pos])
        reachable: set = {start_pos}
        while q:
            r, c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nb = (r + dr, c + dc)
                nr, nc = nb
                if (0 <= nr < self.size and 0 <= nc < self.size
                        and nb not in self._obstacle_locations
                        and nb not in reachable):
                    reachable.add(nb)
                    q.append(nb)

        self.visited_cells = {start_pos}
        observation = self._get_obs()
        info = self._get_info()
        info["reachable_cells_count"] = len(reachable)

        print(f"{self.size * self.size - len(reachable)} células indisponíveis")

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        action = int(action)
        direction = self._action_to_direction[action]
        potential = self._agent_location + direction
        info = self._get_info()
        info["hit_obstacle"] = False

        tr, tc = potential
        is_valid = (
            0 <= tr < self.size
            and 0 <= tc < self.size
            and (tr, tc) not in self._obstacle_locations
        )

        if is_valid:
            self._agent_location = potential
            self.visited_cells.add(tuple(self._agent_location))
        elif (0 <= tr < self.size and 0 <= tc < self.size
              and (tr, tc) in self._obstacle_locations):
            info["hit_obstacle"] = True

        self.steps += 1
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        # Recompensa, terminação e truncamento são geridos pelo CoverageWrapper
        return observation, 0.0, False, False, info

    # ── utilidades ────────────────────────────────────────────────────────────

    def set_visited_cells(self, visited_set: set):
        self.visited_cells = set(visited_set)

    # ── render ────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            pygame.display.set_caption(
                f"Grid World Coverage {self.size}×{self.size}"
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix = self.window_size / self.size

        for r, c in self.visited_cells:
            pygame.draw.rect(
                canvas, (220, 220, 220),
                pygame.Rect(c * pix, r * pix, pix, pix),
            )
        for r, c in self._obstacle_locations:
            pygame.draw.rect(
                canvas, (0, 0, 0),
                pygame.Rect(c * pix, r * pix, pix, pix),
            )

        if self._agent_location is not None:
            ar, ac = self._agent_location
            pygame.draw.circle(
                canvas, (0, 0, 255),
                ((ac + 0.5) * pix, (ar + 0.5) * pix),
                pix / 3,
            )

        for i in range(self.size + 1):
            pygame.draw.line(canvas, (100, 100, 100),
                             (0, i * pix), (self.window_size, i * pix), 1)
            pygame.draw.line(canvas, (100, 100, 100),
                             (i * pix, 0), (i * pix, self.window_size), 1)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None