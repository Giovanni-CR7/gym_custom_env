import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class GridWorldCoverRenderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, size=5, render_mode=None, num_obstacles=3): #recebe num_obstacles
        self.size = size
        self.window_size = 512
        self.num_obstacles = num_obstacles # armazena o número de obstáculos
        self._obstacle_locations = set()  #  armazena as posições dos obstáculos
        self.visited_cells = set() #  para visualização

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {"agent": spaces.Box(0, size - 1, shape=(2,), dtype=int)}
        )

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

    def _get_obs(self):
        return {"agent": self._agent_location}

    def _get_info(self):
        # Informa a localização dos obstáculos para o wrapper
        return {"size": self.size, "obstacles": self._obstacle_locations}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.visited_cells.clear() # limpa as células visitadas para o render

        # Gera obstáculos em posições aleatórias
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._obstacle_locations = set()
        while len(self._obstacle_locations) < self.num_obstacles:
            obstacle_pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
            # Garante que o obstáculo não seja na posição inicial do agente
            if obstacle_pos != tuple(self._agent_location):
                self._obstacle_locations.add(obstacle_pos)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        potential_location = self._agent_location + direction
        info = self._get_info()
        info["hit_obstacle"] = False

        # Verifica se o próximo movimento é uma colisão com um obstáculo
        if tuple(potential_location) in self._obstacle_locations:
            # O agente não se move e recebe uma flag de colisão
            info["hit_obstacle"] = True
        else:
            # Movimento normal, limitado pelas bordas
            self._agent_location = np.clip(potential_location, 0, self.size - 1)

        terminated = False
        truncated = False
        reward = 0
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    #  Método para o wrapper atualizar as células visitadas para renderização
    def set_visited_cells(self, visited):
        self.visited_cells = visited

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

        # Desenha as células já visitadas em cinza claro
        for pos in self.visited_cells:
             pygame.draw.rect(
                canvas,
                (200, 200, 200), # Cinza claro
                pygame.Rect(
                    pix_square_size * np.array(pos),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Desenha os obstáculos como quadrados pretos
        for obstacle in self._obstacle_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0), # Preto
                pygame.Rect(
                    pix_square_size * np.array(obstacle),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Desenha o agente como um círculo azul
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for x in range(self.size + 1):
            pygame.draw.line(canvas,0,(0, pix_square_size * x),(self.window_size, pix_square_size * x),width=3,)
            pygame.draw.line(canvas,0,(pix_square_size * x, 0),(pix_square_size * x, self.window_size),width=3,)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()