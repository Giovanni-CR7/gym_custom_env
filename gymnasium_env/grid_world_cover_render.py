import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class GridWorldCoverRenderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=5, render_mode=None):
        self.size = size  # tamanho da grade
        self.window_size = 512  # tamanho da janela

        # ação: 0=right, 1=up, 2=left, 3=down
        self.action_space = spaces.Discrete(4)

        # observação: posição do agente
        self.observation_space = spaces.Dict(
            {"agent": spaces.Box(0, size - 1, shape=(2,), dtype=int)}
        )

        # mapeamento de ações
        self._action_to_direction = {
            0: np.array([1, 0]),   # direita
            1: np.array([0, 1]),   # cima
            2: np.array([-1, 0]),  # esquerda
            3: np.array([0, -1]),  # baixo
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location}

    def _get_info(self):
        return {"size": self.size}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # agente começa em posição aleatória
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # esse ambiente não tem alvo, logo não tem "terminated" interno
        terminated = False
        truncated = False
        reward = 0  # quem define a recompensa é o wrapper

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

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

        # desenha o agente como quadrado azul
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                pix_square_size * self._agent_location,
                (pix_square_size, pix_square_size),
            ),
        )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

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
