import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

class GridWorldCoverRenderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, size=20, render_mode=None, num_obstacles=40, max_steps=None, seed=None): # Defaults atualizados
        super().__init__()
        self.size = int(size)
        self.window_size = 512 # Pode querer aumentar para grid 20x20 se ficar pequeno (ex: 768)
        self.num_obstacles = int(num_obstacles)
        self._obstacle_locations = set()
        self.visited_cells = set() # Células visitadas PARA RENDERIZAÇÃO
        
        # --- CALCULO ATUALIZADO PARA MAX_STEPS ---
        # Define um limite padrão generoso para grids maiores
        self.max_steps = max_steps if max_steps is not None else int(size * size * 4) 

        # Actions: 0 down, 1 right, 2 up, 3 left
        self.action_space = spaces.Discrete(4)
        # Observation: channels = [free, obstacles, agent, visited] -> shape (4, H, W)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4, self.size, self.size), dtype=np.float32)

        self._action_to_direction = {
            0: np.array([1, 0]),  # Down (row+)
            1: np.array([0, 1]),  # Right (col+)
            2: np.array([-1, 0]), # Up (row-)
            3: np.array([0, -1]), # Left (col-)
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self._seed = seed
        self.np_random = np.random.RandomState(seed) # Use este RNG para consistência

        self._agent_location = None
        self.steps = 0 # Contador interno de passos do ambiente base

    def seed(self, seed=None):
        super().seed(seed) # Necessário no Gymnasium >= 0.26
        self._seed = seed
        self.np_random = np.random.RandomState(seed)
        # random.seed(seed) # Opcional, se usar a biblioteca 'random' diretamente
        # Retorna a seed usada (pode ser útil)
        return [seed]

    def _get_obs(self):
        obs = np.zeros((4, self.size, self.size), dtype=np.float32)
        # Channel 0: free cells (1.0 where not obstacle)
        obs[0] = np.ones((self.size, self.size), dtype=np.float32)
        for r, c in self._obstacle_locations:
            if 0 <= r < self.size and 0 <= c < self.size: # Boundary check
                obs[0, r, c] = 0.0
        # Channel 1: obstacle mask (1.0 where obstacle)
        for r, c in self._obstacle_locations:
             if 0 <= r < self.size and 0 <= c < self.size:
                obs[1, r, c] = 1.0
        # Channel 2: agent location (one-hot)
        r_agent, c_agent = self._agent_location
        if 0 <= r_agent < self.size and 0 <= c_agent < self.size:
             obs[2, r_agent, c_agent] = 1.0
        # Channel 3: visited mask (PARA O AGENTE, não necessariamente o render)
        # O Wrapper agora controla a lógica de 'visited' para recompensa,
        # mas o ambiente pode manter seu próprio registro se necessário,
        # ou este canal pode ser preenchido pelo wrapper.
        # Aqui, vamos preencher com base no `self.visited_cells` do ambiente,
        # que será atualizado pelo wrapper ou pelo teste.
        for r, c in self.visited_cells:
            if 0 <= r < self.size and 0 <= c < self.size:
                 obs[3, r, c] = 1.0
        return obs

    def _get_info(self):
        # Info útil para o Wrapper ou para análise
        return {
            "size": self.size,
            "obstacles": set(self._obstacle_locations),
            "agent_location": tuple(self._agent_location),
            "distance_to_target": None # Placeholder se tivesse um alvo
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Importante chamar o reset do pai
        
        self.visited_cells = set() # Limpa visitados para renderização
        self._obstacle_locations = set()
        self.steps = 0

        # Gera obstáculos usando o RNG do ambiente
        while len(self._obstacle_locations) < self.num_obstacles:
            pos = (self.np_random.randint(0, self.size), self.np_random.randint(0, self.size))
            self._obstacle_locations.add(pos)

        # Gera posição inicial do agente (garante não ser obstáculo)
        while True:
            start_pos = (self.np_random.randint(0, self.size), self.np_random.randint(0, self.size))
            if start_pos not in self._obstacle_locations:
                self._agent_location = np.array(start_pos, dtype=int)
                break
        
        # Adiciona a posição inicial aos visitados (para render)
        self.visited_cells.add(tuple(self._agent_location))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        action = int(action)
        direction = self._action_to_direction[action]
        potential_location = self._agent_location + direction
        
        info = self._get_info() # Pega info antes da mudança
        info["hit_obstacle"] = False # Reseta a flag

        target_r, target_c = potential_location

        # Verifica se o movimento é válido (dentro dos limites e não é obstáculo)
        is_valid_move = (0 <= target_r < self.size and
                         0 <= target_c < self.size and
                         (target_r, target_c) not in self._obstacle_locations)

        if is_valid_move:
            self._agent_location = potential_location
            self.visited_cells.add(tuple(self._agent_location)) # Atualiza visitados para render
        else:
            # Verifica se bateu em obstáculo (dentro dos limites)
            if (0 <= target_r < self.size and
                0 <= target_c < self.size and
                (target_r, target_c) in self._obstacle_locations):
                info["hit_obstacle"] = True
            # Se for fora dos limites, o agente simplesmente não se move

        self.steps += 1

        # O ambiente base agora NÃO define terminated/truncated por cobertura/tempo
        # Essa lógica fica no Wrapper para maior flexibilidade
        terminated = False # O Wrapper decidirá isso com base na cobertura
        truncated = False # O Wrapper decidirá isso com base em self.current_step >= self.max_steps
        
        # Recompensa base é neutra, o Wrapper fará o shaping
        reward = 0.0

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        # O Wrapper usará 'terminated', 'truncated' e 'info' para calcular a recompensa final
        return observation, reward, terminated, truncated, info

    # Método para o teste poder atualizar os visitados para renderização
    def set_visited_cells(self, visited_set):
        self.visited_cells = set(visited_set)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
             self._render_frame() # Já chamado no step, mas pode ser útil chamar externamente

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # Fundo branco
        pix_square_size = self.window_size / self.size # Tamanho de cada célula

        # Desenha células visitadas (cinza claro)
        for r, c in self.visited_cells:
            pygame.draw.rect(
                canvas,
                (220, 220, 220), # Cinza mais claro
                pygame.Rect(
                    c * pix_square_size, # Pygame usa (x, y) = (col, row)
                    r * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                ),
            )

        # Desenha obstáculos (preto)
        for r, c in self._obstacle_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0), # Preto
                 pygame.Rect(
                    c * pix_square_size,
                    r * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                ),
            )

        # Desenha o agente (azul)
        agent_r, agent_c = self._agent_location
        pygame.draw.circle(
            canvas,
            (0, 0, 255), # Azul
            ( (agent_c + 0.5) * pix_square_size, (agent_r + 0.5) * pix_square_size ), # Centro da célula
            pix_square_size / 3, # Raio do círculo
        )

        # Desenha as linhas do grid (cinza escuro)
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas, (100, 100, 100),
                (0, x * pix_square_size), (self.window_size, x * pix_square_size), width=1
            )
            pygame.draw.line(
                 canvas, (100, 100, 100),
                (x * pix_square_size, 0), (x * pix_square_size, self.window_size), width=1
            )

        if self.render_mode == "human":
            # Copia o canvas para a janela
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # Controla o FPS
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            # Retorna a imagem como um array numpy (H, W, C) e transpõe para (W, H, C) que Pygame usa? Não, SB3 espera (H, W, C)
             return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2)) # (W, H, C) -> (H, W, C)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None # Importante para evitar erro se chamar close() duas vezes
            self.clock = None

