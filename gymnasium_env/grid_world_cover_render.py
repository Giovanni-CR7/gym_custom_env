# gymnasium_env/grid_world_cover_render.py

import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import collections # Import necessário para a deque (fila do BFS)

class GridWorldCoverRenderEnv(gym.Env):
    """
    Ambiente Gymnasium de Grid World onde o objetivo é cobrir todas as células livres.
    Inclui obstáculos, renderização com Pygame e verificação de acessibilidade no reset.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, size=20, render_mode=None, num_obstacles=40, max_steps=None, seed=None):
        super().__init__()
        self.size = int(size)
        self.window_size = 512 # Pode ajustar se 20x20 ficar muito pequeno (ex: 768)
        self.num_obstacles = int(num_obstacles)

        # Validação inicial de parâmetros
        if self.num_obstacles >= self.size * self.size:
            raise ValueError(f"Número de obstáculos ({self.num_obstacles}) não pode ser >= tamanho total do grid ({self.size*self.size}).")

        self._obstacle_locations = set()
        self.visited_cells = set() # Conjunto de células visitadas, usado PRIMARIAMENTE para renderização

        # Define um limite padrão generoso para garantir a possibilidade de cobertura inicial
        # Multiplicador 4 para dar bastante margem para exploração aleatória inicial
        self.max_steps = max_steps if max_steps is not None else int(size * size * 4) # Ex: 20*20*4 = 1600

        # Actions: 0: Down, 1: Right, 2: Up, 3: Left
        self.action_space = spaces.Discrete(4)
        # Observation: 4 Canais (C, H, W) - [Livres, Obstáculos, Agente, Visitados]
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4, self.size, self.size), dtype=np.float32)

        # Mapeamento Ação -> Direção [delta_row, delta_col]
        self._action_to_direction = {
            0: np.array([1, 0]),  # Down
            1: np.array([0, 1]),  # Right
            2: np.array([-1, 0]), # Up
            3: np.array([0, -1]), # Left
        }

        # Configuração de Renderização
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Gerador de Números Aleatórios (RNG)
        self._seed = seed
        self.np_random = np.random.RandomState(seed) # RNG principal do ambiente

        # Estado Interno
        self._agent_location = None # Posição atual [row, col]
        self.steps = 0 # Contador de passos interno do ambiente base

    # Seed method (Gymnasium >= 0.26)
    def seed(self, seed=None):
        super().seed(seed)
        self._seed = seed
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _get_obs(self):
        """ Gera a observação em formato de canais numpy. """
        obs = np.zeros((4, self.size, self.size), dtype=np.float32)

        # Canal 0: Células livres (1.0 onde não há obstáculo)
        obs[0] = np.ones((self.size, self.size), dtype=np.float32)
        for r, c in self._obstacle_locations:
            if 0 <= r < self.size and 0 <= c < self.size: # Checagem de limites
                obs[0, r, c] = 0.0

        # Canal 1: Máscara de obstáculos (1.0 onde há obstáculo)
        for r, c in self._obstacle_locations:
             if 0 <= r < self.size and 0 <= c < self.size:
                obs[1, r, c] = 1.0

        # Canal 2: Posição do agente (one-hot encoding)
        r_agent, c_agent = self._agent_location
        if 0 <= r_agent < self.size and 0 <= c_agent < self.size:
             obs[2, r_agent, c_agent] = 1.0

        # Canal 3: Máscara de visitados (usada pelo agente, preenchida pelo Wrapper ou teste)
        # O ambiente base preenche com self.visited_cells (usado para render)
        for r, c in self.visited_cells:
            if 0 <= r < self.size and 0 <= c < self.size:
                 obs[3, r, c] = 1.0

        return obs

    def _get_info(self):
        """ Retorna informações adicionais sobre o estado (útil para o Wrapper). """
        return {
            "size": self.size,
            "obstacles": set(self._obstacle_locations), # Passa cópia dos obstáculos
            "agent_location": tuple(self._agent_location)
        }

    def reset(self, seed=None, options=None):
        """ Reseta o ambiente, garantindo que todas as células livres sejam acessíveis. """
        super().reset(seed=seed) # Importante chamar o reset do pai (Gymnasium >= 0.26)

        generation_attempts = 0
        while True: # Loop infinito até encontrar um mapa válido
            generation_attempts += 1
            # Aviso se estiver demorando muito (pode indicar problema de configuração)
            if generation_attempts > 1 and generation_attempts % 100 == 0:
                 print(f"AVISO: {generation_attempts} tentativas para gerar um mapa válido. Verificando configuração...")

            self._obstacle_locations = set()
            self.steps = 0 # Reseta contador de passos do ambiente base

            # 1. Gerar obstáculos aleatoriamente usando o RNG do ambiente
            while len(self._obstacle_locations) < self.num_obstacles:
                pos = (self.np_random.randint(0, self.size), self.np_random.randint(0, self.size))
                self._obstacle_locations.add(pos)

            # 2. Gerar posição inicial do agente (garante não ser obstáculo)
            start_pos = None # Reset para garantir que gere uma nova
            while start_pos is None:
                potential_start = (self.np_random.randint(0, self.size), self.np_random.randint(0, self.size))
                if potential_start not in self._obstacle_locations:
                    self._agent_location = np.array(potential_start, dtype=int)
                    start_pos = tuple(self._agent_location) # Guarda a posição inicial como tupla

            # 3. Verificar Acessibilidade via BFS (Busca em Largura)
            q = collections.deque([start_pos]) # Fila para o BFS, começa na posição inicial
            reachable_cells = {start_pos}      # Conjunto de células alcançáveis

            while q:
                r, c = q.popleft()
                # Verificar vizinhos (cima, baixo, esquerda, direita)
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    neighbor = (nr, nc)

                    # Checa se o vizinho está dentro do grid, não é obstáculo e ainda não foi adicionado
                    if (0 <= nr < self.size and
                        0 <= nc < self.size and
                        neighbor not in self._obstacle_locations and
                        neighbor not in reachable_cells):
                        reachable_cells.add(neighbor)
                        q.append(neighbor) # Adiciona à fila para explorar a partir dele

            # 4. Validar se todas as células livres foram alcançadas
            expected_free_cells = self.size * self.size - self.num_obstacles
            if len(reachable_cells) == expected_free_cells:
                # if generation_attempts > 1: # Log apenas se não foi na primeira tentativa
                #      print(f"Mapa válido gerado na tentativa {generation_attempts}.") # Debug opcional
                break # Sai do loop while True, mapa encontrado!

            # Se não for válido, o loop continua e tenta gerar novamente

        # Prossegue com o reset normal APÓS garantir um mapa válido
        self.visited_cells = set() # Limpa visitados para renderização
        self.visited_cells.add(start_pos) # Adiciona posição inicial aos visitados (para render)

        observation = self._get_obs()
        info = self._get_info() # Pega info final com o mapa válido

        # Inicializa a renderização se necessário
        if self.render_mode == "human":
            self._render_frame()

        # Retorna o estado inicial padrão do Gymnasium
        return observation, info

    def step(self, action):
        """ Executa um passo no ambiente. """
        action = int(action) # Garante que a ação é um inteiro
        direction = self._action_to_direction[action]
        potential_location = self._agent_location + direction

        # Pega info ANTES de mover o agente
        info = self._get_info()
        info["hit_obstacle"] = False # Inicializa flag

        target_r, target_c = potential_location

        # Verifica se o movimento é válido (dentro dos limites e não colide com obstáculo)
        is_valid_move = (0 <= target_r < self.size and
                         0 <= target_c < self.size and
                         (target_r, target_c) not in self._obstacle_locations)

        if is_valid_move:
            # Atualiza a posição do agente
            self._agent_location = potential_location
            # Adiciona a nova posição ao conjunto de visitados (para renderização)
            self.visited_cells.add(tuple(self._agent_location))
        else:
            # Verifica se a colisão foi com um obstáculo (dentro dos limites)
            if (0 <= target_r < self.size and
                0 <= target_c < self.size and
                (target_r, target_c) in self._obstacle_locations):
                info["hit_obstacle"] = True # Sinaliza a colisão para o Wrapper
            # Se a colisão foi com a borda, o agente simplesmente não se move

        self.steps += 1 # Incrementa o contador de passos interno

        # O ambiente base NÃO define terminated/truncated
        # Essa lógica agora reside COMPLETAMENTE no Wrapper
        terminated = False
        truncated = False

        # A recompensa base é neutra; o Wrapper fará o "reward shaping"
        reward = 0.0

        observation = self._get_obs() # Gera a observação do novo estado

        # Renderiza o frame se estiver no modo 'human'
        if self.render_mode == "human":
            self._render_frame()

        # Retorna a tupla padrão do Gymnasium step()
        return observation, reward, terminated, truncated, info

    def set_visited_cells(self, visited_set):
        """ Método auxiliar para atualizar o conjunto de células visitadas (usado para renderização). """
        self.visited_cells = set(visited_set)

    def render(self):
        """ Método de renderização padrão do Gymnasium. """
        if self.render_mode == "rgb_array":
            return self._render_frame() # Retorna o frame como numpy array
        elif self.render_mode == "human":
             self._render_frame() # Atualiza a janela Pygame

    def _render_frame(self):
        """ Renderiza o estado atual do grid usando Pygame. """
        if self.render_mode is None:
             # Evita erro se chamar render sem modo definido
             gym.logger.warn("Você está chamando render() mas o render_mode não foi definido.")
             return

        # Inicializa Pygame se ainda não foi feito (para modo 'human')
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption(f"Grid World Coverage {self.size}x{self.size}") # Título da janela
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Cria a superfície de desenho (canvas)
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # Fundo branco

        # Calcula o tamanho de cada célula do grid na tela
        pix_square_size = self.window_size / self.size

        # Desenha células visitadas (cinza claro) - usa self.visited_cells
        for r, c in self.visited_cells:
            # Pygame usa coordenadas (x, y) = (coluna, linha)
            pygame.draw.rect(
                canvas,
                (220, 220, 220), # Cor cinza claro
                pygame.Rect(
                    c * pix_square_size, # Posição X (coluna)
                    r * pix_square_size, # Posição Y (linha)
                    pix_square_size,     # Largura
                    pix_square_size,     # Altura
                ),
            )

        # Desenha obstáculos (preto)
        for r, c in self._obstacle_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0), # Cor preta
                 pygame.Rect(
                    c * pix_square_size,
                    r * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                ),
            )

        # Desenha o agente (círculo azul)
        if self._agent_location is not None: # Checa se a localização é válida
            agent_r, agent_c = self._agent_location
            # Calcula o centro da célula para desenhar o círculo
            center_x = (agent_c + 0.5) * pix_square_size
            center_y = (agent_r + 0.5) * pix_square_size
            pygame.draw.circle(
                canvas,
                (0, 0, 255),    # Cor azul
                (center_x, center_y), # Posição do centro
                pix_square_size / 3,  # Raio do círculo
            )

        # Desenha as linhas do grid (cinza escuro e finas)
        for i in range(self.size + 1):
            # Linhas horizontais
            pygame.draw.line(canvas, (100, 100, 100), (0, i * pix_square_size), (self.window_size, i * pix_square_size), width=1)
            # Linhas verticais
            pygame.draw.line(canvas, (100, 100, 100), (i * pix_square_size, 0), (i * pix_square_size, self.window_size), width=1)

        # Se for modo 'human', atualiza a janela Pygame
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect()) # Copia o canvas para a janela
            pygame.event.pump() # Processa eventos internos do Pygame
            pygame.display.update() # Atualiza a tela visível

            # Controla a taxa de quadros por segundo (FPS)
            self.clock.tick(self.metadata["render_fps"])

        # Se for modo 'rgb_array', retorna a imagem como um array numpy
        else:
             # Transpõe os eixos de (W, H, C) do Pygame para (H, W, C) esperado pelo Gymnasium/SB3
             return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))

    def close(self):
        """ Fecha a janela Pygame se ela estiver aberta. """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            # Marca como fechado para evitar erros se close() for chamado novamente
            self.window = None
            self.clock = None

