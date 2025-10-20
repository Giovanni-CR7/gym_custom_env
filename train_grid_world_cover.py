# python train_grid_world_cover.py train  # para treinar
# python train_grid_world_cover.py test   # para testar

import sys
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import os
import collections # Importado para uso no ambiente, mas bom ter aqui

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# Garanta que o Python consiga encontrar este módulo (pode precisar ajustar PYTHONPATH ou estrutura)
from gymnasium_env.grid_world_cover_render import GridWorldCoverRenderEnv

# --- Custom CNN ---
class CustomCnn(BaseFeaturesExtractor):
    """ CNN customizada para extrair features da observação em grid. """
    def __init__(self, observation_space, features_dim=256): # Aumentado features_dim
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        # Rede um pouco mais profunda pode ajudar com 20x20
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Camada extra
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # Outra camada extra
            nn.ReLU(),
            nn.Flatten()
        )
        # Calcula o tamanho da saída achatada da CNN
        with th.no_grad():
            sample = th.as_tensor(np.zeros((1, c, h, w), dtype=np.float32))
            n_flatten = self.cnn(sample).shape[1]
        # Camada linear final para a dimensão de features desejada
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Normalização opcional (se a observação não estiver entre 0 e 1)
        # observations = observations / 1.0 # Ajuste se necessário
        x = self.cnn(observations)
        return self.linear(x)

# --- Wrapper de Cobertura com Recompensa "Trator" ---
class CoverageWrapper(gym.Wrapper):
    """ Aplica a lógica de recompensa focada em eficiência e cobertura. """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.grid_size = env.observation_space.shape[1]
        # Pega o max_steps do ambiente base (que deve ser size*size*4 = 1600)
        self.max_steps = env.max_steps if hasattr(env, "max_steps") else int(self.grid_size * self.grid_size * 4)
        self.visited = set()
        self.obstacles = set()
        self.current_step = 0
        self.total_coverable_cells = self.grid_size * self.grid_size # Será ajustado no reset

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print(f"[Wrapper] Inicializado. Grid Size: {self.grid_size}, Max Steps: {self.max_steps}")

    def _get_obs(self, obs):
        """ Retorna a observação como está (formato de canais). """
        return obs

    def reset(self, **kwargs):
        """ Reseta o wrapper e o ambiente interno. """
        obs, info = self.env.reset(**kwargs)
        # Pega os obstáculos reais do mapa gerado (importante após a validação no env.reset)
        self.obstacles = set(info.get("obstacles", []))
        # Calcula células *realmente* cobríveis
        self.total_coverable_cells = self.grid_size * self.grid_size - len(self.obstacles)

        # Inicializa conjunto de visitados (agente começa em uma célula)
        self.visited = set()
        agent_channel = obs[2]
        agent_pos_search = np.argwhere(agent_channel == 1.0)
        if agent_pos_search.shape[0] > 0:
             ai, aj = agent_pos_search[0]
             self.visited.add((int(ai), int(aj)))
        else:
             print("AVISO [Wrapper Reset]: Posição inicial do agente não encontrada na observação!")
             # Tenta pegar do info como fallback
             try:
                  start_loc = info.get("agent_location")
                  if start_loc: self.visited.add(start_loc)
             except Exception: pass


        self.current_step = 0
        # print(f"[Wrapper Reset] Coverable: {self.total_coverable_cells}, Initial Visited: {len(self.visited)}") # Debug
        return self._get_obs(obs), info

    def step(self, action):
        """ Executa um passo, calcula a recompensa e atualiza o estado. """
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        # Pega a posição atual do agente a partir da observação
        agent_channel = obs[2]
        pos = np.argwhere(agent_channel == 1.0)
        agent_pos = None # Reset
        if pos.shape[0] > 0:
            ai, aj = int(pos[0,0]), int(pos[0,1])
            agent_pos = (ai, aj)
        else: # Fallback se não encontrar na obs (não deveria acontecer)
             try: agent_pos = tuple(self.env.unwrapped._agent_location)
             except Exception: pass

        hit_obstacle = info.get("hit_obstacle", False)

        # --- RECOMPENSA "TRATOR" COM ALTO CUSTO DE PASSO ---
        reward = -0.2  # Custo alto por passo para forçar eficiência

        # Nenhuma penalidade direta por bater em obstáculo
        if hit_obstacle:
            pass # Equivalente a reward += 0.0

        # Se não bateu e a posição é válida
        elif agent_pos is not None:
            if agent_pos not in self.visited:
                self.visited.add(agent_pos)
                reward += 1.0 # Recompensa por célula nova
            else:
                reward += -0.1 # Penalidade por revisitar

        # Bônus terminal por cobertura total (usa o total_coverable_cells real)
        if len(self.visited) >= self.total_coverable_cells:
            terminated = True
            reward += 50.0
            # print(f"[Wrapper Step {self.current_step}] Coverage Complete! Visited: {len(self.visited)}, Total: {self.total_coverable_cells}. Reward: {reward}") # Debug

        # Truncamento por limite de passos (sem penalidade adicional)
        if not terminated and self.current_step >= self.max_steps:
            truncated = True
            # print(f"[Wrapper Step {self.current_step}] Truncated. Visited: {len(self.visited)}, Total: {self.total_coverable_cells}") # Debug


        # Atualiza o canal 'visitados' na observação para consistência (útil para o agente e render)
        if agent_pos is not None:
             obs[3, agent_pos[0], agent_pos[1]] = 1.0

        return self._get_obs(obs), float(reward), bool(terminated), bool(truncated), info

# --- Factory do Ambiente ---
def make_env(render_mode=None, size=20, log_dir=None, num_obstacles=40, seed=None):
    """ Cria e envolve o ambiente. """
    env = GridWorldCoverRenderEnv(size=size, render_mode=render_mode, num_obstacles=num_obstacles, seed=seed)
    env = CoverageWrapper(env) # Aplica o wrapper de recompensa
    if log_dir:
        os.makedirs(log_dir, exist_ok=True) # Cria o diretório se não existir
        # Envolve com Monitor para salvar logs em CSV
        env = Monitor(env, log_dir) # 'is_success' é opcional
    return env

# --- Função de Plotagem ---
def plot_results(log_dir):
    """ Plota a recompensa média móvel a partir do monitor.csv. """
    try:
        monitor_files = [f for f in os.listdir(log_dir) if f.endswith("monitor.csv")]
        if not monitor_files:
            print(f"⚠️ Nenhum arquivo monitor.csv encontrado em {log_dir}")
            return

        filepath = os.path.join(log_dir, monitor_files[0])
        print(f"📊 Gerando gráfico a partir de '{filepath}'...")
        # Tenta carregar, tratando erros se o arquivo estiver corrompido ou mal formatado
        try:
            data = pd.read_csv(filepath, skiprows=1) # Pula a linha de cabeçalho JSON
        except Exception as e:
            print(f"⚠️ Erro ao ler {filepath}: {e}. O arquivo pode estar incompleto.")
            return

        if data.empty or 'r' not in data.columns:
             print(f"⚠️ Arquivo {filepath} está vazio ou não contém a coluna de recompensa 'r'.")
             return

        window_size = 100 # Janela da média móvel
        # Calcula média móvel, tratando casos com poucos dados
        if len(data['r']) >= window_size:
            rolling_avg = data['r'].rolling(window=window_size).mean()
        elif len(data['r']) > 0:
             rolling_avg = data['r'].rolling(window=len(data['r'])).mean()
        else:
             rolling_avg = pd.Series() # Série vazia se não houver dados

        plt.figure(figsize=(12, 6))
        plt.title("Evolução da Recompensa Média por Episódio")
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa Média (Móvel)")
        plt.plot(data.index, data['r'], 'b.', alpha=0.15, label='Recompensa por Episódio') # Pontos mais transparentes
        if not rolling_avg.empty:
             plt.plot(data.index, rolling_avg, 'r-', linewidth=2, label=f'Média Móvel ({window_size} episódios)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout() # Ajusta o layout para evitar sobreposição
        plt.show() # Mostra o gráfico
    except Exception as e:
        print(f"⚠️ Não foi possível gerar o gráfico completo: {e}")

# --- Script Principal (Treino e Teste) ---
if __name__ == "__main__":
    # Verifica se o primeiro argumento é 'train' ou 'test'
    mode = sys.argv[1] if len(sys.argv) > 1 else "test" # Default para testar se nenhum argumento for dado

    # Registra o ambiente no Gymnasium (se ainda não estiver registrado)
    try:
        gym.register(
            id="gymnasium_env/GridWorld-Coverage-v0",
            entry_point="gymnasium_env.grid_world_cover_render:GridWorldCoverRenderEnv",
        )
    except gym.error.Error as e:
        # print(f"Ambiente já registrado: {e}") # Debug opcional
        pass

    # --- PARÂMETROS GLOBAIS ---
    GRID_SIZE = 20
    NUM_OBSTACLES = 40
    LOG_DIR = f"log/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}"
    MODEL_SAVE_PATH = f"data/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}.zip"
    TOTAL_TIMESTEPS = 17_000_000 # Orçamento total de treino

    # --- MODO DE TREINAMENTO ---
    if mode == "train":
        print(f"--- Iniciando Treinamento: Grid {GRID_SIZE}x{GRID_SIZE}, Obstáculos: {NUM_OBSTACLES}, Passos: {TOTAL_TIMESTEPS} ---")
        # Cria o ambiente para treino (sem renderização, com Monitor)
        env = make_env(render_mode=None, size=GRID_SIZE, log_dir=LOG_DIR, num_obstacles=NUM_OBSTACLES, seed=0)

        # Configuração da rede neural (Política e Função de Valor)
        policy_kwargs = dict(
            features_extractor_class=CustomCnn,
            features_extractor_kwargs=dict(features_dim=256), # Aumentado
            net_arch=dict(pi=[256, 128], vf=[256, 128])    # Aumentado
        )

        # Configuração do Algoritmo PPO
        model = PPO(
            "CnnPolicy",           # Usa a política baseada em CNN
            env,                   # O ambiente envolvido
            verbose=1,             # Imprime informações de progresso
            device="cpu",          # Mude para "cuda" se tiver GPU NVIDIA configurada
            policy_kwargs=policy_kwargs, # Arquitetura da rede customizada
            ent_coef=0.01,         # Coeficiente de entropia (exploração) - pode ajustar (ex: 0.005)
            learning_rate=5e-5,    # Taxa de aprendizado REDUZIDA
            gamma=0.999,           # Fator de desconto ALTO (visão de longo prazo)
            n_steps=4096,          # Horizonte de coleta MAIOR
            batch_size=128,        # Tamanho do batch para atualização
            n_epochs=20,           # Número de épocas de otimização por coleta
            clip_range=0.2,        # Clipping do PPO (padrão)
            tensorboard_log=LOG_DIR # Diretório para logs do TensorBoard
        )

        print("Configuração do Modelo PPO:")
        print(f"  Learning Rate: {model.learning_rate}")
        print(f"  N Steps: {model.n_steps}")
        print(f"  Batch Size: {model.batch_size}")
        print(f"  N Epochs: {model.n_epochs}")
        print(f"  Gamma: {model.gamma}")
        print(f"  Ent Coef: {model.ent_coef}")
        print(f"  Clip Range: {model.clip_range}")
        print(f"  Device: {model.device}")
        
        # O Monitor já configura o logger CSV, aqui configuramos stdout e TensorBoard
        # logger = configure(LOG_DIR, ["stdout", "tensorboard"]) # O Monitor já lida com CSV
        # model.set_logger(logger) # set_logger é mais usado se não usar Monitor ou quiser formatters customizados

        try:
            # Inicia o treinamento
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                log_interval=10, # Loga estatísticas a cada 10 atualizações
                progress_bar=True # Mostra barra de progresso
            )
        except KeyboardInterrupt:
            print("\n🛑 Treinamento interrompido pelo usuário.")
        finally:
            # Garante que o diretório de dados existe
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            # Salva o modelo treinado
            model.save(MODEL_SAVE_PATH)
            print(f"✅ Modelo salvo em {MODEL_SAVE_PATH}")
            # Tenta plotar o gráfico no final
            plot_results(LOG_DIR)

    # --- MODO DE TESTE ---
    elif mode == "test":
        print(f"--- Iniciando Modo de Teste: Grid {GRID_SIZE}x{GRID_SIZE}, Obstáculos: {NUM_OBSTACLES} ---")
        print(f"📂 Carregando modelo treinado de '{MODEL_SAVE_PATH}'...")

        # Verifica se o arquivo do modelo existe
        if not os.path.exists(MODEL_SAVE_PATH):
             print(f"❌ Erro: Arquivo do modelo não encontrado em '{MODEL_SAVE_PATH}'. Execute o treino primeiro ('python {sys.argv[0]} train').")
             sys.exit(1) # Termina o script se o modelo não for encontrado

        # Carrega o modelo treinado
        # Não precisa passar 'env' aqui se só for usar para predict, mas é boa prática se for adaptar depois
        model = PPO.load(MODEL_SAVE_PATH)
        print("✅ Modelo carregado.")

        # Cria o ambiente para teste (com renderização "human" e SEM Monitor)
        # Usar uma seed diferente no teste pode ser interessante para ver generalização
        env = make_env(render_mode="human", size=GRID_SIZE, num_obstacles=NUM_OBSTACLES, log_dir=None, seed=np.random.randint(1000))

        num_test_episodes = 5
        for i in range(num_test_episodes):
            print(f"\n--- Iniciando Teste {i+1}/{num_test_episodes} ---")
            obs, info = env.reset() # Obtém a observação e info iniciais
            done, truncated = False, False
            steps = 0
            total_reward = 0
            
            # Pega o set de visitados do wrapper interno para renderizar corretamente
            # Acessa através de env.env por causa do Monitor (se estivesse presente)
            # Como removemos o Monitor para teste, o acesso é direto via env.
            current_visited_for_render = set(env.visited) if hasattr(env, 'visited') else set()

            while not (done or truncated):
                # Pede ao modelo a melhor ação (determinística)
                action, _ = model.predict(obs, deterministic=True)
                # Converte ação para int (necessário para o ambiente)
                a = int(action.item()) if isinstance(action, np.ndarray) else int(action)

                # Executa a ação no ambiente
                obs, reward, done, truncated, info = env.step(a)
                total_reward += reward
                
                # Atualiza o conjunto de visitados para renderização a partir do wrapper
                if hasattr(env, 'visited'):
                     current_visited_for_render = set(env.visited)
                
                # Atualiza a renderização com as células visitadas corretas
                try:
                    # Tenta chamar o método do ambiente base para atualizar o render
                    env.unwrapped.set_visited_cells(current_visited_for_render)
                except AttributeError:
                     print("Aviso: Não foi possível chamar set_visited_cells no ambiente unwrapped.")
                     pass # Renderização pode não mostrar visitados corretamente
                         
                steps += 1
                env.render() # Força a renderização (já deve ser chamada no step do env base)

            print(f"Episódio finalizado em {steps} passos.")
            print(f"  Terminated (Cobertura Completa): {done}")
            print(f"  Truncated (Limite de Passos): {truncated}")
            print(f"  Recompensa Total: {total_reward:.2f}")

            # Calcula e exibe a cobertura final
            try:
                 # Pega informações do ambiente base diretamente (sem Monitor no teste)
                 unwrapped_env = env.unwrapped
                 # Pega obstáculos e calcula total coberto deste episódio específico
                 obstacles_test = unwrapped_env._obstacle_locations
                 total_cover_test = unwrapped_env.size * unwrapped_env.size - len(obstacles_test)
                 # Usa o conjunto 'visited' final do wrapper para a contagem correta
                 visited_cnt = len(current_visited_for_render)
                 coverage_percent = (visited_cnt / total_cover_test * 100) if total_cover_test > 0 else 0
                 print(f"  Cobertura Final: {visited_cnt} de {total_cover_test} células acessíveis ({coverage_percent:.1f}%).")
            except Exception as e:
                 print(f"  Não foi possível calcular a cobertura detalhada: {e}")

        env.close() # Fecha a janela Pygame
        print("\n--- Testes Concluídos ---")

    else:
        print(f"Modo inválido '{mode}'. Use 'train' ou 'test'.")
        sys.exit(1)

