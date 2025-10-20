# python train_grid_world_cover.py train   # para treinar
# python train_grid_world_cover.py test    # para testar

import sys
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import os

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium_env.grid_world_cover_render import GridWorldCoverRenderEnv 

# --- Custom CNN ---
class CustomCnn(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        # ajuste leve na CNN para grid maior 
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # Camada extra
            nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            sample = th.as_tensor(np.zeros((1, c, h, w), dtype=np.float32))
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Normalização opcional pode ajudar CNNs
        # observations = observations / 255.0 # Descomente se sua obs não for 0-1
        x = self.cnn(observations)
        return self.linear(x)

# --- Wrapper de Cobertura ---
class CoverageWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.grid_size = env.observation_space.shape[1]
        # Pega o max_steps definido no ambiente base
        self.max_steps = env.max_steps if hasattr(env, "max_steps") else self.grid_size * self.grid_size * 4
        self.visited = set()
        self.obstacles = set()
        self.current_step = 0
        self.total_coverable_cells = self.grid_size * self.grid_size

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print(f"[Wrapper] Grid Size: {self.grid_size}, Max Steps: {self.max_steps}")


    def _get_obs(self, obs):
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obstacles = set(info.get("obstacles", []))
        self.total_coverable_cells = self.grid_size * self.grid_size - len(self.obstacles)
        
        visited_mask = obs[3]
        self.visited = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if visited_mask[i, j] > 0.5:
                    self.visited.add((i, j))
        
        agent_channel = obs[2]
        # Garante que funciona mesmo se o agente começar fora do grid (não deve acontecer)
        agent_pos_search = np.argwhere(agent_channel == 1.0)
        if agent_pos_search.shape[0] > 0:
             ai, aj = agent_pos_search[0]
             self.visited.add((int(ai), int(aj)))
        
        self.current_step = 0
        # print(f"[Wrapper Reset] Coverable: {self.total_coverable_cells}, Initial Visited: {len(self.visited)}") # Debug
        return self._get_obs(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        agent_channel = obs[2]
        pos = np.argwhere(agent_channel == 1.0)
        if pos.shape[0] == 0:
             try: # Fallback se _agent_location existir
                 ai, aj = tuple(self.env.unwrapped._agent_location)
             except Exception: # Último recurso
                 ai, aj = -1, -1 # Indica posição inválida
        else:
            ai, aj = int(pos[0,0]), int(pos[0,1])
        
        agent_pos = (ai, aj) if ai != -1 else None # Posição atual válida?

        hit_obstacle = info.get("hit_obstacle", False)

        # --- RECOMPENSA "TRATOR" ---
        reward = -0.01  # Custo de passo

        if hit_obstacle:
            reward += 0.0 # NENHUMA penalidade por bater
        # Se não bateu E a posição é válida
        elif agent_pos is not None:
            if agent_pos not in self.visited:
                self.visited.add(agent_pos)
                reward += 1.0 # Recompensa por célula nova
            else:
                reward += -0.1 # Penalidade por revisitar

        # Bônus terminal por cobertura total
        if len(self.visited) >= self.total_coverable_cells:
            terminated = True
            reward += 50.0
            # print(f"[Wrapper Step {self.current_step}] Coverage Complete! Reward: {reward}") # Debug

        # Truncamento por limite de passos (sem penalidade)
        if self.current_step >= self.max_steps:
            truncated = True

        # Atualiza a observação (canal visitado) se a posição for válida
        if agent_pos is not None:
             obs[3, ai, aj] = 1.0

        return self._get_obs(obs), float(reward), bool(terminated), bool(truncated), info

# --- Factory do Ambiente ---
def make_env(render_mode=None, size=20, log_dir=None, num_obstacles=40, seed=None):
    # Passa o tamanho correto para o ambiente base
    env = GridWorldCoverRenderEnv(size=size, render_mode=render_mode, num_obstacles=num_obstacles, seed=seed)
    env = CoverageWrapper(env)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True) # Cria o diretório se não existir
        env = Monitor(env, log_dir)
    return env

# --- Função de Plotagem ---
def plot_results(log_dir):
    try:
        monitor_files = [f for f in os.listdir(log_dir) if f.endswith("monitor.csv")]
        if not monitor_files:
            print(f"⚠️ Nenhum arquivo monitor.csv encontrado em {log_dir}")
            return
        
        filepath = os.path.join(log_dir, monitor_files[0])
        print(f"📊 Gerando gráfico a partir de '{filepath}'...")
        data = pd.read_csv(filepath, skiprows=1)
        
        if data.empty or 'r' not in data.columns:
             print("⚠️ Arquivo monitor.csv está vazio ou não contém a coluna 'r'.")
             return

        window_size = 100
        # Certifica que temos dados suficientes para a janela
        if len(data['r']) >= window_size:
            rolling_avg = data['r'].rolling(window=window_size).mean()
        else:
            rolling_avg = data['r'].rolling(window=len(data['r'])).mean() # Média de tudo se for pouco

        plt.figure(figsize=(12, 6))
        plt.title("Evolução da Recompensa Média por Episódio")
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa Média (Móvel)")
        plt.plot(data.index, data['r'], 'b.', alpha=0.2, label='Recompensa por Episódio')
        plt.plot(data.index, rolling_avg, 'r-', linewidth=2, label=f'Média Móvel ({window_size} episódios)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Não foi possível gerar o gráfico: {e}")

# --- Script Principal ---
if __name__ == "__main__":
    train_mode = True if len(sys.argv) > 1 and sys.argv[1] == "train" else False

    gym.register(
        id="gymnasium_env/GridWorld-Coverage-v0",
        entry_point=GridWorldCoverRenderEnv,
    )

    # --- PARÂMETROS PARA 20x20 ---
    GRID_SIZE = 20
    NUM_OBSTACLES = 40
    LOG_DIR = "log/ppo_coverage_20x20"
    MODEL_SAVE_PATH = "data/ppo_coverage_20x20.zip" # Nome do modelo salvo
    TOTAL_TIMESTEPS = 17_000_000 # Comece com 10M, ajuste se necessário

    if train_mode:
        print(f"--- Iniciando Treinamento: Grid {GRID_SIZE}x{GRID_SIZE}, Obstáculos: {NUM_OBSTACLES}, Passos: {TOTAL_TIMESTEPS} ---")
        env = make_env(render_mode=None, size=GRID_SIZE, log_dir=LOG_DIR, num_obstacles=NUM_OBSTACLES, seed=0)

        policy_kwargs = dict(
            features_extractor_class=CustomCnn,
            features_extractor_kwargs=dict(features_dim=128),
            # Redes um pouco maiores podem ajudar em grids maiores
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )

        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device="cuda", # ou cuda para a gpu nvidia
            policy_kwargs=policy_kwargs,
            ent_coef=0.02,
            learning_rate=3e-4, # Pode precisar diminuir se instável (ex: 1e-4)
            gamma=0.999,
            n_steps=4096, # Horizonte maior
            batch_size=128, # Batch size maior pode ajudar com n_steps maior
            n_epochs=10, # Padrão do PPO
            clip_range=0.2 # Padrão do PPO
        )

        # Configura logger (sobrescreve se já existir)
        logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)

        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
        except KeyboardInterrupt:
            print("\n🛑 Treinamento interrompido pelo usuário.")
        finally:
            # Garante que o diretório de dados existe
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            model.save(MODEL_SAVE_PATH)
            print(f"✅ Modelo salvo em {MODEL_SAVE_PATH}")
            plot_results(LOG_DIR) # Tenta plotar no final

    # --- Bloco de Teste ---
    print(f"📂 Carregando modelo salvo de '{MODEL_SAVE_PATH}'...")
    # Verifica se o modelo existe antes de carregar
    if not os.path.exists(MODEL_SAVE_PATH):
         print(f"❌ Erro: Arquivo do modelo não encontrado em '{MODEL_SAVE_PATH}'. Execute o treino primeiro.")
         sys.exit(1)
         
    model = PPO.load(MODEL_SAVE_PATH)

    # Cria ambiente para teste (com renderização)
    env = make_env(render_mode="human", size=GRID_SIZE, num_obstacles=NUM_OBSTACLES)

    for i in range(5): # Testa por 5 episódios
        print(f"\n--- Iniciando Teste {i+1} ---")
        obs, _ = env.reset()
        done, truncated = False, False
        steps = 0
        total_reward = 0
        visited_in_ep = set() # Track visited for this episode specifically in test

        # Adiciona a posição inicial ao conjunto visitado do teste
        agent_channel = obs[2]
        agent_pos_search = np.argwhere(agent_channel == 1.0)
        if agent_pos_search.shape[0] > 0:
             ai, aj = agent_pos_search[0]
             visited_in_ep.add((int(ai), int(aj)))
             
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            a = int(action.item()) if isinstance(action, np.ndarray) else int(action)

            obs, reward, done, truncated, info = env.step(a)
            total_reward += reward

            # Atualiza visitados para este episódio de teste
            agent_channel = obs[2]
            pos = np.argwhere(agent_channel == 1.0)
            if pos.shape[0] > 0:
                 ai, aj = int(pos[0,0]), int(pos[0,1])
                 visited_in_ep.add((ai, aj))

            # Atualiza a renderização (se o wrapper não fizer isso automaticamente)
            # Acessa o ambiente base através de env.env se estiver usando Monitor + Wrapper
            try:
                 # Tenta acessar o método do ambiente base para renderizar 'visited'
                 env.unwrapped.set_visited_cells(visited_in_ep)
            except AttributeError:
                 # Fallback se a estrutura for diferente
                 try:
                      env.env.unwrapped.set_visited_cells(visited_in_ep)
                 except AttributeError:
                     pass # Renderização pode não mostrar visitados corretamente
                     
            steps += 1
            env.render() # Força a renderização a cada passo no modo 'human'

        print(f"Episódio finalizado em {steps} passos. Terminated={done}, Truncated={truncated}. Recompensa total: {total_reward:.2f}")
        
        # Pega informações de cobertura do ambiente base
        try:
             unwrapped_env = env.unwrapped # Tenta acessar diretamente
             # Recalcula obstáculos e total coberto no teste para garantir
             obstacles_test = unwrapped_env._obstacle_locations
             total_cover_test = unwrapped_env.size * unwrapped_env.size - len(obstacles_test)
             visited_cnt = len(visited_in_ep)
             print(f"Cobertura: {visited_cnt} de {total_cover_test} células acessíveis ({visited_cnt/total_cover_test*100:.1f}%).")
        except AttributeError:
             print("Não foi possível obter informações detalhadas de cobertura do ambiente.")


    env.close()
    print("\n--- Testes Concluídos ---")

