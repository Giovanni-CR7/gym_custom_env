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

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium_env.grid_world_cover_render import GridWorldCoverRenderEnv

class CustomCnn(BaseFeaturesExtractor):
    """ CNN customizada para extrair features da observação em grid. """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            sample = th.as_tensor(np.zeros((1, c, h, w), dtype=np.float32))
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.cnn(observations)
        return self.linear(x)

# --- wrapper de Cobertura ---
class CoverageWrapper(gym.Wrapper):
    """ Aplica a lógica de recompensa otimizada para cobertura de área. """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.grid_size = env.observation_space.shape[1]
        self.max_steps = env.max_steps
        self.visited = set()
        self.total_coverable_cells = 0 
        self.steps_since_new_cell = 0  # contador para detectar estagnação

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print(f"[Wrapper] Inicializado. Grid Size: {self.grid_size}, Max Steps: {self.max_steps}")

    def _get_obs(self, obs):
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.total_coverable_cells = info['reachable_cells_count']
        self.visited = set()
        
        agent_channel = obs[2]
        agent_pos_search = np.argwhere(agent_channel == 1.0)
        if agent_pos_search.shape[0] > 0:
             ai, aj = agent_pos_search[0]
             self.visited.add((int(ai), int(aj)))

        self.current_step = 0
        self.steps_since_new_cell = 0
        return self._get_obs(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        self.steps_since_new_cell += 1

        agent_channel = obs[2]
        pos = np.argwhere(agent_channel == 1.0)
        agent_pos = None
        if pos.shape[0] > 0:
            ai, aj = int(pos[0,0]), int(pos[0,1])
            agent_pos = (ai, aj)
        else:
             try: agent_pos = tuple(self.env.unwrapped._agent_location)
             except Exception: pass

        hit_obstacle = info.get("hit_obstacle", False)
        
        # --- SISTEMA DE RECOMPENSAS---
        reward = -0.005  

        #if hit_obstacle:
           # reward += -0.05  
        if agent_pos is not None:
            if agent_pos not in self.visited:
                self.visited.add(agent_pos)
                coverage_ratio = len(self.visited) / self.total_coverable_cells
                
                # Recompensa progressiva e generosa para incentivar exploração
                base_discovery = 2.0
                bonus = coverage_ratio * 5.0  # Aumenta muito no final
                reward += base_discovery + bonus  # De 2.0 até 7.0
                self.steps_since_new_cell = 0
            else:
                reward += -0.3

        # Calcula cobertura atual
        coverage_ratio = len(self.visited) / self.total_coverable_cells if self.total_coverable_cells > 0 else 0
        
        # --- CONDIÇÕES DE TÉRMINO  ---
        
        # Sucesso: cobertura completa
        if coverage_ratio >= 0.98:  # 98% já é excelente
            terminated = True
            reward += 200.0  # Grande recompensa por completar
            print(f"✅ Cobertura completa! {len(self.visited)}/{self.total_coverable_cells} células ({coverage_ratio*100:.1f}%)")
        
        # Limite de passos (sem término por estagnação - deixa o agente explorar)
        elif self.current_step >= self.max_steps:
            truncated = True
            # Bônus parcial proporcional à cobertura
            reward += coverage_ratio * 100.0

        # Atualiza visualização de células visitadas
        if agent_pos is not None:
             obs[3, agent_pos[0], agent_pos[1]] = 1.0

        return self._get_obs(obs), float(reward), bool(terminated), bool(truncated), info

# --- Factory do Ambiente ---
def make_env(render_mode=None, size=20, log_dir=None, num_obstacles=40, seed=None):
    env = GridWorldCoverRenderEnv(size=size, render_mode=render_mode, num_obstacles=num_obstacles, seed=seed)
    env = CoverageWrapper(env)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
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
             print(f"⚠️ Arquivo {filepath} está vazio ou não contém a coluna de recompensa 'r'.")
             return

        window_size = 100
        if len(data['r']) >= window_size:
            rolling_avg = data['r'].rolling(window=window_size).mean()
        elif len(data['r']) > 0:
             rolling_avg = data['r'].rolling(window=len(data['r'])).mean()
        else:
             rolling_avg = pd.Series()

        plt.figure(figsize=(12, 6))
        plt.title("Evolução da Recompensa Média por Episódio")
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa Média (Móvel)")
        plt.plot(data.index, data['r'], 'b.', alpha=0.15, label='Recompensa por Episódio')
        if not rolling_avg.empty:
             plt.plot(data.index, rolling_avg, 'r-', linewidth=2, label=f'Média Móvel ({window_size} episódios)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_progress.png'))
        plt.show()
    except Exception as e:
        print(f"⚠️ Não foi possível gerar o gráfico completo: {e}")

# --- Script Principal ---
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    try:
        gym.register(
            id="gymnasium_env/GridWorld-Coverage-v0",
            entry_point="gymnasium_env.grid_world_cover_render:GridWorldCoverRenderEnv",
        )
    except gym.error.Error:
        pass

    # --- PARÂMETROS GLOBAIS ---
    GRID_SIZE = 20
    NUM_OBSTACLES = 40
    LOG_DIR = f"log/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}"
    MODEL_SAVE_PATH = f"data/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}.zip"
    TOTAL_TIMESTEPS = 5_000_000  # reduzido para testes mais rápidos

    if mode == "train":
        print(f"--- Iniciando Treinamento: Grid {GRID_SIZE}x{GRID_SIZE}, Obstáculos: {NUM_OBSTACLES}, Passos: {TOTAL_TIMESTEPS} ---")
        env = make_env(render_mode=None, size=GRID_SIZE, log_dir=LOG_DIR, num_obstacles=NUM_OBSTACLES, seed=0)

        policy_kwargs = dict(
            features_extractor_class=CustomCnn,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )

        # --- HIPERPARÂMETROS ---
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device="cuda" ,  # cpu ou cuda
            policy_kwargs=policy_kwargs,
            
            # parâmetros de exploração
            ent_coef=0.02,  # aumentado para mais exploração
            
            # Taxa de aprendizado com schedule (decai ao longo do tempo)
            learning_rate=lambda f: 3e-4 * f,  # Começa em 3e-4 e decai linearmente
            
            # Desconto temporal
            gamma=0.99,  # reduzi de 
            
            # Coleta de experiência (CRITICAL FIX)
            n_steps=512,  # Reduzido drasticamente de 4096
            batch_size=64,  # Proporcional ao n_steps
            n_epochs=10,  # Reduzido de 20 para convergência mais rápida
            
            # Clipping
            clip_range=0.2,
            clip_range_vf=None,  # Sem clipping na value function
            
            # Otimizador
            max_grad_norm=0.5,
            
            # Logging
            tensorboard_log=LOG_DIR
        )

        # Callback para salvar checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=LOG_DIR,
            name_prefix='ppo_coverage_checkpoint'
        )

        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS, 
                progress_bar=True,
                callback=checkpoint_callback
            )
        except KeyboardInterrupt:
            print("\n🛑 Treinamento interrompido pelo usuário.")
        finally:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            model.save(MODEL_SAVE_PATH)
            print(f"✅ Modelo salvo em {MODEL_SAVE_PATH}")
            plot_results(LOG_DIR)

    elif mode == "test":
        print(f"--- Iniciando Modo de Teste: Grid {GRID_SIZE}x{GRID_SIZE}, Obstáculos: {NUM_OBSTACLES} ---")
        print(f"📂 Carregando modelo treinado de '{MODEL_SAVE_PATH}'...")

        if not os.path.exists(MODEL_SAVE_PATH):
             print(f"❌ Erro: Arquivo do modelo não encontrado em '{MODEL_SAVE_PATH}'. Execute o treino primeiro ('python {sys.argv[0]} train').")
             sys.exit(1)

        model = PPO.load(MODEL_SAVE_PATH)
        print("✅ Modelo carregado.")
        
        env = make_env(render_mode="human", size=GRID_SIZE, num_obstacles=NUM_OBSTACLES, log_dir=None, seed=np.random.randint(1000))

        num_test_episodes = 5
        total_coverage_sum = 0
        
        for i in range(num_test_episodes):
            print(f"\n--- Iniciando Teste {i+1}/{num_test_episodes} ---")
            obs, info = env.reset()
            done, truncated = False, False
            steps = 0
            total_reward = 0
            
            current_visited_for_render = set(env.visited) if hasattr(env, 'visited') else set()

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                a = int(action.item()) if isinstance(action, np.ndarray) else int(action)

                obs, reward, done, truncated, info = env.step(a)
                total_reward += reward
                steps += 1
                
                if env.render_mode == "human":
                    if hasattr(env, 'visited'):
                         current_visited_for_render = set(env.visited)
                    try:
                        env.unwrapped.set_visited_cells(current_visited_for_render)
                    except AttributeError: pass
                    env.render()

            print(f"Episódio finalizado em {steps} passos.")
            print(f"  Terminated (Cobertura Completa): {done}")
            print(f"  Truncated (Limite/Estagnação): {truncated}")
            print(f"  Recompensa Total: {total_reward:.2f}")

            try:
                 total_cover_test = env.total_coverable_cells
                 visited_cnt = len(env.visited)
                 coverage_percent = (visited_cnt / total_cover_test * 100) if total_cover_test > 0 else 0
                 total_coverage_sum += coverage_percent
                 print(f"  Cobertura Final: {visited_cnt} de {total_cover_test} células acessíveis ({coverage_percent:.1f}%).")
            except Exception as e:
                 print(f"  Não foi possível calcular a cobertura detalhada: {e}")

        avg_coverage = total_coverage_sum / num_test_episodes
        print(f"\n📊 Média de Cobertura: {avg_coverage:.1f}%")
        env.close()
        print("\n--- Testes Concluídos ---")
    else:
        print(f"Modo inválido '{mode}'. Use 'train' ou 'test'.")
        sys.exit(1)

