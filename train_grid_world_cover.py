# python train_grid_world_cover.py train   # para treinar
# python train_grid_world_cover.py test    # para testar

import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import pandas as pd
import os
from gymnasium_env.grid_world_cover_render import GridWorldCoverRenderEnv

# ---------------------------
# Custom CNN feature extractor
# ---------------------------
class CustomCnn(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
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

# ---------------------------
# Wrapper que usa a observação em canais e aplica recompensa
# ---------------------------
class CoverageWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # assume env.observation_space = Box(0,1, shape=(4,H,W))
        self.env = env
        self.grid_size = env.observation_space.shape[1]
        self.max_steps = env.max_steps if hasattr(env, "max_steps") else 200
        self.visited = set()
        self.obstacles = set()
        self.current_step = 0
        self.total_coverable_cells = self.grid_size * self.grid_size

        # wrapper observation: keep same channel-format as env (C,H,W)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _get_obs(self, obs):
        # env already returns channels [free, obstacles, agent, visited]
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # read obstacles from info (env provides set)
        self.obstacles = set(info.get("obstacles", []))
        # compute total coverable cells
        self.total_coverable_cells = self.grid_size * self.grid_size - len(self.obstacles)
        # initialize visited from channel or from env.visited_cells
        # obs channel 3 is visited mask
        visited_mask = obs[3]
        self.visited = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if visited_mask[i, j] > 0.5:
                    self.visited.add((i, j))
        # also ensure agent pos is in visited
        agent_channel = obs[2]
        ai, aj = np.argwhere(agent_channel == 1.0)[0]
        self.visited.add((int(ai), int(aj)))
        self.current_step = 0
        return self._get_obs(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        # agent pos from channel 2
        agent_channel = obs[2]
        pos = np.argwhere(agent_channel == 1.0)
        if pos.shape[0] == 0:
            # fallback: if not found, try to inspect env attribute
            try:
                ai, aj = tuple(self.env._agent_location)
            except Exception:
                ai, aj = 0, 0
        else:
            ai, aj = int(pos[0,0]), int(pos[0,1])
        agent_pos = (ai, aj)

        hit_obstacle = info.get("hit_obstacle", False)

        # reward shaping (inspired by repo)
        reward = -0.01  # small step cost
        if hit_obstacle:
            reward += -0.01
        else:
            if agent_pos not in self.visited:
                self.visited.add(agent_pos)
                reward += 1.0
            else:
                reward += -0.1

        # terminal bonus for full coverage
        if len(self.visited) >= self.total_coverable_cells:
            terminated = True
            reward += 50.0

        # truncation by max steps
        if self.current_step >= self.max_steps:
            truncated = True

        # update obs visited channel for consistency (so render shows visited)
        # (obs is numpy array; update channel 3)
        obs[3, ai, aj] = 1.0

        return self._get_obs(obs), float(reward), bool(terminated), bool(truncated), info

# ---------------------------
# Factory
# ---------------------------

def make_env(render_mode=None, size=5, log_dir=None, num_obstacles=3, seed=None):
    env = GridWorldCoverRenderEnv(size=size, render_mode=render_mode, num_obstacles=num_obstacles, seed=seed)
    env = CoverageWrapper(env)
    if log_dir is not None:
        env = Monitor(env, log_dir)
    return env


def plot_results(log_dir):
    """
    Plota a recompensa média por episódio a partir do arquivo monitor.csv.
    """
    try:
        # Encontra o arquivo de monitor no diretório de log
        monitor_files = [f for f in os.listdir(log_dir) if f.endswith("monitor.csv")]
        if not monitor_files:
            print(f"⚠️ Nenhum arquivo monitor.csv encontrado em {log_dir}")
            return
        
        # Carrega os dados usando pandas
        filepath = os.path.join(log_dir, monitor_files[0])
        data = pd.read_csv(filepath, skiprows=1) # Pula o cabeçalho inicial
        
        # Calcula a média móvel para suavizar o gráfico (janela de 100 episódios)
        window_size = 100
        # 'r' é a coluna de recompensa no monitor.csv
        rolling_avg = data['r'].rolling(window=window_size).mean()

        plt.figure(figsize=(10, 5))
        plt.title("Evolução da Recompensa Média por Episódio")
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa Média (Móvel)")
        
        # Plota a recompensa original (mais ruidosa)
        plt.plot(data['r'], 'b.', alpha=0.2, label='Recompensa por Episódio')
        # Plota a média móvel (linha contínua)
        plt.plot(rolling_avg, 'r-', linewidth=2, label=f'Média Móvel ({window_size} episódios)')
        
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ Não foi possível gerar o gráfico: {e}")
        
# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    train = True if len(sys.argv) > 1 and sys.argv[1] == "train" else False

    # register same id as before
    gym.register(
        id="gymnasium_env/GridWorld-Coverage-v0",
        entry_point=GridWorldCoverRenderEnv,
    )

    GRID_SIZE = 5
    LOG_DIR = "log/ppo_coverage_obstacles"

    if train:
        env = make_env(render_mode=None, size=GRID_SIZE, log_dir=LOG_DIR, num_obstacles=3, seed=0)

        policy_kwargs = dict(
            features_extractor_class=CustomCnn,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[128, 64], vf=[128, 64])
        )

        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device="cpu",
            policy_kwargs=policy_kwargs,
            ent_coef=0.02,
            learning_rate=3e-4,
            gamma=0.999,
            n_steps=1024,
        )

        logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)

        try:
            model.learn(total_timesteps=1_500_000)
        except KeyboardInterrupt:
            print("\n🛑 Treinamento interrompido pelo usuário.")
        finally:
            model.save("data/ppo_coverage_obstacles")
            print("✅ Modelo salvo em data/ppo_coverage_obstacles")
            plot_results(LOG_DIR)

    # Test
    print("📂 Carregando modelo salvo...")
    model = PPO.load("data/ppo_coverage_obstacles")

    env = make_env(render_mode="human", size=GRID_SIZE, num_obstacles=3)

    for i in range(3):
        print(f"\n--- Iniciando Teste {i+1} ---")
        obs, _ = env.reset()
        done, truncated = False, False
        steps = 0
        total_reward = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            # normalize action to int
            try:
                a = int(np.asarray(action).item())
            except Exception:
                if hasattr(action, "__iter__"):
                    a = int(list(action)[0])
                else:
                    a = int(action)
            obs, reward, done, truncated, info = env.step(a)
            total_reward += reward
            # update renderable visited
            if hasattr(env, 'visited'):
                env.unwrapped.set_visited_cells(env.unwrapped.visited_cells)
            else:
                # wrapper stores visited; try to set
                if hasattr(env, 'env') and hasattr(env.env, 'visited'):
                    env.unwrapped.set_visited_cells(env.env.visited)

            steps += 1

        print(f"Episódio finalizado em {steps} passos. Recompensa total: {total_reward:.2f}")
        # try to read visited count robustly
        try:
            visited_cnt = len(env.unwrapped.visited_cells)
            total_cover = env.unwrapped.size * env.unwrapped.size - len(env.unwrapped._obstacle_locations)
        except Exception:
            visited_cnt = len(env.env.visited)
            total_cover = env.env.total_coverable_cells
        print(f"Cobertura: {visited_cnt} de {total_cover} células acessíveis.")

    env.close()

