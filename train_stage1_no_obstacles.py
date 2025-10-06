# train_stage1_no_obstacles.py

import sys
import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from train_grid_world_cover import CoverageWrapper, CustomCnn
from gymnasium_env.grid_world_cover_render import GridWorldCoverRenderEnv

def plot_results(log_dir):
    try:
        monitor_files = [f for f in os.listdir(log_dir) if f.endswith("monitor.csv")]
        if not monitor_files:
            print(f"⚠️ Nenhum arquivo monitor.csv encontrado em {log_dir}")
            return
        
        filepath = os.path.join(log_dir, monitor_files[0])
        data = pd.read_csv(filepath, skiprows=1)
        
        window_size = 100
        rolling_avg = data['r'].rolling(window=window_size).mean()

        plt.figure(figsize=(12, 6))
        plt.title("Estágio 1: Evolução da Recompensa (Sem Obstáculos)")
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

# Factory do ambiente, igual ao anterior
def make_env(render_mode=None, size=5, log_dir=None, num_obstacles=0, seed=None):
    env = GridWorldCoverRenderEnv(size=size, render_mode=render_mode, num_obstacles=num_obstacles, seed=seed)
    env = CoverageWrapper(env)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
    return env

# --- Script Principal ---
if __name__ == "__main__":
    GRID_SIZE = 5
    LOG_DIR_STAGE1 = "log/ppo_stage1_navigator"
    MODEL_PATH_STAGE1 = "data/ppo_stage1_navigator"

    print("--- Iniciando Estágio 1: Treinamento sem Obstáculos ---")

    # A principal mudança está aqui: num_obstacles=0
    env = make_env(render_mode=None, size=GRID_SIZE, log_dir=LOG_DIR_STAGE1, num_obstacles=0, seed=0)

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
        ent_coef=0.02, # Usando o valor ajustado
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048 # Usando o valor ajustado
    )

    TRAINING_STEPS = 500_000
    
    try:
        model.learn(total_timesteps=TRAINING_STEPS, progress_bar=True)
    except KeyboardInterrupt:
        print("\n🛑 Treinamento interrompido.")
    finally:
        os.makedirs("data", exist_ok=True)
        model.save(MODEL_PATH_STAGE1)
        print(f"✅ Modelo do Estágio 1 salvo em {MODEL_PATH_STAGE1}")
        plot_results(LOG_DIR_STAGE1)