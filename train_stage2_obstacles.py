import sys
import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Importe tudo que precisa do outro arquivo
from train_stage1_no_obstacles import make_env, plot_results, CustomCnn


# --- Script Principal ---
if __name__ == "__main__":
    GRID_SIZE = 5
    LOG_DIR_STAGE2 = "log/ppo_stage2_final"
    MODEL_PATH_STAGE1 = "data/ppo_stage1_navigator.zip"
    MODEL_PATH_FINAL = "data/ppo_stage2_final"

    print("--- Iniciando Estágio 2: Treinamento de Adaptação com Obstáculos ---")

    # 1. Crie o ambiente final, com obstáculos
    env = make_env(render_mode=None, size=GRID_SIZE, log_dir=LOG_DIR_STAGE2, num_obstacles=3, seed=0)

    # 2. Carregue o modelo pré-treinado do Estágio 1
    print(f"📂 Carregando modelo base de {MODEL_PATH_STAGE1}...")
    model = PPO.load(MODEL_PATH_STAGE1, env=env) # Passar o novo 'env' aqui já o associa
    
    # 3. Configure o logger para o novo diretório de logs
    new_logger = configure(LOG_DIR_STAGE2, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # 4. Continue o treinamento no novo ambiente
    # Vamos treinar por mais 1 milhão de passos para ele se adaptar bem
    TRAINING_STEPS = 1_000_000

    try:
        model.learn(total_timesteps=TRAINING_STEPS, reset_num_timesteps=False, progress_bar=True)
    except KeyboardInterrupt:
        print("\n🛑 Treinamento interrompido.")
    finally:
        os.makedirs("data", exist_ok=True)
        model.save(MODEL_PATH_FINAL)
        print(f"✅ Modelo Final salvo em {MODEL_PATH_FINAL}")
        plot_results(LOG_DIR_STAGE2)