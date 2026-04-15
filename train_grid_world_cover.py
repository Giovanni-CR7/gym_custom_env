# python train_grid_world_cover.py train  # para treinar
# python train_grid_world_cover.py test   # para testar

import sys
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch as th
import torch.nn as nn
import os

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
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
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.grid_size = env.observation_space.shape[1]
        self.max_steps = env.max_steps
        self.visited = set()
        self.total_coverable_cells = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print(f"[Wrapper] Grid Size: {self.grid_size}, Max Steps: {self.max_steps}")

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

        # --- Recompensas normalizadas: tudo relativo ao número de células alcançáveis ---
        N = self.total_coverable_cells

        # Cada célula nova vale uma fração do bônus de conclusão
        # → descobrir todas as células equivale a ~50% do bônus de conclusão
        self.reward_new_cell     = 50.0 / N      # ex: 49 células → ~1.02 por célula

        # Penalidade por revisitar: pequena o suficiente pra não paralisar o agente
        self.reward_revisit      = -self.reward_new_cell * 0.1

        # Custo por passo: incentiva eficiência sem punir exploração
        self.reward_step_cost    = -self.reward_new_cell * 0.05

        # Bônus ao completar: significativo, mas não obscurece o aprendizado incremental
        self.reward_completion   = 100.0

        # Bônus parcial ao truncar: proporcional à cobertura, teto em 50% do bônus de conclusão
        self.reward_truncation_max = 50.0

        return self._get_obs(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        # Localiza agente
        pos = np.argwhere(obs[2] == 1.0)
        agent_pos = None
        if pos.shape[0] > 0:
            agent_pos = (int(pos[0, 0]), int(pos[0, 1]))
        else:
            try:
                agent_pos = tuple(self.env.unwrapped._agent_location)
            except Exception:
                pass

        # --- Recompensa base: custo de existir incentiva eficiência ---
        reward = self.reward_step_cost

        if agent_pos is not None:
            if agent_pos not in self.visited:
                self.visited.add(agent_pos)
                coverage_ratio = len(self.visited) / self.total_coverable_cells

                # Recompensa de descoberta com bônus progressivo nas últimas células
                # (incentiva "fechar" a cobertura e não parar em 80%)
                discovery_bonus = 1.0 + coverage_ratio  # vai de 1.0x a 2.0x
                reward += self.reward_new_cell * discovery_bonus
            else:
                reward += self.reward_revisit

        coverage_ratio = (
            len(self.visited) / self.total_coverable_cells
            if self.total_coverable_cells > 0 else 0
        )

        # --- Condições de término ---

        # Sucesso: cobertura completa
        if coverage_ratio >= 0.98:
            terminated = True
            reward += self.reward_completion
            print(f"✅ Cobertura completa! {len(self.visited)}/{self.total_coverable_cells} ({coverage_ratio*100:.1f}%)")

        # Limite de passos
        elif self.current_step >= self.max_steps:
            truncated = True
            # Bônus proporcional, mas só relevante perto de 100%
            # coverage_ratio² penaliza coberturas baixas e premia coberturas altas
            reward += (coverage_ratio ** 2) * self.reward_truncation_max

        if agent_pos is not None:
            obs[3, agent_pos[0], agent_pos[1]] = 1.0

        return self._get_obs(obs), float(reward), bool(terminated), bool(truncated), info

# --- Callback de Coverage ---
class CoverageLoggerCallback(BaseCallback):
    """Salva o coverage rate real ao fim de cada episódio."""
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, "coverage_log.csv")
        self._episode_coverage = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                env = self.training_env.envs[i]
                try:
                    wrapper = env.env if hasattr(env, 'env') else env
                    visited = len(wrapper.visited)
                    total = wrapper.total_coverable_cells
                    coverage = visited / total if total > 0 else 0
                    self._episode_coverage.append({
                        "timestep": self.num_timesteps,
                        "episode": len(self._episode_coverage) + 1,
                        "coverage_rate": coverage,
                        "visited": visited,
                        "total": total,
                    })
                except Exception:
                    pass
        return True

    def _on_training_end(self):
        if self._episode_coverage:
            pd.DataFrame(self._episode_coverage).to_csv(self.filepath, index=False)
            print(f"📊 Coverage log salvo em {self.filepath}")

# --- Factory do Ambiente ---
def make_env(render_mode=None, size=None, log_dir=None, num_obstacles=None, seed=None):
    if size is None or num_obstacles is None:
        raise ValueError("Passe 'size' e 'num_obstacles' explicitamente para make_env().")
    # max_steps generoso e com piso mínimo — funciona para qualquer grid
    max_steps = max(500, size * size * 6)
    env = GridWorldCoverRenderEnv(
        size=size,
        render_mode=render_mode,
        num_obstacles=num_obstacles,
        seed=seed,
        max_steps=max_steps,
    )
    env = CoverageWrapper(env)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
    return env

# --- Função de Plotagem ---
def plot_results(log_dir, grid_size=None, num_obstacles=None):
    try:
        # --- Título dinâmico ---
        title_suffix = ""
        if grid_size and num_obstacles is not None:
            title_suffix = f" — Grid {grid_size}×{grid_size}, {num_obstacles} obstáculos"

        # --- Carrega monitor.csv ---
        monitor_files = [f for f in os.listdir(log_dir) if f.endswith("monitor.csv")]
        if not monitor_files:
            print(f"⚠️ Nenhum arquivo monitor.csv encontrado em {log_dir}")
            return

        monitor_path = os.path.join(log_dir, monitor_files[0])
        print(f"📊 Gerando gráficos a partir de '{monitor_path}'...")
        monitor_data = pd.read_csv(monitor_path, skiprows=1)

        if monitor_data.empty or 'r' not in monitor_data.columns:
            print(f"⚠️ Arquivo {monitor_path} está vazio ou não contém a coluna de recompensa 'r'.")
            return

        # --- Carrega coverage_log.csv (opcional) ---
        coverage_path = os.path.join(log_dir, "coverage_log.csv")
        coverage_data = None
        if os.path.exists(coverage_path):
            coverage_data = pd.read_csv(coverage_path)

        def smooth_with_ci(series, window):
            """Retorna média móvel e intervalo de confiança 95%."""
            mean = series.rolling(window=window, min_periods=1).mean()
            std  = series.rolling(window=window, min_periods=1).std().fillna(0)
            ci   = 1.96 * std / np.sqrt(window)
            return mean, mean - ci, mean + ci

        has_coverage = coverage_data is not None and not coverage_data.empty
        n_panels = 3 if has_coverage else 2
        WINDOW = max(20, len(monitor_data) // 20)  # janela adaptativa

        fig = plt.figure(figsize=(14, 5 * n_panels))
        fig.suptitle(
            f"Métricas de Treinamento — PPO{title_suffix}",
            fontsize=15, fontweight='bold', y=1.01
        )
        gs = gridspec.GridSpec(n_panels, 1, hspace=0.45)

        # ── Painel 1: Recompensa ──────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        episodes = monitor_data.index + 1
        rewards  = monitor_data['r']
        mean_r, lo_r, hi_r = smooth_with_ci(rewards, WINDOW)

        ax1.scatter(episodes, rewards, color='steelblue', alpha=0.15, s=8, label='Episódio')
        ax1.plot(episodes, mean_r, color='steelblue', linewidth=2,
                 label=f'Média móvel ({WINDOW} ep.)')
        ax1.fill_between(episodes, lo_r, hi_r, alpha=0.2, color='steelblue', label='IC 95%')
        ax1.axhline(rewards.max(), color='green', linestyle='--', linewidth=1, alpha=0.6,
                    label=f'Máx: {rewards.max():.1f}')
        ax1.set_title("Recompensa por Episódio", fontweight='bold')
        ax1.set_xlabel("Episódio")
        ax1.set_ylabel("Recompensa Total")
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Painel 2: Coverage Rate ───────────────────────────────
        if has_coverage:
            ax2 = fig.add_subplot(gs[1])
            cov_ep  = coverage_data['episode']
            cov_val = coverage_data['coverage_rate'] * 100  # em %
            cov_window = max(10, len(cov_val) // 20)
            mean_c, lo_c, hi_c = smooth_with_ci(cov_val, cov_window)

            ax2.scatter(cov_ep, cov_val, color='darkorange', alpha=0.2, s=8, label='Episódio')
            ax2.plot(cov_ep, mean_c, color='darkorange', linewidth=2,
                     label=f'Média móvel ({cov_window} ep.)')
            ax2.fill_between(cov_ep, lo_c, hi_c, alpha=0.2, color='darkorange', label='IC 95%')
            ax2.set_ylim(0, 105)
            ax2.set_title("Coverage Rate por Episódio (%)", fontweight='bold')
            ax2.set_xlabel("Episódio")
            ax2.set_ylabel("Cobertura (%)")
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
            next_panel = 2
        else:
            print("ℹ️  coverage_log.csv não encontrado — painel de coverage omitido.")
            next_panel = 1

        # ── Painel 3: Comprimento de episódio (eficiência) ────────
        ax3 = fig.add_subplot(gs[next_panel])
        lengths = monitor_data['l']
        mean_l, lo_l, hi_l = smooth_with_ci(lengths, WINDOW)

        ax3.scatter(episodes, lengths, color='mediumpurple', alpha=0.15, s=8, label='Episódio')
        ax3.plot(episodes, mean_l, color='mediumpurple', linewidth=2,
                 label=f'Média móvel ({WINDOW} ep.)')
        ax3.fill_between(episodes, lo_l, hi_l, alpha=0.2, color='mediumpurple', label='IC 95%')
        ax3.set_title("Comprimento de Episódio (passos)", fontweight='bold')
        ax3.set_xlabel("Episódio")
        ax3.set_ylabel("Passos")
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)

        out_path = os.path.join(log_dir, 'training_progress.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✅ Gráfico salvo em {out_path}")
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
    TOTAL_TIMESTEPS = 10_000_000

    if mode == "train":
        os.makedirs("data", exist_ok=True)
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
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
            device="cuda",  # cpu ou cuda
            policy_kwargs=policy_kwargs,

            # parâmetros de exploração
            ent_coef=0.05,  # aumentado para mais exploração

            # Taxa de aprendizado com schedule (decai ao longo do tempo)
            learning_rate=lambda f: 3e-4 * f,  # Começa em 3e-4 e decai linearmente

            # Desconto temporal
            gamma=0.999,

            # Coleta de experiência (CRITICAL FIX)
            n_steps=2048,  # coleta os dados em blocos de 512 passos
            batch_size=256,  # Proporcional ao n_steps
            n_epochs=10,  # responsável por quantas vezes o modelo vê cada dado coletado

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
        coverage_callback = CoverageLoggerCallback(log_dir=LOG_DIR)

        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                progress_bar=True,
                callback=[checkpoint_callback, coverage_callback]
            )
        except KeyboardInterrupt:
            print("\n🛑 Treinamento interrompido pelo usuário.")
        finally:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            model.save(MODEL_SAVE_PATH)
            print(f"✅ Modelo salvo em {MODEL_SAVE_PATH}")
            plot_results(LOG_DIR, grid_size=GRID_SIZE, num_obstacles=NUM_OBSTACLES)

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