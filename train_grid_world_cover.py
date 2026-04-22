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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from gymnasium.spaces import Box
from gymnasium_env.grid_world_cover_render import GridWorldCoverRenderEnv


# ─────────────────────────────────────────────────────────────────────────────
# FIX #1 — Wrapper de Cobertura com observação achatada (MlpPolicy)
# ─────────────────────────────────────────────────────────────────────────────
class CoverageWrapper(gym.Wrapper):
    """
    Wrapper principal de cobertura.

    Correções aplicadas:
    - observation_space achatado (c*h*w,) → compatível com MlpPolicy
    - Canal de visitados (obs[3]) é RECONSTRUÍDO inteiramente a cada step,
      garantindo que o agente sempre enxergue toda a trajetória percorrida.
    - Recompensas rebalanceadas para sinal mais claro.
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.grid_size = env.observation_space.shape[1]
        self.max_steps = env.max_steps

        c, h, w = env.observation_space.shape
        self._c, self._h, self._w = c, h, w

        # FIX: espaço achatado para MlpPolicy
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(c * h * w,),
            dtype=np.float32,
        )
        self.action_space = env.action_space

        self.visited: set = set()
        self.total_coverable_cells: int = 0
        self.current_step: int = 0

        print(f"[Wrapper] Grid {self.grid_size}×{self.grid_size} | "
              f"obs shape: ({c}×{h}×{w}) → achatado: ({c*h*w},) | "
              f"max_steps: {self.max_steps}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _rebuild_visited_channel(self, obs: np.ndarray) -> np.ndarray:
        """Reconstrói o canal 3 (mapa de visitados) do zero a cada passo."""
        obs[3] = 0.0
        for (vi, vj) in self.visited:
            obs[3, vi, vj] = 1.0
        return obs

    def _flat(self, obs: np.ndarray) -> np.ndarray:
        return obs.flatten().astype(np.float32)

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.total_coverable_cells = info.get("reachable_cells_count", self._h * self._w)
        self.visited = set()
        self.current_step = 0
        self.prev_coverage_ratio = 0.0  # para reward shaping denso

        # Posição inicial do agente
        agent_pos = self._get_agent_pos(obs)
        if agent_pos:
            self.visited.add(agent_pos)

        # FIX: limpa e reconstrói canal de visitados
        obs = self._rebuild_visited_channel(obs)

        # Recompensas
        self.reward_new_cell     =  1.0
        self.reward_step_cost    = -0.005
        self.reward_revisit      = -0.05
        self.reward_hit_obstacle = -0.2
        self.reward_completion   =  20.0

        return self._flat(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        agent_pos = self._get_agent_pos(obs)

        # ── Cálculo de recompensa ─────────────────────────────────────────────
        reward = self.reward_step_cost

        if info.get("hit_obstacle", False):
            reward += self.reward_hit_obstacle

        if agent_pos is not None:
            if agent_pos not in self.visited:
                self.visited.add(agent_pos)
                reward += self.reward_new_cell
            else:
                reward += self.reward_revisit

        coverage_ratio = (
            len(self.visited) / self.total_coverable_cells
            if self.total_coverable_cells > 0 else 0.0
        )

        # ── Reward shaping denso (resolve horizonte longo) ────────────────────
        # Com gamma=0.995 e 1000 passos, o bônus de conclusão chega descontado
        # a quase zero no início do episódio. Este sinal dá feedback proporcional
        # ao progresso a cada passo, mantendo o gradiente sempre informativo.
        reward += (coverage_ratio - self.prev_coverage_ratio) * 5.0
        self.prev_coverage_ratio = coverage_ratio

        # Condição de vitória
        if coverage_ratio >= 0.98:
            terminated = True
            reward += self.reward_completion
            print(f"✅ Cobertura completa! "
                  f"{len(self.visited)}/{self.total_coverable_cells} "
                  f"({coverage_ratio*100:.1f}%)")

        elif self.current_step >= self.max_steps:
            truncated = True

        # FIX: reconstrói o canal de visitados COMPLETO (não só a célula atual)
        obs = self._rebuild_visited_channel(obs)

        return self._flat(obs), float(reward), bool(terminated), bool(truncated), info

    # ── utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_agent_pos(obs: np.ndarray):
        """Retorna (row, col) do agente a partir do canal 2, ou None."""
        positions = np.argwhere(obs[2] == 1.0)
        if positions.shape[0] > 0:
            return (int(positions[0, 0]), int(positions[0, 1]))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Callback de Coverage
# ─────────────────────────────────────────────────────────────────────────────
class CoverageLoggerCallback(BaseCallback):
    """Registra o coverage rate real ao fim de cada episódio."""

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, "coverage_log.csv")
        self._records: list = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                try:
                    env = self.training_env.envs[i]
                    # Atravessa Monitor → CoverageWrapper
                    wrapper = env
                    while hasattr(wrapper, "env") and not isinstance(wrapper, CoverageWrapper):
                        wrapper = wrapper.env
                    if isinstance(wrapper, CoverageWrapper):
                        visited = len(wrapper.visited)
                        total   = wrapper.total_coverable_cells
                        coverage = visited / total if total > 0 else 0.0
                        self._records.append({
                            "timestep":      self.num_timesteps,
                            "episode":       len(self._records) + 1,
                            "coverage_rate": coverage,
                            "visited":       visited,
                            "total":         total,
                        })
                except Exception:
                    pass
        return True

    def _on_training_end(self):
        if self._records:
            pd.DataFrame(self._records).to_csv(self.filepath, index=False)
            print(f"📊 Coverage log salvo em {self.filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Factory do ambiente
# ─────────────────────────────────────────────────────────────────────────────
def make_env(render_mode=None, size=None, log_dir=None,
             num_obstacles=None, seed=None):
    if size is None or num_obstacles is None:
        raise ValueError("Passe 'size' e 'num_obstacles' para make_env().")

    max_steps = max(1000, size * size * 10)  # mais fôlego para grids maiores

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


# ─────────────────────────────────────────────────────────────────────────────
# Plotagem de resultados
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(log_dir: str, grid_size=None, num_obstacles=None):
    try:
        title_suffix = ""
        if grid_size and num_obstacles is not None:
            title_suffix = f" — Grid {grid_size}×{grid_size}, {num_obstacles} obstáculos"

        monitor_files = [f for f in os.listdir(log_dir) if f.endswith("monitor.csv")]
        if not monitor_files:
            print(f"⚠️  Nenhum monitor.csv em {log_dir}")
            return

        monitor_path = os.path.join(log_dir, monitor_files[0])
        print(f"📊 Gerando gráficos de '{monitor_path}'…")
        monitor_data = pd.read_csv(monitor_path, skiprows=1)

        if monitor_data.empty or "r" not in monitor_data.columns:
            print("⚠️  monitor.csv vazio ou sem coluna 'r'.")
            return

        coverage_path = os.path.join(log_dir, "coverage_log.csv")
        coverage_data = (
            pd.read_csv(coverage_path)
            if os.path.exists(coverage_path) else None
        )

        def smooth_with_ci(series, window):
            mean = series.rolling(window=window, min_periods=1).mean()
            std  = series.rolling(window=window, min_periods=1).std().fillna(0)
            ci   = 1.96 * std / np.sqrt(window)
            return mean, mean - ci, mean + ci

        has_coverage = coverage_data is not None and not coverage_data.empty
        n_panels = 3 if has_coverage else 2
        WINDOW   = max(20, len(monitor_data) // 20)

        fig = plt.figure(figsize=(14, 5 * n_panels))
        fig.suptitle(
            f"Métricas de Treinamento — PPO{title_suffix}",
            fontsize=15, fontweight="bold", y=1.01,
        )
        gs = gridspec.GridSpec(n_panels, 1, hspace=0.45)

        # Painel 1 — Recompensa
        ax1 = fig.add_subplot(gs[0])
        episodes = monitor_data.index + 1
        rewards  = monitor_data["r"]
        mean_r, lo_r, hi_r = smooth_with_ci(rewards, WINDOW)
        ax1.scatter(episodes, rewards, color="steelblue", alpha=0.15, s=8, label="Episódio")
        ax1.plot(episodes, mean_r, color="steelblue", linewidth=2,
                 label=f"Média móvel ({WINDOW} ep.)")
        ax1.fill_between(episodes, lo_r, hi_r, alpha=0.2, color="steelblue", label="IC 95%")
        ax1.axhline(rewards.max(), color="green", linestyle="--", linewidth=1, alpha=0.6,
                    label=f"Máx: {rewards.max():.1f}")
        ax1.set_title("Recompensa por Episódio", fontweight="bold")
        ax1.set_xlabel("Episódio"); ax1.set_ylabel("Recompensa Total")
        ax1.legend(loc="upper left", fontsize=8); ax1.grid(True, alpha=0.3)

        # Painel 2 — Coverage Rate (opcional)
        if has_coverage:
            ax2 = fig.add_subplot(gs[1])
            cov_ep  = coverage_data["episode"]
            cov_val = coverage_data["coverage_rate"] * 100
            cov_w   = max(10, len(cov_val) // 20)
            mean_c, lo_c, hi_c = smooth_with_ci(cov_val, cov_w)
            ax2.scatter(cov_ep, cov_val, color="darkorange", alpha=0.2, s=8, label="Episódio")
            ax2.plot(cov_ep, mean_c, color="darkorange", linewidth=2,
                     label=f"Média móvel ({cov_w} ep.)")
            ax2.fill_between(cov_ep, lo_c, hi_c, alpha=0.2, color="darkorange", label="IC 95%")
            ax2.set_ylim(0, 105)
            ax2.set_title("Coverage Rate por Episódio (%)", fontweight="bold")
            ax2.set_xlabel("Episódio"); ax2.set_ylabel("Cobertura (%)")
            ax2.legend(loc="upper left", fontsize=8); ax2.grid(True, alpha=0.3)
            next_panel = 2
        else:
            print("ℹ️  coverage_log.csv não encontrado — painel omitido.")
            next_panel = 1

        # Painel 3 — Comprimento de episódio
        ax3 = fig.add_subplot(gs[next_panel])
        lengths = monitor_data["l"]
        mean_l, lo_l, hi_l = smooth_with_ci(lengths, WINDOW)
        ax3.scatter(episodes, lengths, color="mediumpurple", alpha=0.15, s=8, label="Episódio")
        ax3.plot(episodes, mean_l, color="mediumpurple", linewidth=2,
                 label=f"Média móvel ({WINDOW} ep.)")
        ax3.fill_between(episodes, lo_l, hi_l, alpha=0.2, color="mediumpurple", label="IC 95%")
        ax3.set_title("Comprimento de Episódio (passos)", fontweight="bold")
        ax3.set_xlabel("Episódio"); ax3.set_ylabel("Passos")
        ax3.legend(loc="upper left", fontsize=8); ax3.grid(True, alpha=0.3)

        out_path = os.path.join(log_dir, "training_progress.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"✅ Gráfico salvo em {out_path}")
        plt.show()

    except Exception as e:
        print(f"⚠️  Não foi possível gerar os gráficos: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Script principal
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    try:
        gym.register(
            id="gymnasium_env/GridWorld-Coverage-v0",
            entry_point="gymnasium_env.grid_world_cover_render:GridWorldCoverRenderEnv",
        )
    except gym.error.Error:
        pass

    # ── Parâmetros globais ────────────────────────────────────────────────────
    GRID_SIZE      = 10
    NUM_OBSTACLES  = 20
    N_ENVS         = 8          # FIX: paralelismo — acelera muito o treinamento
    LOG_DIR        = f"log/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}"
    MODEL_SAVE_PATH = f"data/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}.zip"
    TOTAL_TIMESTEPS = 7_000_000  # 7M para grid 10×10

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    if mode == "train":
        os.makedirs("data", exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        print(f"--- Treinamento: Grid {GRID_SIZE}×{GRID_SIZE} | "
              f"Obstáculos: {NUM_OBSTACLES} | "
              f"Envs paralelos: {N_ENVS} | "
              f"Steps: {TOTAL_TIMESTEPS:,} ---")

        # FIX: make_vec_env com N_ENVS ambientes paralelos
        vec_env = make_vec_env(
            lambda: make_env(
                size=GRID_SIZE,
                num_obstacles=NUM_OBSTACLES,
                log_dir=LOG_DIR,
            ),
            n_envs=N_ENVS,
        )

        # FIX: MlpPolicy — sem CNN, sem MaxPool, sem perda de informação espacial
        # MlpPolicy é mais rápida na CPU (SB3 avisa isso explicitamente para não-CNN)
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device="cpu",
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
            # ── Hiperparâmetros ───────────────────────────────────────────────
            ent_coef=0.03,  # exploração aumentada para grid maior
            learning_rate=3e-4,
            gamma=0.995,  # FIX: horizonte mais longo para episódios de ~1000 passos
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            vf_coef=0.6,
            clip_range=0.2,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
        )

        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=LOG_DIR,
            name_prefix="ppo_coverage_checkpoint",
        )
        coverage_cb = CoverageLoggerCallback(log_dir=LOG_DIR)

        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                progress_bar=True,
                callback=[checkpoint_cb, coverage_cb],
            )
        except KeyboardInterrupt:
            print("\n🛑 Treinamento interrompido pelo usuário.")
        finally:
            os.makedirs("data", exist_ok=True)  # garante que existe mesmo após crash
            model.save(MODEL_SAVE_PATH)
            print(f"✅ Modelo salvo em {MODEL_SAVE_PATH}")
            plot_results(LOG_DIR, grid_size=GRID_SIZE, num_obstacles=NUM_OBSTACLES)

    # ── TEST ──────────────────────────────────────────────────────────────────
    elif mode == "test":
        print(f"--- Teste: Grid {GRID_SIZE}×{GRID_SIZE} | Obstáculos: {NUM_OBSTACLES} ---")
        print(f"📂 Carregando modelo de '{MODEL_SAVE_PATH}'…")

        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"❌ Modelo não encontrado em '{MODEL_SAVE_PATH}'. "
                  f"Execute 'python {sys.argv[0]} train' primeiro.")
            sys.exit(1)

        model = PPO.load(MODEL_SAVE_PATH)
        print("✅ Modelo carregado.")

        env = make_env(
            render_mode="human",
            size=GRID_SIZE,
            num_obstacles=NUM_OBSTACLES,
            log_dir=None,
            seed=np.random.randint(1000),
        )

        num_test_episodes  = 5
        total_coverage_sum = 0.0

        for i in range(num_test_episodes):
            print(f"\n--- Episódio de teste {i+1}/{num_test_episodes} ---")
            obs, info = env.reset()
            done = truncated = False
            steps = 0
            total_reward = 0.0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                a = int(action.item()) if isinstance(action, np.ndarray) else int(action)
                obs, reward, done, truncated, info = env.step(a)
                total_reward += reward
                steps += 1

                # Atualiza visualização com células visitadas
                if env.render_mode == "human":
                    wrapper = env
                    while hasattr(wrapper, "env") and not isinstance(wrapper, CoverageWrapper):
                        wrapper = wrapper.env
                    if isinstance(wrapper, CoverageWrapper):
                        try:
                            env.unwrapped.set_visited_cells(set(wrapper.visited))
                        except AttributeError:
                            pass
                    env.render()

            print(f"  Passos:      {steps}")
            print(f"  Terminated:  {done}  | Truncated: {truncated}")
            print(f"  Recompensa:  {total_reward:.2f}")

            # Recupera wrapper para cobertura real
            wrapper = env
            while hasattr(wrapper, "env") and not isinstance(wrapper, CoverageWrapper):
                wrapper = wrapper.env

            if isinstance(wrapper, CoverageWrapper):
                total_c = wrapper.total_coverable_cells
                visited  = len(wrapper.visited)
                pct      = (visited / total_c * 100) if total_c > 0 else 0.0
                total_coverage_sum += pct
                print(f"  Cobertura:   {visited}/{total_c} células ({pct:.1f}%)")

        avg_coverage = total_coverage_sum / num_test_episodes
        print(f"\n📊 Cobertura média: {avg_coverage:.1f}%")
        env.close()
        print("\n--- Testes concluídos ---")

    else:
        print(f"Modo inválido '{mode}'. Use 'train' ou 'test'.")
        sys.exit(1)