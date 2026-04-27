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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from gymnasium_env.grid_world_cover_render import GridWorldCoverRenderEnv


# ─────────────────────────────────────────────────────────────────────────────
# CNN SEM MaxPool — preserva toda a informação espacial do grid
# ─────────────────────────────────────────────────────────────────────────────
# O erro do código original era usar 2× MaxPool(2) em grids pequenos:
#   7×7  → 3×3 → 1×1   (toda informação destruída)
#   10×10 → 5×5 → 2×2  (quase toda informação destruída)
#
# Solução: Conv com padding=1 mantém o tamanho. Sem MaxPool.
# Para 10×10: cada célula mantém contexto dos vizinhos sem perder resolução.
# ─────────────────────────────────────────────────────────────────────────────
class SmallGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape  # ex: (4, 10, 10)

        self.cnn = nn.Sequential(
            # Camada 1: extrai features locais (quais vizinhos estão livres/visitados)
            nn.Conv2d(c, 32, kernel_size=3, padding=1),   # → (32, h, w)
            nn.ReLU(),
            # Camada 2: combina features locais em padrões regionais
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64, h, w)
            nn.ReLU(),
            # Camada 3: raciocínio de maior alcance
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → (64, h, w)
            nn.ReLU(),
            nn.Flatten(),                                  # → 64*h*w
        )

        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, c, h, w)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(obs))


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper de Cobertura
# ─────────────────────────────────────────────────────────────────────────────
class CoverageWrapper(gym.Wrapper):
    """
    Mantém obs como (4, H, W) para a CNN — sem flatten.
    Canal 3 é reconstruído inteiramente a cada step (bug crítico do original).
    """

    def __init__(self, env):
        super().__init__(env)
        self.grid_size = env.observation_space.shape[1]
        self.max_steps = env.max_steps

        # Observação mantida como (C, H, W) — CNN precisa da estrutura espacial
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.visited: set = set()
        self.total_coverable_cells: int = 0
        self.current_step: int = 0
        self.prev_coverage: float = 0.0

    def _rebuild_visited_channel(self, obs: np.ndarray) -> np.ndarray:
        obs[3] = 0.0
        for r, c in self.visited:
            obs[3, r, c] = 1.0
        return obs

    @staticmethod
    def _agent_pos(obs: np.ndarray):
        pos = np.argwhere(obs[2] == 1.0)
        return (int(pos[0, 0]), int(pos[0, 1])) if len(pos) > 0 else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.total_coverable_cells = info.get("reachable_cells_count", obs.shape[1] * obs.shape[2])
        self.visited = set()
        self.current_step = 0
        self.prev_coverage = 0.0

        ap = self._agent_pos(obs)
        if ap:
            self.visited.add(ap)

        obs = self._rebuild_visited_channel(obs)
        return obs.copy(), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        ap = self._agent_pos(obs)

        # ── Recompensa ────────────────────────────────────────────────────────
        reward = -0.01  # custo de passo (desencoraja loops infinitos)

        if info.get("hit_obstacle", False):
            reward -= 0.3  # penalidade clara por bater em obstáculo

        if ap is not None:
            if ap not in self.visited:
                self.visited.add(ap)
                reward += 3.0   # sinal forte por célula nova
            # SEM penalidade de revisita: o agente precisa atravessar células
            # já visitadas para chegar em regiões inexploradas. Penalizar isso
            # torna a exploração de longa distância não-lucrativa e o agente
            # aprende a ficar em clusters locais em vez de avançar.

        coverage = len(self.visited) / self.total_coverable_cells if self.total_coverable_cells > 0 else 0.0

        # Shaping denso: recompensa proporcional ao progresso incremental
        # Resolve o problema do horizonte longo (gamma^1000 ≈ 0)
        reward += (coverage - self.prev_coverage) * 10.0
        self.prev_coverage = coverage

        # Vitória
        if coverage >= 0.98:
            terminated = True
            reward += 50.0
            print(f"✅ {len(self.visited)}/{self.total_coverable_cells} ({coverage*100:.1f}%)")
        elif self.current_step >= self.max_steps:
            truncated = True

        obs = self._rebuild_visited_channel(obs)
        return obs.copy(), float(reward), bool(terminated), bool(truncated), info


# ─────────────────────────────────────────────────────────────────────────────
# Callback de Coverage
# ─────────────────────────────────────────────────────────────────────────────
class CoverageLoggerCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.filepath = os.path.join(log_dir, "coverage_log.csv")
        self._records: list = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                try:
                    wrapper = self.training_env.envs[i]
                    while hasattr(wrapper, "env") and not isinstance(wrapper, CoverageWrapper):
                        wrapper = wrapper.env
                    if isinstance(wrapper, CoverageWrapper) and wrapper.total_coverable_cells > 0:
                        v = len(wrapper.visited)
                        t = wrapper.total_coverable_cells
                        self._records.append({
                            "timestep": self.num_timesteps,
                            "episode":  len(self._records) + 1,
                            "coverage_rate": v / t,
                            "visited": v,
                            "total":   t,
                        })
                except Exception:
                    pass
        return True

    def _on_training_end(self):
        if self._records:
            pd.DataFrame(self._records).to_csv(self.filepath, index=False)
            print(f"📊 Coverage log → {self.filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def make_env(size, num_obstacles, render_mode=None, log_dir=None, seed=None):
    # Passos generosos: garante que o agente tem tempo de cobrir tudo
    max_steps = size * size * 15
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
# Plotagem
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(log_dir, grid_size=None, num_obstacles=None):
    try:
        suffix = f" — Grid {grid_size}×{grid_size}, {num_obstacles} obstáculos" if grid_size else ""
        mfiles = [f for f in os.listdir(log_dir) if f.endswith("monitor.csv")]
        if not mfiles:
            print("⚠️  Nenhum monitor.csv encontrado.")
            return

        mdata = pd.read_csv(os.path.join(log_dir, mfiles[0]), skiprows=1)
        if mdata.empty or "r" not in mdata.columns:
            print("⚠️  monitor.csv vazio.")
            return

        cpath = os.path.join(log_dir, "coverage_log.csv")
        cdata = pd.read_csv(cpath) if os.path.exists(cpath) else None

        def smooth(s, w):
            m = s.rolling(w, min_periods=1).mean()
            std = s.rolling(w, min_periods=1).std().fillna(0)
            ci = 1.96 * std / np.sqrt(w)
            return m, m - ci, m + ci

        has_cov = cdata is not None and not cdata.empty
        n = 3 if has_cov else 2
        W = max(20, len(mdata) // 20)
        fig = plt.figure(figsize=(14, 5 * n))
        fig.suptitle(f"Métricas — PPO{suffix}", fontsize=15, fontweight="bold", y=1.01)
        gs = gridspec.GridSpec(n, 1, hspace=0.45)

        ax1 = fig.add_subplot(gs[0])
        ep = mdata.index + 1
        r = mdata["r"]
        m, lo, hi = smooth(r, W)
        ax1.scatter(ep, r, color="steelblue", alpha=0.15, s=8)
        ax1.plot(ep, m, color="steelblue", lw=2, label=f"Média ({W} ep.)")
        ax1.fill_between(ep, lo, hi, alpha=0.2, color="steelblue")
        ax1.axhline(r.max(), color="green", ls="--", lw=1, alpha=0.6, label=f"Máx: {r.max():.1f}")
        ax1.set_title("Recompensa", fontweight="bold")
        ax1.set_xlabel("Episódio"); ax1.set_ylabel("Recompensa")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        if has_cov:
            ax2 = fig.add_subplot(gs[1])
            cv = cdata["coverage_rate"] * 100
            cep = cdata["episode"]
            cw = max(10, len(cv) // 20)
            m, lo, hi = smooth(cv, cw)
            ax2.scatter(cep, cv, color="darkorange", alpha=0.2, s=8)
            ax2.plot(cep, m, color="darkorange", lw=2, label=f"Média ({cw} ep.)")
            ax2.fill_between(cep, lo, hi, alpha=0.2, color="darkorange")
            ax2.set_ylim(0, 105)
            ax2.set_title("Coverage Rate (%)", fontweight="bold")
            ax2.set_xlabel("Episódio"); ax2.set_ylabel("Cobertura (%)")
            ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
            next_p = 2
        else:
            next_p = 1

        ax3 = fig.add_subplot(gs[next_p])
        l = mdata["l"]
        m, lo, hi = smooth(l, W)
        ax3.scatter(ep, l, color="mediumpurple", alpha=0.15, s=8)
        ax3.plot(ep, m, color="mediumpurple", lw=2, label=f"Média ({W} ep.)")
        ax3.fill_between(ep, lo, hi, alpha=0.2, color="mediumpurple")
        ax3.set_title("Comprimento de Episódio", fontweight="bold")
        ax3.set_xlabel("Episódio"); ax3.set_ylabel("Passos")
        ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

        out = os.path.join(log_dir, "training_progress.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✅ Gráfico → {out}")
        plt.show()
    except Exception as e:
        print(f"⚠️  Erro ao gerar gráfico: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
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

    # ── PARÂMETROS ────────────────────────────────────────────────────────────
    GRID_SIZE       = 10
    NUM_OBSTACLES   = 20
    N_ENVS          = 8
    LOG_DIR         = f"log/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}"
    MODEL_PATH      = f"data/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}.zip"
    TOTAL_TIMESTEPS = 7_000_000

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    if mode == "train":
        os.makedirs("data", exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        print(f"--- Grid {GRID_SIZE}×{GRID_SIZE} | {NUM_OBSTACLES} obstáculos | "
              f"{N_ENVS} envs | {TOTAL_TIMESTEPS:,} steps ---")

        vec_env = make_vec_env(
            lambda: make_env(GRID_SIZE, NUM_OBSTACLES, log_dir=LOG_DIR),
            n_envs=N_ENVS,
        )

        policy_kwargs = dict(
            features_extractor_class=SmallGridCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        )

        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            device="cuda",       # CNN se beneficia da GPU
            policy_kwargs=policy_kwargs,

            # ── Hiperparâmetros ───────────────────────────────────────────────
            learning_rate=3e-4,
            gamma=0.995,         # horizonte longo para episódios de ~1500 passos
            ent_coef=0.01,       # CNN converge melhor, não precisa de ent_coef alto
            n_steps=512,         # rollout menor → atualiza mais vezes → aprende mais rápido
            batch_size=256,
            n_epochs=10,
            vf_coef=0.5,
            clip_range=0.2,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
        )

        print(f"\n📐 Arquitetura CNN:")
        print(f"   Input:  (4, {GRID_SIZE}, {GRID_SIZE})")
        print(f"   Conv1:  (32, {GRID_SIZE}, {GRID_SIZE})  ← sem MaxPool, resolução preservada")
        print(f"   Conv2:  (64, {GRID_SIZE}, {GRID_SIZE})")
        print(f"   Conv3:  (64, {GRID_SIZE}, {GRID_SIZE})")
        print(f"   Flat:   {64 * GRID_SIZE * GRID_SIZE}  → Linear → 256")
        print(f"   Policy: [256, 128] | Value: [256, 128]\n")

        ckpt_cb     = CheckpointCallback(save_freq=100_000, save_path=LOG_DIR,
                                         name_prefix="ckpt")
        coverage_cb = CoverageLoggerCallback(log_dir=LOG_DIR)

        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                progress_bar=True,
                callback=[ckpt_cb, coverage_cb],
            )
        except KeyboardInterrupt:
            print("\n🛑 Interrompido.")
        finally:
            os.makedirs("data", exist_ok=True)
            model.save(MODEL_PATH)
            print(f"✅ Modelo → {MODEL_PATH}")
            plot_results(LOG_DIR, grid_size=GRID_SIZE, num_obstacles=NUM_OBSTACLES)

    # ── TEST ──────────────────────────────────────────────────────────────────
    elif mode == "test":
        print(f"--- Teste: Grid {GRID_SIZE}×{GRID_SIZE} ---")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Modelo não encontrado: {MODEL_PATH}")
            sys.exit(1)

        model = PPO.load(MODEL_PATH)
        print("✅ Modelo carregado.")

        env = make_env(GRID_SIZE, NUM_OBSTACLES,
                       render_mode="human",
                       seed=np.random.randint(1000))

        total_cov = 0.0
        N = 5

        for i in range(N):
            print(f"\n--- Episódio {i+1}/{N} ---")
            obs, _ = env.reset()
            done = truncated = False
            steps = total_r = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = env.step(int(action))
                total_r += r; steps += 1
                if env.render_mode == "human":
                    w = env
                    while hasattr(w, "env") and not isinstance(w, CoverageWrapper):
                        w = w.env
                    if isinstance(w, CoverageWrapper):
                        try: env.unwrapped.set_visited_cells(set(w.visited))
                        except AttributeError: pass
                    env.render()

            print(f"  Passos: {steps} | Recompensa: {total_r:.1f} | "
                  f"Terminado: {done} | Truncado: {truncated}")

            w = env
            while hasattr(w, "env") and not isinstance(w, CoverageWrapper):
                w = w.env
            if isinstance(w, CoverageWrapper) and w.total_coverable_cells > 0:
                pct = len(w.visited) / w.total_coverable_cells * 100
                total_cov += pct
                print(f"  Cobertura: {len(w.visited)}/{w.total_coverable_cells} ({pct:.1f}%)")

        print(f"\n📊 Cobertura média: {total_cov/N:.1f}%")
        env.close()
    else:
        print("Use 'train' ou 'test'.")
        sys.exit(1)