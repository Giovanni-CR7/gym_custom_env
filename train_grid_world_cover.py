# python train_grid_world_cover.py train  # treinar
# python train_grid_world_cover.py test   # testar

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
# PARÂMETROS GLOBAIS
# ─────────────────────────────────────────────────────────────────────────────
GRID_SIZE       = 20
NUM_OBSTACLES   = 40
VISION_RADIUS   = 2          # visão 5×5 = agente ± 2 células em cada direção
N_ENVS          = 8
LOG_DIR         = f"log/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}"
MODEL_PATH      = f"data/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}.zip"
TOTAL_TIMESTEPS = 20_000_000  # 20M para grid 20×20


# ─────────────────────────────────────────────────────────────────────────────
# CNN para 20×20 — stride=2 na camada do meio em vez de MaxPool
# MaxPool(2) em sequência destruiria resolução; stride controlado preserva.
# ─────────────────────────────────────────────────────────────────────────────
class GridCNN(BaseFeaturesExtractor):
    """
    Entrada: (4, 20, 20)
    Conv1 stride=1 → (32, 20, 20)
    Conv2 stride=2 → (64, 10, 10)   ← única redução, ainda informativa
    Conv3 stride=1 → (64, 10, 10)
    Flatten → 6400 → Linear(256)
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c,  32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
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
# WRAPPER — visão parcial 5×5 + mapa global de visitados
# ─────────────────────────────────────────────────────────────────────────────
class CoverageWrapper(gym.Wrapper):
    """
    Observação: 4 canais × 20×20
      Canal 0: espaço livre   — apenas células dentro da janela 5×5 reveladas
      Canal 1: obstáculos     — apenas células dentro da janela 5×5 reveladas
      Canal 2: posição global do agente (sempre completo)
      Canal 3: mapa global de visitados (sempre completo)

    Isso simula um agente com sensor local de 5×5 e memória de mapa global
    (equivalente a SLAM simplificado), que é o setup realista para cobertura.
    """

    def __init__(self, env, vision_radius: int = 2):
        super().__init__(env)
        self.vision_radius = vision_radius
        self.grid_size     = env.observation_space.shape[1]
        self.max_steps     = env.max_steps

        self.observation_space = env.observation_space   # (4, H, W) — sem alterar
        self.action_space      = env.action_space

        # Estado interno
        self.visited:   set = set()
        self.revealed:  set = set()   # células já vistas pelo sensor 5×5
        self.total_coverable_cells: int = 0
        self.current_step:   int   = 0
        self.prev_coverage:  float = 0.0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _agent_pos(self, obs: np.ndarray):
        pos = np.argwhere(obs[2] == 1.0)
        return (int(pos[0, 0]), int(pos[0, 1])) if len(pos) > 0 else None

    def _update_revealed(self, ar: int, ac: int):
        """Marca como reveladas todas as células dentro da janela 5×5."""
        r0 = max(0, ar - self.vision_radius)
        r1 = min(self.grid_size - 1, ar + self.vision_radius)
        c0 = max(0, ac - self.vision_radius)
        c1 = min(self.grid_size - 1, ac + self.vision_radius)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                self.revealed.add((r, c))

    def _apply_fog(self, obs: np.ndarray) -> np.ndarray:
        """
        Canais 0 e 1: mostra apenas células já reveladas pelo sensor 5×5.
        Canais 2 e 3: mantém informação global (agente + visitados).
        """
        out = np.zeros_like(obs)
        for (r, c) in self.revealed:
            out[0, r, c] = obs[0, r, c]
            out[1, r, c] = obs[1, r, c]
        out[2] = obs[2]   # posição global do agente
        out[3] = obs[3]   # mapa global de visitados
        return out

    def _rebuild_visited_channel(self, obs: np.ndarray) -> np.ndarray:
        obs[3] = 0.0
        for r, c in self.visited:
            obs[3, r, c] = 1.0
        return obs

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.total_coverable_cells = info.get(
            "reachable_cells_count", self.grid_size * self.grid_size
        )
        self.visited      = set()
        self.revealed     = set()
        self.current_step = 0
        self.prev_coverage = 0.0

        ap = self._agent_pos(obs)
        if ap:
            self.visited.add(ap)
            self._update_revealed(ap[0], ap[1])

        obs = self._rebuild_visited_channel(obs)
        obs = self._apply_fog(obs)
        return obs.copy(), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        ap = self._agent_pos(obs)

        # Atualiza célula visitada e área revelada pelo sensor
        if ap is not None:
            self.visited.add(ap)
            self._update_revealed(ap[0], ap[1])

        # ── Recompensa ────────────────────────────────────────────────────────
        reward = -0.01   # step cost

        if info.get("hit_obstacle", False):
            reward -= 0.3

        coverage = (len(self.visited) / self.total_coverable_cells
                    if self.total_coverable_cells > 0 else 0.0)

        # Shaping progressivo: células novas perto de 100% valem até 5× mais
        progress = coverage - self.prev_coverage
        multiplier = 1.0 + coverage * 4.0
        reward += progress * 10.0 * multiplier
        self.prev_coverage = coverage

        # Bônus por célula nova (sinal esparso adicional)
        if ap is not None and progress > 0:
            reward += 2.0

        # Vitória
        if coverage >= 0.98:
            terminated = True
            efficiency = (self.max_steps - self.current_step) / self.max_steps
            reward += 50.0 + efficiency * 30.0
            print(f"✅ {len(self.visited)}/{self.total_coverable_cells} "
                  f"({coverage*100:.1f}%) em {self.current_step} passos")
        elif self.current_step >= self.max_steps:
            truncated = True

        # Reconstrói canais globais e aplica fog of war
        obs = self._rebuild_visited_channel(obs)
        obs = self._apply_fog(obs)
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
                    w = self.training_env.envs[i]
                    while hasattr(w, "env") and not isinstance(w, CoverageWrapper):
                        w = w.env
                    if isinstance(w, CoverageWrapper) and w.total_coverable_cells > 0:
                        v = len(w.visited)
                        t = w.total_coverable_cells
                        self._records.append({
                            "timestep":      self.num_timesteps,
                            "episode":       len(self._records) + 1,
                            "coverage_rate": v / t,
                            "visited":       v,
                            "total":         t,
                            "steps":         w.current_step,
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
def make_env(size, num_obstacles, vision_radius=2,
             render_mode=None, log_dir=None, seed=None):
    # Para 20×20: 6000 passos máximos (15× o número de células)
    max_steps = size * size * 15
    env = GridWorldCoverRenderEnv(
        size=size,
        render_mode=render_mode,
        num_obstacles=num_obstacles,
        seed=seed,
        max_steps=max_steps,
    )
    env = CoverageWrapper(env, vision_radius=vision_radius)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Plotagem
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(log_dir, grid_size=None, num_obstacles=None):
    try:
        suffix = (f" — Grid {grid_size}×{grid_size}, {num_obstacles} obs."
                  if grid_size else "")
        mfiles = [f for f in os.listdir(log_dir) if f.endswith("monitor.csv")]
        if not mfiles:
            print("⚠️  Nenhum monitor.csv.")
            return
        mdata = pd.read_csv(os.path.join(log_dir, mfiles[0]), skiprows=1)
        if mdata.empty or "r" not in mdata.columns:
            return

        cpath = os.path.join(log_dir, "coverage_log.csv")
        cdata = pd.read_csv(cpath) if os.path.exists(cpath) else None
        has_cov = cdata is not None and not cdata.empty

        def smooth(s, w):
            m   = s.rolling(w, min_periods=1).mean()
            std = s.rolling(w, min_periods=1).std().fillna(0)
            ci  = 1.96 * std / np.sqrt(w)
            return m, m - ci, m + ci

        W = max(20, len(mdata) // 20)
        n = 3 if has_cov else 2
        fig = plt.figure(figsize=(14, 5 * n))
        fig.suptitle(f"PPO — Cobertura 20×20 com Visão 5×5{suffix}",
                     fontsize=14, fontweight="bold", y=1.01)
        gs = gridspec.GridSpec(n, 1, hspace=0.45)
        ep = mdata.index + 1

        ax = fig.add_subplot(gs[0])
        r = mdata["r"]
        m, lo, hi = smooth(r, W)
        ax.scatter(ep, r, color="steelblue", alpha=0.15, s=6)
        ax.plot(ep, m, color="steelblue", lw=2, label=f"Média ({W} ep.)")
        ax.fill_between(ep, lo, hi, alpha=0.2, color="steelblue")
        ax.axhline(r.max(), color="green", ls="--", lw=1, alpha=0.6,
                   label=f"Máx: {r.max():.1f}")
        ax.set_title("Recompensa por Episódio", fontweight="bold")
        ax.set_xlabel("Episódio"); ax.set_ylabel("Recompensa")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        if has_cov:
            ax = fig.add_subplot(gs[1])
            cv  = cdata["coverage_rate"] * 100
            cep = cdata["episode"]
            cw  = max(10, len(cv) // 20)
            m, lo, hi = smooth(cv, cw)
            ax.scatter(cep, cv, color="darkorange", alpha=0.2, s=6)
            ax.plot(cep, m, color="darkorange", lw=2, label=f"Média ({cw} ep.)")
            ax.fill_between(cep, lo, hi, alpha=0.2, color="darkorange")
            ax.axhline(98, color="red", ls="--", lw=1, alpha=0.6, label="Meta 98%")
            ax.set_ylim(0, 105)
            ax.set_title("Coverage Rate (%)", fontweight="bold")
            ax.set_xlabel("Episódio"); ax.set_ylabel("Cobertura (%)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[2 if has_cov else 1])
        l = mdata["l"]
        m, lo, hi = smooth(l, W)
        ax.scatter(ep, l, color="mediumpurple", alpha=0.15, s=6)
        ax.plot(ep, m, color="mediumpurple", lw=2, label=f"Média ({W} ep.)")
        ax.fill_between(ep, lo, hi, alpha=0.2, color="mediumpurple")
        ax.set_title("Comprimento de Episódio", fontweight="bold")
        ax.set_xlabel("Episódio"); ax.set_ylabel("Passos")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        out = os.path.join(log_dir, "training_progress.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✅ Gráfico → {out}")
        plt.show()
    except Exception as e:
        print(f"⚠️  Erro gráfico: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
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

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    if mode == "train":
        os.makedirs("data", exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        print(f"--- Grid {GRID_SIZE}×{GRID_SIZE} | Visão 5×5 | "
              f"{NUM_OBSTACLES} obstáculos | {N_ENVS} envs | "
              f"{TOTAL_TIMESTEPS:,} steps ---")

        vec_env = make_vec_env(
            lambda: make_env(GRID_SIZE, NUM_OBSTACLES,
                             vision_radius=VISION_RADIUS,
                             log_dir=LOG_DIR),
            n_envs=N_ENVS,
        )

        policy_kwargs = dict(
            features_extractor_class=GridCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        )

        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            device="cuda",
            policy_kwargs=policy_kwargs,

            learning_rate=3e-4,
            gamma=0.999,       # episódios longos (até 6000 passos) — desconto lento
            ent_coef=0.01,
            n_steps=1024,      # 8 envs × 1024 = 8192 amostras por rollout
            batch_size=512,
            n_epochs=10,
            vf_coef=0.5,
            clip_range=0.2,
            max_grad_norm=0.5,
            target_kl=0.02,    # para atualização se KL > 0.02 — evita degradação
            tensorboard_log=LOG_DIR,
        )

        print(f"\n📐 CNN: (4,{GRID_SIZE},{GRID_SIZE}) → Conv(32,s1) → "
              f"Conv(64,s2) → Conv(64,s1) → flat → Linear(256)")
        print(f"👁  Visão: janela {2*VISION_RADIUS+1}×{2*VISION_RADIUS+1} "
              f"com fog-of-war acumulativo\n")

        callbacks = [
            CheckpointCallback(save_freq=100_000, save_path=LOG_DIR,
                               name_prefix="ckpt"),
            CoverageLoggerCallback(log_dir=LOG_DIR),
        ]

        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                progress_bar=True,
                callback=callbacks,
            )
        except KeyboardInterrupt:
            print("\n🛑 Interrompido.")
        finally:
            os.makedirs("data", exist_ok=True)
            model.save(MODEL_PATH)
            print(f"✅ Modelo → {MODEL_PATH}")
            plot_results(LOG_DIR, grid_size=GRID_SIZE,
                         num_obstacles=NUM_OBSTACLES)

    # ── TEST ──────────────────────────────────────────────────────────────────
    elif mode == "test":
        print(f"--- Teste: Grid {GRID_SIZE}×{GRID_SIZE} | Visão 5×5 ---")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Modelo não encontrado: {MODEL_PATH}")
            sys.exit(1)

        model = PPO.load(MODEL_PATH)
        print("✅ Modelo carregado.")

        env = make_env(
            GRID_SIZE, NUM_OBSTACLES,
            vision_radius=VISION_RADIUS,
            render_mode="human",
            seed=np.random.randint(10000),
        )

        total_cov   = 0.0
        total_steps = 0.0
        N = 5

        for i in range(N):
            print(f"\n--- Episódio {i+1}/{N} ---")
            obs, _ = env.reset()
            done = truncated = False
            steps = 0
            total_r = 0.0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = env.step(int(action))
                total_r += r
                steps   += 1

                if env.render_mode == "human":
                    # Passa células visitadas ao env base para renderização
                    w = env
                    while hasattr(w, "env") and not isinstance(w, CoverageWrapper):
                        w = w.env
                    if isinstance(w, CoverageWrapper):
                        try:
                            env.unwrapped.set_visited_cells(set(w.visited))
                        except AttributeError:
                            pass
                    env.render()

            print(f"  Passos:     {steps}")
            print(f"  Recompensa: {total_r:.1f}")
            print(f"  Terminado:  {done} | Truncado: {truncated}")

            w = env
            while hasattr(w, "env") and not isinstance(w, CoverageWrapper):
                w = w.env
            if isinstance(w, CoverageWrapper) and w.total_coverable_cells > 0:
                pct = len(w.visited) / w.total_coverable_cells * 100
                total_cov   += pct
                total_steps += steps
                print(f"  Cobertura:  {len(w.visited)}/{w.total_coverable_cells}"
                      f" ({pct:.1f}%)")

        print(f"\n📊 Cobertura média: {total_cov/N:.1f}%")
        print(f"📊 Passos médios:   {total_steps/N:.0f}")
        env.close()

    else:
        print("Use 'train' ou 'test'.")
        sys.exit(1)