# python train_grid_world_cover.py train     # treino do zero
# python train_grid_world_cover.py finetune  # continua do melhor checkpoint
# python train_grid_world_cover.py test      # testa o modelo salvo

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
# CNN SEM MaxPool
# ─────────────────────────────────────────────────────────────────────────────
class SmallGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, c, h, w)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(obs))


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper de Cobertura
# ─────────────────────────────────────────────────────────────────────────────
class CoverageWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.grid_size    = env.observation_space.shape[1]
        self.max_steps    = env.max_steps
        self.observation_space = env.observation_space
        self.action_space      = env.action_space

        self.visited: set         = set()
        self.total_coverable_cells: int = 0
        self.current_step: int    = 0
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
        self.total_coverable_cells = info.get("reachable_cells_count",
                                               obs.shape[1] * obs.shape[2])
        self.visited      = set()
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

        # ── Recompensa base ───────────────────────────────────────────────────
        reward = -0.01  # step cost

        if info.get("hit_obstacle", False):
            reward -= 0.3

        if ap is not None and ap not in self.visited:
            self.visited.add(ap)
            reward += 3.0

        coverage = (len(self.visited) / self.total_coverable_cells
                    if self.total_coverable_cells > 0 else 0.0)

        # ── MELHORIA 1: Shaping progressivo ──────────────────────────────────
        # Quanto mais próximo de 100%, mais valiosa é cada célula nova.
        # Na última célula (coverage=0.98→0.99): multiplicador ≈ 4.9×
        # Na primeira célula (coverage=0→0.01):  multiplicador ≈ 1.0×
        progress = coverage - self.prev_coverage
        progressive_multiplier = 1.0 + coverage * 4.0
        reward += progress * 10.0 * progressive_multiplier
        self.prev_coverage = coverage

        # ── Vitória ───────────────────────────────────────────────────────────
        if coverage >= 0.98:
            terminated = True
            # MELHORIA 2: Bônus de eficiência — premia terminar mais rápido
            efficiency = (self.max_steps - self.current_step) / self.max_steps
            reward += 50.0 + efficiency * 20.0  # até +20 extra por velocidade
            print(f"✅ {len(self.visited)}/{self.total_coverable_cells} "
                  f"({coverage*100:.1f}%) em {self.current_step} passos")

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
                            "timestep":     self.num_timesteps,
                            "episode":      len(self._records) + 1,
                            "coverage_rate": v / t,
                            "visited":      v,
                            "total":        t,
                            "steps":        wrapper.current_step,
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
        suffix = (f" — Grid {grid_size}×{grid_size}, {num_obstacles} obstáculos"
                  if grid_size else "")
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
            m   = s.rolling(w, min_periods=1).mean()
            std = s.rolling(w, min_periods=1).std().fillna(0)
            ci  = 1.96 * std / np.sqrt(w)
            return m, m - ci, m + ci

        has_cov = cdata is not None and not cdata.empty
        has_steps = has_cov and "steps" in cdata.columns
        n_panels = 2 + int(has_cov) + int(has_steps)
        W = max(20, len(mdata) // 20)

        fig = plt.figure(figsize=(14, 5 * n_panels))
        fig.suptitle(f"Métricas — PPO{suffix}", fontsize=15,
                     fontweight="bold", y=1.01)
        gs = gridspec.GridSpec(n_panels, 1, hspace=0.45)
        panel = 0

        # Painel: Recompensa
        ax = fig.add_subplot(gs[panel]); panel += 1
        ep = mdata.index + 1
        r  = mdata["r"]
        m, lo, hi = smooth(r, W)
        ax.scatter(ep, r, color="steelblue", alpha=0.15, s=8)
        ax.plot(ep, m, color="steelblue", lw=2, label=f"Média ({W} ep.)")
        ax.fill_between(ep, lo, hi, alpha=0.2, color="steelblue")
        ax.axhline(r.max(), color="green", ls="--", lw=1, alpha=0.6,
                   label=f"Máx: {r.max():.1f}")
        ax.set_title("Recompensa", fontweight="bold")
        ax.set_xlabel("Episódio"); ax.set_ylabel("Recompensa")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Painel: Coverage Rate
        if has_cov:
            ax = fig.add_subplot(gs[panel]); panel += 1
            cv  = cdata["coverage_rate"] * 100
            cep = cdata["episode"]
            cw  = max(10, len(cv) // 20)
            m, lo, hi = smooth(cv, cw)
            ax.scatter(cep, cv, color="darkorange", alpha=0.2, s=8)
            ax.plot(cep, m, color="darkorange", lw=2, label=f"Média ({cw} ep.)")
            ax.fill_between(cep, lo, hi, alpha=0.2, color="darkorange")
            ax.axhline(98, color="red", ls="--", lw=1, alpha=0.5,
                       label="Meta: 98%")
            ax.set_ylim(0, 105)
            ax.set_title("Coverage Rate (%)", fontweight="bold")
            ax.set_xlabel("Episódio"); ax.set_ylabel("Cobertura (%)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Painel: Passos por episódio (eficiência)
        if has_steps:
            ax = fig.add_subplot(gs[panel]); panel += 1
            st  = cdata["steps"]
            sep = cdata["episode"]
            sw  = max(10, len(st) // 20)
            m, lo, hi = smooth(st, sw)
            ax.scatter(sep, st, color="mediumseagreen", alpha=0.2, s=8)
            ax.plot(sep, m, color="mediumseagreen", lw=2,
                    label=f"Média ({sw} ep.)")
            ax.fill_between(sep, lo, hi, alpha=0.2, color="mediumseagreen")
            ax.set_title("Passos até Cobertura ≥98% (eficiência)",
                         fontweight="bold")
            ax.set_xlabel("Episódio"); ax.set_ylabel("Passos")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Painel: Comprimento de episódio (monitor)
        ax = fig.add_subplot(gs[panel]); panel += 1
        l = mdata["l"]
        m, lo, hi = smooth(l, W)
        ax.scatter(ep, l, color="mediumpurple", alpha=0.15, s=8)
        ax.plot(ep, m, color="mediumpurple", lw=2, label=f"Média ({W} ep.)")
        ax.fill_between(ep, lo, hi, alpha=0.2, color="mediumpurple")
        ax.set_title("Comprimento de Episódio (todos)", fontweight="bold")
        ax.set_xlabel("Episódio"); ax.set_ylabel("Passos")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        out = os.path.join(log_dir, "training_progress.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✅ Gráfico → {out}")
        plt.show()
    except Exception as e:
        print(f"⚠️  Erro ao gerar gráfico: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de treino
# ─────────────────────────────────────────────────────────────────────────────
def _policy_kwargs():
    return dict(
        features_extractor_class=SmallGridCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

def _make_vec(grid_size, num_obstacles, n_envs, log_dir):
    return make_vec_env(
        lambda: make_env(grid_size, num_obstacles, log_dir=log_dir),
        n_envs=n_envs,
    )

def _callbacks(log_dir, save_freq=50_000):
    return [
        CheckpointCallback(save_freq=save_freq, save_path=log_dir,
                           name_prefix="ckpt"),
        CoverageLoggerCallback(log_dir=log_dir),
    ]


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
    GRID_SIZE     = 10
    NUM_OBSTACLES = 20
    N_ENVS        = 8
    LOG_DIR       = f"log/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}"
    MODEL_PATH    = f"data/ppo_coverage_{GRID_SIZE}x{GRID_SIZE}.zip"

    # Checkpoint mais próximo de 6.1M steps — ajuste se necessário
    CHECKPOINT    = f"{LOG_DIR}/ckpt_6100000_steps.zip"

    # ── TRAIN (do zero) ───────────────────────────────────────────────────────
    if mode == "train":
        TOTAL = 7_000_000
        os.makedirs("data", exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        print(f"--- Treino do zero | Grid {GRID_SIZE}×{GRID_SIZE} | "
              f"{N_ENVS} envs | {TOTAL:,} steps ---")

        vec_env = _make_vec(GRID_SIZE, NUM_OBSTACLES, N_ENVS, LOG_DIR)

        model = PPO(
            "CnnPolicy", vec_env,
            verbose=1, device="cuda",
            policy_kwargs=_policy_kwargs(),

            learning_rate=3e-4,
            gamma=0.995,
            ent_coef=0.01,
            n_steps=512,
            batch_size=256,
            n_epochs=10,
            vf_coef=0.5,
            clip_range=0.2,
            max_grad_norm=0.5,
            target_kl=0.015,       # MELHORIA: evita policy degradation
            tensorboard_log=LOG_DIR,
        )

        try:
            model.learn(total_timesteps=TOTAL, progress_bar=True,
                        callback=_callbacks(LOG_DIR))
        except KeyboardInterrupt:
            print("\n🛑 Interrompido.")
        finally:
            os.makedirs("data", exist_ok=True)
            model.save(MODEL_PATH)
            print(f"✅ Modelo → {MODEL_PATH}")
            plot_results(LOG_DIR, grid_size=GRID_SIZE,
                         num_obstacles=NUM_OBSTACLES)

    # ── FINETUNE (continua do melhor checkpoint) ───────────────────────────────
    elif mode == "finetune":
        TOTAL = 3_000_000

        # Encontra o checkpoint automaticamente se não existir o exato
        if not os.path.exists(CHECKPOINT):
            ckpts = sorted([
                f for f in os.listdir(LOG_DIR)
                if f.startswith("ckpt_") and f.endswith("_steps.zip")
            ], key=lambda x: int(x.split("_")[1]))
            if not ckpts:
                print(f"❌ Nenhum checkpoint em {LOG_DIR}")
                sys.exit(1)
            # Pega o checkpoint com mais steps (geralmente o melhor antes do crash)
            # Idealmente você troca manualmente para o de ~6.1M steps
            CHECKPOINT = os.path.join(LOG_DIR, ckpts[-1])
            print(f"⚠️  Checkpoint não encontrado exato. Usando: {CHECKPOINT}")

        print(f"--- Fine-tuning | {CHECKPOINT} ---")
        print(f"    + {TOTAL:,} steps adicionais")
        print(f"    + learning_rate=1e-4 (metade)")
        print(f"    + clip_range=0.15 (mais conservador)")
        print(f"    + target_kl=0.012 (mais estrito)")

        vec_env = _make_vec(GRID_SIZE, NUM_OBSTACLES, N_ENVS, LOG_DIR)

        model = PPO.load(CHECKPOINT, env=vec_env, device="cuda")

        # Fine-tuning: passos menores e mais conservadores
        model.learning_rate = 1e-4
        model.clip_range    = lambda _: 0.15
        model.target_kl     = 0.012
        model.ent_coef      = 0.005  # menos exploração — refina política existente

        try:
            model.learn(
                total_timesteps=TOTAL,
                progress_bar=True,
                callback=_callbacks(LOG_DIR, save_freq=50_000),
                reset_num_timesteps=False,  # continua contagem de steps
            )
        except KeyboardInterrupt:
            print("\n🛑 Interrompido.")
        finally:
            os.makedirs("data", exist_ok=True)
            model.save(MODEL_PATH)
            print(f"✅ Modelo salvo → {MODEL_PATH}")
            plot_results(LOG_DIR, grid_size=GRID_SIZE,
                         num_obstacles=NUM_OBSTACLES)

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

        total_cov  = 0.0
        total_steps = 0.0
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
                print(f"  Cobertura:  {len(w.visited)}/{w.total_coverable_cells} ({pct:.1f}%)")

        print(f"\n📊 Cobertura média:     {total_cov/N:.1f}%")
        print(f"📊 Passos médios:       {total_steps/N:.0f}")
        env.close()

    else:
        print("Use 'train', 'finetune' ou 'test'.")
        sys.exit(1)