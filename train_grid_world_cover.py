# python train_grid_world_cover.py train   # para treinar
# python train_grid_world_cover.py test    # para testar

import sys
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from gymnasium_env.grid_world_cover_render import GridWorldCoverRenderEnv


class CoverageWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.grid_size = env.unwrapped.size
        self.max_steps = self.grid_size * self.grid_size * 2 
        self.visited = set()
        self.obstacles = set()
        self.current_step = 0

        # Número total de células que podem ser cobertas (será atualizado no reset)
        self.total_coverable_cells = self.grid_size * self.grid_size

        # inclui posição do agente + grid visitado + grid de obstáculos
        self.observation_space = gym.spaces.Dict({
            "agent": env.observation_space["agent"],
            "visited": gym.spaces.Box(
                low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8
            ),
            "obstacles": gym.spaces.Box( 
                low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8
            )
        })

    def _get_obs(self, obs):
        visited_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for (x, y) in self.visited:
            visited_grid[x, y] = 1

        # NOVO: Cria um grid para os obstáculos
        obstacles_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for (x, y) in self.obstacles:
            obstacles_grid[x, y] = 1

        return {
            "agent": obs["agent"],
            "visited": visited_grid,
            "obstacles": obstacles_grid
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Pega a localização dos obstáculos do ambiente base
        self.obstacles = info["obstacles"]
        self.total_coverable_cells = self.grid_size * self.grid_size - len(self.obstacles)
        
        self.visited = {tuple(obs["agent"])}
        self.current_step = 0
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Definir agent_pos antes de usá-lo
        agent_pos = tuple(obs["agent"])
        
        # Sistema de recompensas
        if info.get("hit_obstacle", False):
            reward = -5  # Penalidade por colisão
        elif agent_pos not in self.visited:
            self.visited.add(agent_pos)
            reward = 10  # Recompensa por nova descoberta
        else:
            reward = -0.5  # Penalidade por explorar área conhecida
            
        # Bonus por progresso na cobertura
        coverage_bonus = (len(self.visited) / self.total_coverable_cells) * 5
        
        # Verificação do término do episódio
        if len(self.visited) == self.total_coverable_cells:
            terminated = True
            reward += 100  # Recompensa por completar o objetivo
        elif self.current_step >= self.max_steps:
            truncated = True
            reward -= 10  # Penalidade por atingir o limite de passos
        else:
            terminated = False
            
        return self._get_obs(obs), reward + coverage_bonus, terminated, truncated, info


def make_env(render_mode=None, size=5):
    # Passamos o número de obstáculos para o ambiente
    env = gym.make("gymnasium_env/GridWorld-Coverage-v0", size=size, render_mode=render_mode, num_obstacles=3)
    env = CoverageWrapper(env)
    return env


if __name__ == "__main__":
    train = True if len(sys.argv) > 1 and sys.argv[1] == "train" else False

    gym.register(
        id="gymnasium_env/GridWorld-Coverage-v0",
        entry_point=GridWorldCoverRenderEnv,
    )
    
    GRID_SIZE = 5

    if train:
        env = make_env(render_mode=None, size=GRID_SIZE)
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device="cpu",
            ent_coef=0.05,           # Aumentado para mais exploração
            gamma=0.95,              # Reduzido para foco em recompensas imediatas
            n_steps=128,             # Reduzido para atualizações mais frequentes
            batch_size=256,          # Ajustado para melhor aprendizado
            learning_rate=0.00025,   # Taxa de aprendizado ajustada
            clip_range=0.2,          # Para maior estabilidade
        )

        logger = configure("log/ppo_coverage_obstacles", ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)

        try:
            model.learn(total_timesteps=1_500_000)
        except KeyboardInterrupt:
            print("\n🛑 Treinamento interrompido pelo usuário.")
        finally:
            model.save("data/ppo_coverage_obstacles")
            print("✅ Modelo salvo em data/ppo_coverage_obstacles")

    print("📂 Carregando modelo salvo...")
    model = PPO.load("data/ppo_coverage_obstacles")

    # teste com render
    env = make_env(render_mode="human", size=GRID_SIZE)
    
    # Loop para rodar vários testes
    for i in range(5):
        print(f"\n--- Iniciando Teste {i+1} ---")
        obs, _ = env.reset()
        done, truncated = False, False
        steps = 0
        total_reward = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action.item())
            total_reward += reward
            # Adicionado para mostrar as células visitadas e os obstáculos no render
            if hasattr(env, 'visited'):
                env.unwrapped.set_visited_cells(env.visited)

            # print(f"Step {steps}: Action={action}, Reward={reward:.2f}, Visited={len(env.visited)}/{env.total_coverable_cells}")
            steps += 1
        
        print(f"Episódio finalizado em {steps} passos. Recompensa total: {total_reward:.2f}")
        print(f"Cobertura: {len(env.visited)} de {env.total_coverable_cells} células acessíveis.")


    env.close()