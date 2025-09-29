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
        self.max_steps = self.grid_size * self.grid_size * 2 # Aumentar um pouco pode ser útil
        self.visited = set()
        self.obstacles = set()
        self.current_step = 0

        # Número total de células que podem ser cobertas (será atualizado no reset)
        self.total_coverable_cells = self.grid_size * self.grid_size

        # Redefine observation_space: agora inclui posição do agente + grid visitado + grid de obstáculos
        self.observation_space = gym.spaces.Dict({
            "agent": env.observation_space["agent"],
            "visited": gym.spaces.Box(
                low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8
            ),
            "obstacles": gym.spaces.Box( # NOVO: O agente precisa "ver" os obstáculos
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
        
        # NOVO: Pega a localização dos obstáculos do ambiente base
        self.obstacles = info["obstacles"]
        self.total_coverable_cells = self.grid_size * self.grid_size - len(self.obstacles)
        
        self.visited = {tuple(obs["agent"])}
        self.current_step = 0
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        agent_pos = tuple(obs["agent"])

        # NOVO: Sistema de recompensa modificado
        if info.get("hit_obstacle", False):
            reward = -2  # Penalidade forte por colidir com um obstáculo
        elif agent_pos not in self.visited:
            self.visited.add(agent_pos)
            reward = 1   # Recompensa por visitar uma nova célula
        else:
            reward = -0.1 # Pequena penalidade para não ficar parado

        # NOVO: Fim quando cobre todo o grid ACESSÍVEL
        if len(self.visited) == self.total_coverable_cells:
            terminated = True
            reward += 20 # Recompensa maior por completar o objetivo
        elif self.current_step >= self.max_steps:
            truncated = True
        else:
            terminated = False

        return self._get_obs(obs), reward, terminated, truncated, info


def make_env(render_mode=None, size=10):
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
    
    GRID_SIZE = 10

    if train:
        env = make_env(render_mode=None, size=GRID_SIZE)
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device="cpu",
            ent_coef=0.01,
            gamma=0.99 # Um gamma um pouco maior pode ajudar em tarefas de longo prazo
        )

        logger = configure("log/ppo_coverage_obstacles", ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)

        try:
            # Pode ser necessário mais passos de treino, pois o problema é mais complexo
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