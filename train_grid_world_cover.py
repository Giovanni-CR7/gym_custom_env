#
# python train_grid_world_cover.py train   # para treinar
# python train_grid_world_cover.py test    # para testar
#

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
        self.current_step = 0

        # lista de todas as células possíveis
        self.all_cells = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}

        # redefine observation_space: agora inclui posição do agente + grid visitado
        self.observation_space = gym.spaces.Dict({
            "agent": env.observation_space["agent"],  # Box(2,)
            "visited": gym.spaces.Box(
                low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8
            )
        })

    def _get_obs(self, obs):
        visited_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for (x, y) in self.visited:
            visited_grid[x, y] = 1
        return {
            "agent": obs["agent"],
            "visited": visited_grid
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.visited = {tuple(obs["agent"])}
        self.current_step = 0
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        agent_pos = tuple(obs["agent"])

        # recompensa simples
        if agent_pos not in self.visited:
            self.visited.add(agent_pos)
            reward = 1
        else:
            reward = 0

        # fim quando cobre todo o grid
        if len(self.visited) == self.grid_size * self.grid_size:
            terminated = True
            reward += 10
        elif self.current_step >= self.max_steps:
            truncated = True
        else:
            terminated = False

        return self._get_obs(obs), reward, terminated, truncated, info




def make_env(render_mode=None, size=3):
    env = gym.make("gymnasium_env/GridWorld-Coverage-v0", size=size, render_mode=render_mode)
    env = CoverageWrapper(env)
    return env



if __name__ == "__main__":
    train = True if len(sys.argv) > 1 and sys.argv[1] == "train" else False

    gym.register(
        id="gymnasium_env/GridWorld-Coverage-v0",
        entry_point=GridWorldCoverRenderEnv,
    )

    if train:
        env = make_env(render_mode=None, size=10)  # treino sem render
        model = PPO(
            "MultiInputPolicy",  #entender essa estrutura de rede
            env,
            verbose=1,
            device="cpu",
            ent_coef=0.01, #entender hiperparâmetro 
            gamma=0.9 #entender hiperparâmetro 
        )
        #deixar o plot visual

        logger = configure("log/ppo_coverage", ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)

        try:
            model.learn(total_timesteps=1_000_000)
        except KeyboardInterrupt:
            print("\n🛑 Treinamento interrompido pelo usuário.")
        finally:
            model.save("data/ppo_coverage")
            print("✅ Modelo salvo em data/ppo_coverage")

    print("📂 Carregando modelo salvo...")
    model = PPO.load("data/ppo_coverage")

    # teste com render
    env = make_env(render_mode="human", size=10)
    obs, _ = env.reset()
    done, truncated = False, False
    steps = 0

    while not (done or truncated) and steps < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action.item())
        print(f"Step {steps}: Action={action}, Reward={reward:.2f}, Visited={len(env.visited)}")
        steps += 1


    env.close()

#deixar o plot visual