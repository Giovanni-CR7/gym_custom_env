import gymnasium as gym
from gymnasium_env.grid_world import GridWorldEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)
env = gym.make("gymnasium_env/GridWorld-v0", size=5)
env = FlattenObservation(env)
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)

print('model trained')

(obs, _) = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action.item())
    print(f"Action: {action}, Reward: {reward}, Next State: {obs}")