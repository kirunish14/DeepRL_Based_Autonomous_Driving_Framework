from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage

from env.carla_env import CarlaEnv

env = CarlaEnv()

check_env(env)
print("Environment Check Passed")


env = Monitor(env)

env = DummyVecEnv([lambda: env])


# Check Gym compatibility


# =========================
# CREATE PPO MODEL
# =========================


model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,

    learning_rate=1e-4,

    n_steps=1024,
    batch_size=64,
    n_epochs=10,

    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.02,

    target_kl=0.08,


    tensorboard_log="./outputs/logs/"
)

print("PPO Model Created")

# =========================
# TRAIN MODEL
# =========================
print("Starting training for 100000 timesteps")

print(model.n_steps)

model.learn(
    total_timesteps=50000
)

print("Training finished")

# =========================
# SAVE MODEL
# =========================

model.save(
    "./models/ppo_carla_model"
)

print("Model Saved")

# Cleanup
env.close()