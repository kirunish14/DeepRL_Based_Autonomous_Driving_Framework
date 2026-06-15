from stable_baselines3 import PPO

from env.carla_env import CarlaEnv

import time

# =========================
# LOAD ENVIRONMENT
# =========================

env = CarlaEnv()

# =========================
# LOAD TRAINED MODEL
# =========================

model = PPO.load(
    "./models/ppo_carla_model"
)

print("Model Loaded")

# =========================
# RESET ENVIRONMENT
# =========================

obs, info = env.reset()

try:

    while True:

        # =========================
        # PREDICT ACTION
        # =========================

        action, _states = model.predict(
            obs,
            deterministic=False
        )

        # =========================
        # APPLY ACTION
        # =========================

        obs, reward, done, truncated, info = env.step(
            action
        )

        print(f"Reward: {reward:.2f}")

        # =========================
        # RESET EPISODE
        # =========================

        if done:

            print("Episode Finished")

            obs, info = env.reset()

        time.sleep(0.05)

except KeyboardInterrupt:

    print("Evaluation Stopped")

finally:

    env.close()