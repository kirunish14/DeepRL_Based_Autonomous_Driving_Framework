from env.carla_env import CarlaEnv
import numpy as np
import cv2
import time

# Create environment
env = CarlaEnv()

# Reset environment
obs, info = env.reset()

print("Environment Reset Successful")

try:

    while True:

        # Random actions for testing
        action = np.array([
            np.random.uniform(-0.3, 0.3),  # steering
            0.3                            # throttle
        ])

        obs, reward, done, truncated, info = env.step(action)

        print(f"Reward: {reward:.2f}")

        # Show camera
        cv2.imshow("RL Camera", obs)

        # Reset if episode done
        if done:

            print("Episode Ended")

            obs, info = env.reset()

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:

    env.close()

    cv2.destroyAllWindows()

    print("Environment Closed")