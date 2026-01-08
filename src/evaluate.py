"""
Evaluate a trained policy on CartPole-v1 with optional rendering.
"""

import os
import time
import torch

from env import make_env
from policy import PolicyNetwork


MODEL_PATH = os.path.join("results", "policy.pt")


def main():
    # Create environment with rendering enabled
    env = make_env(render_mode="human")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Load trained policy
    policy = PolicyNetwork(obs_dim=obs_dim, n_actions=n_actions)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    policy.eval()

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, _ = policy.sample_action(obs_tensor)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Small delay so rendering is visible
        time.sleep(0.01)

    env.close()
    print(f"Evaluation complete. Total reward: {total_reward}")


if __name__ == "__main__":
    main()
