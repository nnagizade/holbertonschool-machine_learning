#!/usr/bin/env python3
"""Monte Carlo algorithm for estimating a state-value function."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Perform the Monte Carlo algorithm to estimate a value function.

    Args:
        env: the environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: function that takes a state and returns the next
            action to take.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.

    Returns:
        V, the updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = \
                env.step(action)
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break
        episode = np.array(episode, dtype=int)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state_t, reward_t = episode[t]
            G = reward_t + gamma * G
            if state_t not in episode[:t, 0]:
                V[state_t] = V[state_t] + alpha * (G - V[state_t])
    return V
