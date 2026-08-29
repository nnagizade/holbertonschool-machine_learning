#!/usr/bin/env python3
"""Performs Q-learning"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs Q-learning

    Args:
        env: the FrozenLakeEnv instance
        Q: a numpy.ndarray containing the Q-table
        episodes: the total number of episodes to train over
        max_steps: the maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate
        epsilon: the initial threshold for epsilon greedy
        min_epsilon: the minimum value that epsilon should decay to
        epsilon_decay: the decay rate for updating epsilon between
            episodes

    Returns:
        Q, total_rewards
            Q: the updated Q-table
            total_rewards: a list containing the rewards per episode
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        rewards_current_episode = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(
                action)

            if terminated and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action])

            state = new_state
            rewards_current_episode += reward

            if terminated or truncated:
                break

        epsilon = min_epsilon + (
            initial_epsilon - min_epsilon) * np.exp(
                -epsilon_decay * episode)

        total_rewards.append(rewards_current_episode)

    return Q, total_rewards
