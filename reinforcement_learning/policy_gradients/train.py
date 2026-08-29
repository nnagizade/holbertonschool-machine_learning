#!/usr/bin/env python3
"""Training module for policy gradient."""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98,
          show_result=False):
    """
    Implement a full training.

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor
        show_result: if True, render the environment every 1000
            episodes computed

    Returns:
        all values of the score (sum of all rewards during one
        episode loop)
    """
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        state = state[None, :]
        grads = []
        rewards = []
        score = 0

        done = False
        while not done:
            if show_result and episode % 1000 == 0:
                env.render()

            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )
            done = terminated or truncated

            grads.append(grad)
            rewards.append(reward)
            score += reward

            state = next_state[None, :]

        for i in range(len(grads)):
            future_rewards = sum(
                r * (gamma ** t) for t, r in enumerate(rewards[i:])
            )
            weight += alpha * grads[i] * future_rewards

        scores.append(score)
        msg = "Episode: {} Score: {}".format(episode, score)
        print(msg)

    return scores
