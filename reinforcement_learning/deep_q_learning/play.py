#!/usr/bin/env python3
"""Display a game of Atari's Breakout played by a trained DQN agent."""
import gymnasium as gym
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

from train import (
    WINDOW_LENGTH, AtariProcessor, GymnasiumCompatibilityWrapper,
    build_model,
)


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = GymnasiumCompatibilityWrapper(env)
    nb_actions = env.action_space.n

    model = build_model(WINDOW_LENGTH, nb_actions)
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    dqn = DQNAgent(
        model=model, nb_actions=nb_actions, memory=memory,
        policy=GreedyQPolicy(), processor=AtariProcessor(),
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')

    dqn.test(env, nb_episodes=5, visualize=True)
