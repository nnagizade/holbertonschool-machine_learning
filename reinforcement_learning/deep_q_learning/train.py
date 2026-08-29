#!/usr/bin/env python3
"""Train a DQN agent to play Atari's Breakout using keras-rl2."""
import numpy as np
import gymnasium as gym
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.core import Processor

WINDOW_LENGTH = 4
INPUT_SHAPE = (84, 84)


class GymnasiumCompatibilityWrapper(gym.Wrapper):
    """Adapt a Gymnasium environment to the legacy Gym API that
    keras-rl2 expects (single-value reset, 4-tuple step)."""

    def reset(self, **kwargs):
        """Reset the environment and return only the observation."""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Take a step, collapsing terminated/truncated into done."""
        observation, reward, terminated, truncated, info = \
            self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def render(self):
        """Render the environment using its configured render mode."""
        return self.env.render()


class AtariProcessor(Processor):
    """Preprocess raw Atari frames for the DQN model."""

    def process_observation(self, observation):
        """Resize a raw RGB frame to an 84x84 grayscale array."""
        image = Image.fromarray(observation).resize(INPUT_SHAPE)
        return np.array(image.convert('L'), dtype=np.uint8)

    def process_state_batch(self, batch):
        """Scale a batch of stacked frames to floats in [0, 1]."""
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clip rewards to {-1, 0, 1} to stabilize training."""
        return np.clip(reward, -1.0, 1.0)


def build_model(window_length, nb_actions):
    """Build the CNN that approximates the Q-function.

    Args:
        window_length: number of stacked frames per state.
        nb_actions: size of the environment's action space.

    Returns:
        An uncompiled Keras Sequential model.
    """
    input_shape = (window_length,) + INPUT_SHAPE
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def build_agent(model, nb_actions):
    """Build and compile the DQN agent.

    Args:
        model: the Keras model approximating the Q-function.
        nb_actions: size of the environment's action space.

    Returns:
        A compiled keras-rl2 DQNAgent.
    """
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = EpsGreedyQPolicy(eps=0.1)
    agent = DQNAgent(
        model=model, nb_actions=nb_actions, memory=memory, policy=policy,
        processor=AtariProcessor(), nb_steps_warmup=50000, gamma=0.99,
        target_model_update=10000, train_interval=4, delta_clip=1.0,
    )
    agent.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    return agent


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5')
    env = GymnasiumCompatibilityWrapper(env)
    nb_actions = env.action_space.n

    model = build_model(WINDOW_LENGTH, nb_actions)
    dqn = build_agent(model, nb_actions)
    dqn.fit(env, nb_steps=1750000, log_interval=10000, verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)
