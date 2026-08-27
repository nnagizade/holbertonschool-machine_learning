#!/usr/bin/env python3
"""
Module defining the Simple_GAN class for custom Keras GAN training.
"""
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """
    Custom Keras Model class implementing a Simple Generative Adversarial
    Network (GAN) architecture.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Initializes the Simple_GAN model with loss functions and optimizers.

        Args:
            generator: The generator Keras model.
            discriminator: The discriminator Keras model.
            latent_generator: Function to sample latent vectors.
            real_examples: Tensor containing real target sample dataset.
            batch_size: Integer batch size for training steps.
            disc_iter: Integer count of discriminator steps per generator step.
            learning_rate: Float learning rate for Adam optimizers.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # Define generator loss and optimizer
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape)
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer,
            loss=generator.loss
        )

        # Define discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer,
            loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generates fake samples from the generator network.

        Args:
            size: Number of samples to draw (defaults to batch_size).
            training: Boolean indicator for Keras model layers.

        Returns:
            Tensor of generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Samples random real examples from the dataset.

        Args:
            size: Number of samples to select (defaults to batch_size).

        Returns:
            Tensor of selected real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Overridden train_step executing discriminator updates followed by
        a generator update.

        Args:
            useless_argument: Unused input required by Keras fit API.

        Returns:
            Dictionary containing 'discr_loss' and 'gen_loss'.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                pred_real = self.discriminator(real_samples, training=True)
                pred_fake = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(pred_real, pred_fake)
            grads_disc = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads_disc, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            pred_fake = self.discriminator(fake_samples, training=True)
            gen_loss = self.generator.loss(pred_fake)
        grads_gen = tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(grads_gen, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
