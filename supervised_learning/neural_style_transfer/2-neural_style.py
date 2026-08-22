#!/usr/bin/env python3
"""Defines the NST class that performs tasks for neural style transfer."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                     'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor.

        style_image: the image used as a style reference, stored as a
            numpy.ndarray
        content_image: the image used as a content reference, stored as
            a numpy.ndarray
        alpha: the weight for content cost
        beta: the weight for style cost

        Raises TypeError if style_image is not a np.ndarray with the
        shape (h, w, 3)
        Raises TypeError if content_image is not a np.ndarray with the
        shape (h, w, 3)
        Raises TypeError if alpha is not a non-negative number
        Raises TypeError if beta is not a non-negative number
        """
        if not isinstance(style_image, np.ndarray) or \
                len(style_image.shape) != 3 or \
                style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or \
                len(content_image.shape) != 3 or \
                content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape "
                "(h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescale an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.

        image: a numpy.ndarray of shape (h, w, 3) containing the image
            to be scaled

        Returns: the scaled image as a tf.tensor with the shape
            (1, h_new, w_new, 3), where max(h_new, w_new) == 512 and
            min(h_new, w_new) is scaled proportionately. The image
            should be resized using bicubic interpolation. After
            resizing, the image's pixel values should be rescaled from
            the range [0, 255] to [0, 1].
        """
        if not isinstance(image, np.ndarray) or \
                len(image.shape) != 3 or \
                image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape

        if w > h:
            w_new = 512
            h_new = int(h * 512 / w)
        else:
            h_new = 512
            w_new = int(w * 512 / h)

        image = image[tf.newaxis, :]
        image = tf.image.resize(
            image, size=(h_new, w_new), method=tf.image.ResizeMethod.BICUBIC)

        image = image / 255
        image = tf.clip_by_value(image, 0, 1)

        return image

    def load_model(self):
        """
        Create the model used to calculate cost.

        The model uses the VGG19 Keras model as a base, with the model's
        input being the same as the VGG19 input. The model's output is
        a list containing the outputs of the VGG19 layers listed in
        style_layers followed by content_layer. Saves the model in the
        instance attribute model. MaxPooling2D layers are replaced with
        AveragePooling2D layers.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        vgg.save("vgg_base.h5")

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}

        vgg = tf.keras.models.load_model(
            "vgg_base.h5", custom_objects=custom_objects)

        outputs = []

        for layer_name in self.style_layers:
            outputs.append(vgg.get_layer(layer_name).output)

        outputs.append(vgg.get_layer(self.content_layer).output)

        model = tf.keras.models.Model(vgg.input, outputs)

        for layer in model.layers:
            layer.trainable = False

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate the gram matrix of a layer's output.

        input_layer: an instance of tf.Tensor or tf.Variable of shape
            (1, h, w, c) containing the layer output whose gram matrix
            should be calculated

        Returns: a tf.Tensor of shape (1, c, c) containing the gram
            matrix of input_layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape

        features = tf.reshape(input_layer, (h * w, c))
        n = tf.cast(h * w, tf.float32)

        gram = tf.matmul(features, features, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        gram = gram / n

        return gram
