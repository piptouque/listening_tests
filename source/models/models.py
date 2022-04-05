from typing import Tuple

import tensorflow as tf


class NonLinearRegression(tf.keras.Model):
    """_summary_

    """
    # TODO: infer from somewhere..
    _SIZE_INPUT = 46 * 512

    def __init__(self, config: object):
        super(NonLinearRegression, self).__init__()

        self._config = config.model

        def _lin_layer(size_input: int) -> tf.keras.Model:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(
                    size_input,
                    activation='relu'),
                tf.keras.layers.Dropout(0.2)
            ])

        self._flat = tf.keras.layers.Flatten()
        self._lin_1 = _lin_layer(self._SIZE_INPUT)
        self._lin_2 = _lin_layer(self._SIZE_INPUT // 4)
        self._lin_3 = _lin_layer(1024)
        # regression output
        self._lin_4 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, z: tf.Tensor) -> tf.Tensor:
        #
        z_flat = self._flat(z)
        y = self._lin_1(z_flat)
        y = self._lin_2(y)
        y = self._lin_3(y)
        y = self._lin_4(y)
        return y


class ConvolutionalEncoder(tf.keras.Model):
    """_summary_
    """

    def __init__(self, config: object):
        super(ConvolutionalEncoder, self).__init__()

        self._config = config.model

        def _conv_layer(nb_filters: int, rate_dilation: Tuple[int, int]) -> tf.keras.Model:
            return tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    nb_filters,
                    kernel_size=self._config.cnn.size_kernel,
                    dilation_rate=rate_dilation,
                    padding='same',
                    use_bias=True,
                    activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ])

        self._conv_1 = _conv_layer(64, 3)
        self._pool_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self._conv_2 = _conv_layer(128, 3)
        self._pool_2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self._conv_3 = _conv_layer(256, 3)
        self._pool_3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self._conv_4 = _conv_layer(512, 3)
        self._avg = tf.keras.layers.Lambda(
            lambda x: tf.math.reduce_mean(x, axis=-2, keepdims=True))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        #
        z = self._conv_1(x)
        z = self._pool_1(z)
        z = self._conv_2(z)
        z = self._pool_2(z)
        z = self._conv_3(z)
        z = self._pool_3(z)
        z = self._conv_4(z)
        z = self._avg(z)
        return z
