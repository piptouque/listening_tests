from typing import Tuple

import tensorflow as tf


class NonLinearRegression(tf.keras.Model):
    """_summary_

    """

    def __init__(self, size_input: int, config: object):
        super(NonLinearRegression, self).__init__()

        self._config = config
        self._size_input = size_input

        def _lin_layer(size_input: int) -> tf.keras.Model:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(
                    size_input,
                    activation='relu'),
                tf.keras.layers.Dropout(0.2)
            ])

        self._flat = tf.keras.layers.Flatten()
        self._lin_1 = _lin_layer(self._size_input)
        self._lin_2 = _lin_layer(self._size_input // 4)
        self._lin_3 = _lin_layer(1024)
        # regression output
        self._lin_4 = tf.keras.layers.Dense(1, activation='sigmoid')
        self._reshape = tf.keras.layers.Reshape(1, 1)

    def call(self, z: tf.Tensor) -> tf.Tensor:
        #
        z_flat = self._flat(z)
        y = self._lin_1(z_flat)
        y = self._lin_2(y)
        y = self._lin_3(y)
        y = self._lin_4(y)
        y = self._reshape(y)
        return y


class ConvolutionalEncoder(tf.keras.Model):
    """_summary_

    """
    # TODO: infer from somewhere..

    def __init__(self, config: object):
        super(ConvolutionalEncoder, self).__init__()
        self._config = config

        def _conv_layer(nb_filters: int, rate_dilation: Tuple[int, int]) -> tf.keras.Model:
            return tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(
                    nb_filters,
                    kernel_size=self._config.conv.size_kernel,
                    dilation_rate=rate_dilation,
                    padding='same',
                    use_bias=True,
                    activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ])

        def _conv_pooling_layer(nb_filters: int, rate_dilation: Tuple[int, int]) -> tf.keras.Model:
            return tf.keras.Sequential([
                _conv_layer(nb_filters, rate_dilation),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2), padding='same'),
            ])

        self._layers = tf.keras.Sequential([
            _conv_pooling_layer(64, self._config.conv.rate_dilation),
            _conv_pooling_layer(128, self._config.conv.rate_dilation),
            _conv_pooling_layer(256, self._config.conv.rate_dilation),
            _conv_layer(512, self._config.conv.rate_dilation),
            # tf.keras.layers.Lambda( lambda x: tf.math.reduce_mean(x, axis=-2, keepdims=True))
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._layers(x)


class DeconvolutionalDecoder(tf.keras.Model):
    """_summary_
    """

    def __init__(self, shape_output: Tuple[int, int, int], config: object):
        super(DeconvolutionalDecoder, self).__init__()

        self._config = config
        self._shape_output = shape_output

        def _deconv_layer(nb_filters: int, rate_dilation: Tuple[int, int]) -> tf.keras.Model:
            return tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(
                    nb_filters,
                    kernel_size=self._config.conv.size_kernel,
                    dilation_rate=rate_dilation,
                    padding='same',
                    use_bias=True,
                    activation=None
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ])

        def _deconv_upsample_layer(nb_filters: int, rate_dilation: Tuple[int, int]) -> tf.keras.Model:
            return tf.keras.Sequential([
                _deconv_layer(nb_filters, rate_dilation),
                tf.keras.layers.UpSampling2D(size=(2, 2))
            ])

        self._layers = tf.keras.Sequential([
            _deconv_upsample_layer(256, self._config.conv.rate_dilation),
            _deconv_upsample_layer(126, self._config.conv.rate_dilation),
            _deconv_upsample_layer(64, self._config.conv.rate_dilation),
            _deconv_layer(self._shape_output[-1],
                          self._config.conv.rate_dilation),
            tf.keras.layers.Resizing(
                self._shape_output[-3], self._shape_output[-2])
        ])

    def call(self, z: tf.Tensor) -> tf.Tensor:
        return self._layers(z)


class ConvolutionalAutoEncoder(tf.keras.Model):
    """_summary_
    """

    def __init__(self, config: object, shape_data: Tuple[int, int, int]):
        super(ConvolutionalAutoEncoder, self).__init__()

        self._config = config
        self._shape_data = shape_data

        self._encoder = ConvolutionalEncoder(self._config)
        # get output shape of the last layer of the encoder

        # shape of the playing directions (same as output of encoder)
        self._shape_decoder_input = self._encoder.compute_output_shape(
            self._shape_data)

        #
        def _lin_layer(size_input: int) -> tf.keras.Model:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(
                    size_input,
                    activation='relu'),
                tf.keras.layers.Dropout(0.2)
            ])

        self._bottleneck_in = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                self._config.ae.size_latent, activation='sigmoid'),
            tf.keras.layers.Dropout(0.1)
        ])
        self._bottleneck_out = tf.keras.Sequential([
            tf.keras.layers.Dense(tf.math.reduce_prod(
                self._shape_decoder_input[1:]), activation='sigmoid'),
            tf.keras.layers.Reshape(self._shape_decoder_input[1:])
        ])
        self._decoder = DeconvolutionalDecoder(self._shape_data, self._config)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z = self._encoder(x)
        ##
        h = self._bottleneck_in(z)
        h = self._bottleneck_out(h)
        ##
        y = self._decoder(h)
        return y
