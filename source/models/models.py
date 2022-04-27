from typing import Tuple

import tensorflow as tf
from abc import ABCMeta, abstractmethod, abstractproperty


class AutoEncoder(tf.keras.Model, metaclass=ABCMeta):
    """_summary_
    """
    @abstractproperty
    def encoder(self) -> tf.keras.Model:
        pass

    @abstractproperty
    def decoder(self) -> tf.keras.Model:
        pass

    @abstractproperty
    def flat(self) -> tf.keras.Model:
        pass

    @abstractproperty
    def unflat(self) -> tf.keras.Model:
        pass

    @abstractmethod
    def get_reconstruction_loss_fn(self) -> tf.keras.losses.Loss:
        pass


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

    def call(self, h: tf.Tensor) -> tf.Tensor:
        return self._layers(h)


class ConvolutionalAutoEncoder(AutoEncoder):
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

        self._flat = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
        ])
        self._unflat = tf.keras.Sequential([
            tf.keras.layers.Reshape(self._shape_decoder_input[1:])
        ])
        self._decoder = DeconvolutionalDecoder(self._shape_data, self._config)

    @property
    def encoder(self) -> tf.keras.Model:
        return self._encoder

    @property
    def decoder(self) -> tf.keras.Model:
        return self._decoder

    @property
    def flat(self) -> tf.keras.Model:
        return self._flat

    @property
    def unflat(self) -> tf.keras.Model:
        return self._unflat

    def get_reconstruction_loss_fn(self) -> tf.keras.losses.Loss:
        return tf.keras.losses.MeanSquaredError()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = self._encoder(x)
        ##
        # h = self._l_flatten(z)
        # h = self._l_unflatten(h)
        ##
        y = self._decoder(h)
        return y


class ListenerModule(tf.keras.Model):
    """_summary_

    """

    def __init__(self, autoencoder: AutoEncoder, annotation_generator: tf.keras.Model, beta: float):
        self._autoencoder = autoencoder
        self._annotation_generator = annotation_generator
        self._beta = tf.clip_by_value(beta, 0.0, 1.0)

        self._fn_annotation_loss = tf.keras.losses.MeanSquaredError()
        self._fn_encoding_loss = self._autoencoder.get_reconstruction_loss_fn()

    def call(self, x: tf.Tensor, training=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """_summary_

        Args:
            x (tf.Tensor): Mel-spectrograms.
            ann (tf.Tensor): 1-D value tensors.
            training (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: _description_
        """
        h_l = self._autoencoder.encoder(x)
        z_l = self._autoencoder.flat(h_l)
        a_hat = self._annotation_generator(z_l)
        # add losses of
        if training:
            x_hat = self._autoencoder.decode(h_l)
            self.add_loss(
                (1 - self._beta) * self._fn_encoding_loss(x, x_hat)
            )
            # self.add_loss( self._beta * self._fn_annotation_loss(a, a_hat))
        return z_l, a_hat

    def get_loss_fn(self,  out: Tuple[tf.Tensor, tf.Tensor], truth: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        _, a_hat = out
        a = truth
        h_l = self._autoencoder.unflat(outs)
        loss_code = (1 - self._beta) * self._fn_encoding_loss(x, x_hat)
        loss_ann = self._beta * self._fn_annotation_loss()
