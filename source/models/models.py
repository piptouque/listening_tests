from typing import Tuple, Union

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

class MaskConstraint(tf.keras.constraints.Constraint):
    def __init__(self, mask: tf.Tensor, axis: Union[int, Tuple[int]]):
        tf.Assert(tf.math.equal(tf.rank(mask), tf.rank(axis)))
        self._mask = tf.constant(tf.clip_by_value(tf.math.round(mask), 0, 1))
        self._axis = axis

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        s = tf.ones(tf.rank(w))
        s[axis] = tf.shape(w)
        mask = tf.reshape(self._mask,  s)
        tf.Assert(tf.math.equal(tf.shape(w)[axis], tf.shape(self._mask)[axis]))
        return w * self._mask



class VectorQuantiser(tf.keras.Layer):
    """
    Vector Quantification part of a VQ-VAE,
    as described in the original paper:
        [1] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, 'Neural discrete representation learning’,
            in Advances in neural information processing systems, 2017, vol. 30.
    And implemented based on:
        https://github.com/hiwonjoon/tf-vqvae/blob/master/model.py
    """
    def __init__(dim_embedding:int, nb_embeddings: int, beta: float):
        self._vecs_embedding = tf.random.uniform(shape=(dim_embedding, nb_embeddings))

    def call(self, z_e: tf.Tensor) -> tf.Tensor:
        tf.Assert(tf.math.equal(tf.shape(z_e)[-1], tf.shape(self._vecs_embedding)[-1]))
        h = tf.expand_dims(z_e, axis=-2)
        h = tf.norm(h - self._vecs_embedding, axis=-1)
        ids = tf.argmin(h, axis=-1)
        z_q = tf.gather(self._vecs_embedding, ids)
        # compute and add losses
        l_commit = tf.reduce_mean(tf.norm(tf.stop_gradient(z_e) - z_q, axis=-1)**2, axis=[0,1,2])
        l_codebook = tf.reduce_mean(tf.norm(z_e - tf.stop_gradient(z_q), axis=-1)**2, axis=[0,1,2])
        self.add_loss(l_codebook + l_commit * beta)
        return {
            'quantised': z_q,
            'vecs': self._vecs_embedding,
            'ids': ids
        }

class MuLawQuantiser(tf.keras.Layer):
    """
    as described in:
        [2] A. van den Oord et al., 'WaveNet: A Generative Model for Raw Audio',
            arXiv:1609.03499 [cs.SD], 2016.
    """
    def __init__(self, mu: int):
        self._mu = mu




class GatedActivationUnit(tf.keras.Model):
    def __init__(
        nb_filters_dilation: int,
        size_kernel: int,
        rate_dilation: int,
        causal=False: bool,
        use_bias=True: bool
        ):
        self._conv_filter = tf.keras.Conv1D(
                nb_filters_dilation,
                kernel_size=size_kernel,
                dilation_rate=rate_dilation,
                padding='causal' if causal else 'same',
                use_bias=use_bias,
                activation='tanh'
            )
        self._conv_gate = tf.keras.Conv1D(
                nb_filters_dilation,
                kernel_size=size_kernel,
                dilation_rate=rate_dilation,
                padding='causal' if causal else 'same',
                use_bias=use_bias,
                activation='sigmoid'
            )
    def call(x: tf.Tensor) -> tf.Tensor:
        x_filter = self._conv_filter(x)
        x_gate = self._conv_gate(x)
        return x_filter * x_gate
    

class WaveNet(tf.keras.Model):
    """
    Modified version of WaveNet
    Based on: https://github.com/WindQAQ/tensorflow-wavenet/
    """
    class ResidualBlock(tf.keras.Model):
        def __init__(
            nb_filters_residual: int,
            nb_filters_dilation: int,
            nb_filters_skip: int,
            size_kernel: int,
            rate_dilation: int,
            causal: bool,
            use_bias: bool
            ):
            self._gau = GatedActivationUnit(
                   nb_filters_dilation,
                   size_kernel,
                   rate_dilation,
                   causal=causal,
                   use_bias=use_bias
            )
            self._conv_skip = tf.keras.Conv1D(
                nb_filters_skip,
                kernel_size=1,
                padding='same',
                use_bias=use_bias
            )
            self._conv_residual = tf.keras.Conv1D(
                nb_filters_skip,
                kernel_size=1,
                padding='same',
                use_bias=use_bias
            )
        def call(x: tf.Tensor) -> tf.Tensor:
            y_gau = self._gau(x)
            y_skip = self._conv_skip(y_gau)
            y_residual = self._conv_residual(y_gau)
            return y_residual + x, y_skip
        
    def __init__(self,
            size_output: int,
            nb_blocks_residual: int,
            nb_filters_residual: int,
            nb_filters_dilation: int,
            nb_filters_skip: int,
            size_kernel: int,
            rate_dilation_init: int,
            causal=False: bool,
            use_bias=False: bool
            ):
        self._conv_init = tf.keras.Conv1D(
            nb_filters_residual,
            kernel_size=size_kernel, # TODO?: different kernel size at first?
            padding='same'
        )
        self._blocks_res = [
            ResidualBlock(
                nb_filters_residual,
                nb_filters_dilation,
                nb_filters_skip,
                size_kernel,
                rate_dilation_init ** idx_block,
                causal=causal,
                use_bias=use_bias
            )
            for idx_block in range(nb_blocks_residual)]
        self._block_end = tf.keras.Sequential(
            tf.keras.Add(),
            tf.keras.activations.ReLU(),
            tf.keras.Conv1D(
                nb_filters_skip,
                kernel_size=1,
                padding='same',
                use_bias=True
            ),
            tf.keras.activations.ReLU(),
            tf.keras.Conv1D(
                size_output,
                kernel_size=1,
                padding='same',
                use_bias=True
            ),
            tf.keras.activations.ReLU(),
            tf.keras
        )


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
