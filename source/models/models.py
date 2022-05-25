from typing import Tuple, Union, List

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
    def __init__(self, mask: tf.Tensor, axis: Union[int, Tuple[int]]) -> None:
        tf.debugging.assert_equal(tf.rank(mask), tf.rank(axis))
        self._mask = tf.constant(tf.clip_by_value(tf.math.round(mask), 0, 1))
        self._axis = axis

    def call(self, w: tf.Tensor) -> tf.Tensor:
        s = tf.ones(tf.rank(w))
        s[self._axis] = tf.shape(w)
        mask = tf.reshape(self._mask,  s)
        tf.debugging.assert_equal(tf.shape(w)[self._axis], tf.shape(mask)[self._axis])
        return w * self._mask

class GatedActivationUnit(tf.keras.layers.Layer):
    def __init__(self,
        nb_filters_dilation: int,
        size_kernel: int,
        rate_dilation: int,
        causal: bool=False,
        use_bias: bool=True
        ) -> None:
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
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_filter = self._conv_filter(x)
        x_gate = self._conv_gate(x)
        return x_filter * x_gate
    


class VectorQuantiser(tf.keras.layers.Layer):
    """
    Vector Quantification part of a VQ-VAE,
    as described in the original papero:
        [1] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, 'Neural discrete representation learning’,
            in Advances in neural information processing systems, 2017, vol. 30.
    And implemented based on:
        https://github.com/hiwonjoon/tf-vqvae/
        https://github.com/JeremyCCHsu/vqvae-speech/
    """

    def __init__(self, shape_input: Tuple[int], nb_embeddings: int, axes: Tuple[int], beta: float) -> None:
        """
            axis: 0 -> first non-batch dim

            ---
            _vecs_embedding: dim -> (nb_embeddings, not quantised dim)
        """
        super(VectorQuantiser, self).__init__()
        self._axes = tuple(axes) 
        s = (nb_embeddings,)
        s = s + tuple([shape_input[axis] for axis in self._axes])
        self._vecs_embedding = tf.random.uniform(shape=s, dtype=tf.dtypes.float32)
        self._axes_embedding = tuple(range(1,len(self._axes)+1))
        self._beta = beta

    def call(self, z_e: tf.Tensor) -> tf.Tensor:
        """
            z_e: dim -> (batch, not quantised dim (time), quantised dim (channel)) 
        """
        # tf.debugging.assert_equal(tf.shape(z_e)[1:], self._shape_input[1:])
        # dim -> (batch, not quantised dim, nb_embeddings)
        dot_z_embedding = tf.tensordot(z_e, self._vecs_embedding, axes=[self._axes, self._axes_embedding])
        norm_sq_z = tf.reduce_sum(tf.square(z_e), axis=self._axes)
        # dim -> (batch, not quantised dim, 1)
        norm_sq_embedding = tf.reduce_sum(tf.square(self._vecs_embedding), axis=self._axes_embedding)
        # add one dim to make it broadcast implicitly
        # dim -> (batch, not quantised dim, 1)
        norm_sq_z = tf.expand_dims(norm_sq_z, axis=-1)
        # Euclidean distance as: dist(x,y) = norm(x)^2 + norm(y)^2  - 2<x,y>
        # Uses implicit broadcasts repeatedly, beware the order of operations.
        dist = (- 2 * dot_z_embedding + norm_sq_z) + norm_sq_embedding
        ids_embedding= tf.argmin(dist, axis=-1)
        z_q = tf.nn.embedding_lookup(self._vecs_embedding, ids_embedding)
        # fix: tf.norm does not interpret 1-uples as integers.
        axes = self._axes[0] if len(self._axes) == 1 else self._axes
        loss_commit = tf.reduce_mean(tf.norm(tf.stop_gradient(z_e) - z_q, axis=axes))
        loss_codebook = tf.reduce_mean(tf.norm(z_e - tf.stop_gradient(z_q), axis=axes))
        self.add_loss(loss_codebook + loss_commit * self._beta)
        # similarity as normalised dot product of latent and embedding vectors
        similarity = (dot_z_embedding / tf.sqrt(norm_sq_z)) / tf.sqrt(norm_sq_embedding)
        return {
            'quantised': z_q,
            'vecs': self._vecs_embedding,
            'similarity': similarity,
            'ids': ids_embedding
       }

class WaveNet(tf.keras.Model):
    """
    Modified version of WaveNet
    Based on: https://github.com/WindQAQ/tensorflow-wavenet/
    """
    class ResidualBlock(tf.keras.Model):
        def __init__(self,
            nb_filters_dilation: int,
            nb_filters_skip: int,
            size_kernel: int,
            rate_dilation: int,
            causal: bool=False,
            use_bias: bool=False
            ) -> None:
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
        def call(self, x: tf.Tensor) -> tf.Tensor:
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
            causal: bool=False,
            use_bias: bool=False
            ) -> None:
        super(WaveNet, self).__init__()
        self._conv_init = tf.keras.Conv1D(
            nb_filters_residual,
            kernel_size=size_kernel, # TODO?: different kernel size at first?
            padding='same'
        )
        self._blocks_res = [
            self.ResidualBlock(
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
            # TODO: softmax??
        )

class JukeboxModel(tf.keras.Model):
    class ResidualSubBlock(tf.keras.Model):
        def __init__(self,
                nb_filters: int,
                size_kernel: int,
                rate_dilation: int,
            ) -> None:
            super(JukeboxModel.ResidualSubBlock, self).__init__()
            self._convs = tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                   nb_filters,
                   kernel_size=size_kernel,
                   dilation_rate=rate_dilation,
                   padding='same'
                ),
                tf.keras.layers.Conv1D(
                    nb_filters,
                    kernel_size=size_kernel,
                    dilation_rate=1,
                    padding='same'
                )
            ])
        def call(self, x: tf.Tensor) -> tf.Tensor:
            h = self._convs(x) + x
            return h


    class DownsamplingBlock(tf.keras.Model):
        def __init__(self,
            nb_filters: int,
            size_kernel_down: int,
            stride_down: int,
            nb_blocks_res: int, 
            size_kernel_res: int,
            rate_dilation_res: int,
        ) -> None:
            super(JukeboxModel.DownsamplingBlock, self).__init__()
            self._conv_down = tf.keras.layers.Conv1D(
                nb_filters,
                kernel_size=size_kernel_down,
                strides=stride_down,
                padding='same'
            )
            self._blocks_res = [
                JukeboxModel.ResidualSubBlock(
                    nb_filters,
                    size_kernel=size_kernel_res,
                    rate_dilation=rate_dilation_res
                )
            for _ in range(nb_blocks_res)]
        def call(self, x: tf.Tensor) -> tf.Tensor:
            x_down = self._conv_down(x)
            for block in self._blocks_res:
                x_down = block(x_down)
            z = x_down
            return z


    class UpsamplingBlock(tf.keras.Model):
        def __init__(self,
            nb_filters: int,
            size_kernel_up: int,
            stride_up: int,
            nb_blocks_res: int,
            size_kernel_res: int,
            rate_dilation_res: int,
        ) -> None:
            super(JukeboxModel.UpsamplingBlock, self).__init__()
            self._blocks_res = [
                JukeboxModel.ResidualSubBlock(
                    nb_filters,
                    size_kernel=size_kernel_res,
                    rate_dilation=rate_dilation_res
                )
            for _ in range(nb_blocks_res)]
            self._conv_up = tf.keras.layers.Conv1DTranspose(
                nb_filters,
                kernel_size=size_kernel_up,
                strides=stride_up,
                padding='same'
            )
        def call(self, z: tf.Tensor) -> tf.Tensor:
            for block in self._blocks_res:
                z = block(z)
            x = self._conv_up(z)
            return x


    class Encoder(tf.keras.Model):
        def __init__(self,
            nb_filters: int,
            nb_blocks_down: int,
            size_kernel_down: int,
            stride_down: int,
            nb_blocks_res: int,
            size_kernel_res: int,
            rate_dilation_res: int
        ) -> None:
            super(JukeboxModel.Encoder, self).__init__()
            self._blocks_down = [
               JukeboxModel.DownsamplingBlock(
                   nb_filters,
                   size_kernel_down,
                   stride_down,
                   nb_blocks_res,
                   size_kernel_res,
                   rate_dilation_res
               ) 
            for _ in range(nb_blocks_down)]
        def call(self, x: tf.Tensor) -> tf.Tensor:
            for block in self._blocks_down:
               x = block(x)
            return x


    class Decoder(tf.keras.Model):
        def __init__(self,
            nb_channels_output: int,
            nb_filters: int,
            nb_blocks_up: int,
            size_kernel_up: int,
            stride_up: int,
            nb_blocks_res: int,
            size_kernel_res: int,
            rate_dilation_res: int
        ) -> None:
            super(JukeboxModel.Decoder, self).__init__()
            self._blocks_up = [
               JukeboxModel.UpsamplingBlock(
                   nb_filters,
                   size_kernel_up,
                   stride_up,
                   nb_blocks_res,
                   size_kernel_res,
                   rate_dilation_res
               ) 
            for _ in range(nb_blocks_up)]
            self._conv_proj = tf.keras.layers.Conv1D(
                nb_channels_output,
                kernel_size=size_kernel_res,
                strides=1,
                padding='same'
            )
        def call(self, z: tf.Tensor) -> tf.Tensor:
            h = z
            for block in self._blocks_up:
                h = block(h)
            x_hat = self._conv_proj(h)
            return x_hat


    def __init__(self,
        shape_input: tf.Tensor,
        nb_levels: int = 3,
        nb_filters: int=32,
        nb_blocks_sample: List[int]=(3, 5, 7),
        size_kernel_sample: int=4,
        stride_sample: int=2,
        nb_blocks_res: List[int]=(8, 4, 4),
        size_kernel_res: int= 3,
        rate_dilation_res: int= 3,
        size_codebook: int=256,
        beta_codebook: float=0.99
    ) -> None:
        tf.debugging.assert_equal(tf.rank(nb_blocks_sample), tf.rank(nb_blocks_sample))
        super(JukeboxModel, self).__init__()
        self._encoders = [
            self.Encoder(
                nb_filters,
                nb_blocks_down=nb_blocks_sample[idx_level],
                size_kernel_down=size_kernel_sample,
                stride_down=stride_sample,
                nb_blocks_res=nb_blocks_res[idx_level],
                size_kernel_res=size_kernel_res,
                rate_dilation_res=rate_dilation_res 
            )
            for idx_level in range(nb_levels)
        ]
        shape_code = self._encoders[-1].compute_output_shape(shape_input)
        shapes_code = [encoder.compute_output_shape(shape_input) for encoder in self._encoders]
        self._decoders = [
            self.Decoder(
                nb_channels_output=shape_code[-1],
                nb_filters=nb_filters,
                nb_blocks_up=nb_blocks_sample[idx_level],
                size_kernel_up=size_kernel_sample,
                stride_up=stride_sample,
                nb_blocks_res=nb_blocks_res[idx_level],
                size_kernel_res=size_kernel_res,
                rate_dilation_res=rate_dilation_res 
            )
            for idx_level in range(nb_levels)
        ]
        self._codebook = VectorQuantiser(
            shape_input=shape_code,
            nb_embeddings=size_codebook,
            axes=[2],
            beta=beta_codebook
        )

    def _encode(self, x: tf.Tensor) -> List[tf.Tensor]:
        return [encoder(x) for encoder in self._encoders]
    def _quantise(self, zs: List[tf.Tensor]) -> List[tf.Tensor]:
        return [self._codebook(z)['quantised'] for z in zs]
    def _decode(self, zs: List[tf.Tensor]) -> List[tf.Tensor]:
        return [decoder(zs[idx_level]) for idx_level, decoder in enumerate(self._decoders)]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        xs_hat = []
        # for idx_level in range(len(self._encoders)):
        idx_level = 0
        z = self._encoders[idx_level](x)
        z_q = self._codebook(z)['quantised']
        x_hat = self._decoders[idx_level](z_q)
        xs_hat.append(x_hat) 
        # for each level, loss over diff of power spectra of x_hat and x
        return xs_hat 

class GstModel(tf.keras.Model):
    class ReferenceEncoder(tf.keras.Model):
        """
        """
        def __init__(self,
            size_input: tf.Tensor,
            size_kernel: int,
            rate_dilation: int,
            nb_heads_att: int,
        ):
            super(GstModel.ReferenceEncoder, self).__init__()
            stride_conv = (2, 2)
            def _block_conv(nb_filters: int):
                return tf.keras.Sequential(
                    tf.keras.Conv2D(
                        nb_filters,
                        kernel_size=size_kernel,
                        strides=stride_conv,
                        padding='same'
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU()
                )
            self._stack_conv = tf.keras.Sequential(
                _block_conv(32),
                _block_conv(32),
                _block_conv(64),
                _block_conv(64),
                _block_conv(128),
                _block_conv(128)
            )

            size_output_conv = self._stack_conv.compute_output_shape(size_input)

            # replacing RNN with attention.
            self._net_rec = tf.keras.layers.MultiHeadAttention(
                num_heads=nb_heads_att,
                key_dim=size_output_conv,
                value_dim=size_output_conv,
                attention_axes=(0),
                dropout=0.1,
            )

        def call(self, x: tf.Tensor) -> tf.Tensor:
            """
            Input: log-mel spectrograms
                (time, freq bin, channels)
            """
            h = self._stack_conv(x)
            z = self._net_rec(h)