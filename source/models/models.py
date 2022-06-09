from typing import Tuple, List

import tensorflow as tf


class GatedActivationUnit(tf.keras.layers.Layer):
    """Classic 'activation' used in some models"""
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

    def __init__(self, shape_code: Tuple[int], nb_embeddings: int, axes: Tuple[int], beta: float) -> None:
        """
            axis: 0 -> first non-batch dim

            ---
            _vecs_embedding: dim -> (nb_embeddings, not quantised dim)
        """
        super(VectorQuantiser, self).__init__()
        self._axes = tuple(axes) 
        self._axes_embedding = tuple(range(1,len(self._axes)+1))
        self._beta = beta
        #
        shape_embedding = (nb_embeddings,)
        shape_embedding = shape_embedding + tuple([shape_code[axis] for axis in self._axes])
        self._shape_embedding = shape_embedding
        self._vecs_embedding = None

    def call(self, z_e: tf.Tensor) -> tf.Tensor:
        """
            z_e: dim -> (batch, not quantised dim (time), quantised dim (channel)) 
        """
        if self._vecs_embedding is None:
            self._vecs_embedding = tf.Variable(
                trainable=True,
                initial_value=tf.random.uniform(shape=self._shape_embedding, dtype=z_e.dtype)
            )
        # tf.debugging.assert_equal(tf.shape(z_e)[1:], self._shape_code[1:])
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
        dist_quantisation  = (- 2 * dot_z_embedding + norm_sq_z) + norm_sq_embedding
        ids_embedding = tf.argmin(dist_quantisation, axis=-1)
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

class JukeboxModel(tf.keras.Model):
    """Jukebox model"""

    @staticmethod
    def _conv_dim(
        nb_dim_data: int,
        nb_filters: int,
        size_kernel: int,
        rate_dilation: int = 1,
        stride: int = 1
    ) -> tf.keras.layers.Layer:
        if nb_dim_data == 1:
            return tf.keras.layers.Conv1D(
                   nb_filters,
                   kernel_size=size_kernel,
                   dilation_rate=rate_dilation,
                   strides=stride,
                   padding='same'
                )
        elif nb_dim_data == 2:
            return tf.keras.layers.Conv2D(
                   nb_filters,
                   kernel_size=size_kernel,
                   dilation_rate=rate_dilation,
                   strides=stride,
                   padding='same'
                )
        else:
            raise NotImplementedError()

    @staticmethod
    def _conv_transpose_dim(
        nb_dim_data: int,
        nb_filters: int,
        size_kernel: int,
        rate_dilation: int = 1,
        stride: int = 1
    ) -> tf.keras.layers.Layer:
        if nb_dim_data == 1:
            return tf.keras.layers.Conv1DTranspose(
                   nb_filters,
                   kernel_size=size_kernel,
                   dilation_rate=rate_dilation,
                   strides=stride,
                   padding='same'
                )
        elif nb_dim_data == 2:
            return tf.keras.layers.Conv2DTranspose(
                   nb_filters,
                   kernel_size=size_kernel,
                   dilation_rate=rate_dilation,
                   strides=stride,
                   padding='same'
                )
        else:
            raise NotImplementedError()

    class ResidualSubBlock(tf.keras.Model):
        """Often used block in the article"""
        def __init__(self,
                nb_dim_data: int,
                nb_filters: int,
                size_kernel: int,
                rate_dilation: int,
            ) -> None:
            super(JukeboxModel.ResidualSubBlock, self).__init__()
            self._convs = tf.keras.Sequential([
                JukeboxModel._conv_dim(nb_dim_data=nb_dim_data, nb_filters=nb_filters, size_kernel=size_kernel, rate_dilation=rate_dilation),
                JukeboxModel._conv_dim(nb_dim_data=nb_dim_data, nb_filters=nb_filters, size_kernel=size_kernel, rate_dilation=1)
            ])
        def call(self, x: tf.Tensor) -> tf.Tensor:
            h = self._convs(x) + x
            return h

    class DownsamplingBlock(tf.keras.Model):
        """Downsampling block in the encoder"""
        def __init__(self,
            nb_dim_data: int,
            nb_filters: int,
            size_kernel_down: int,
            stride_down: int,
            nb_blocks_res: int, 
            size_kernel_res: int,
            rate_dilation_res: int,
        ) -> None:
            super(JukeboxModel.DownsamplingBlock, self).__init__()
            self._conv_down = JukeboxModel._conv_dim(
                nb_dim_data=nb_dim_data,
                nb_filters=nb_filters,
                size_kernel=size_kernel_down,
                stride=stride_down,
            )
            self._blocks_res = [
                JukeboxModel.ResidualSubBlock(
                    nb_dim_data=nb_dim_data,
                    nb_filters=nb_filters,
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
        """Upsampling block in the decoder"""
        def __init__(self,
            nb_dim_data: int,
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
                    nb_dim_data=nb_dim_data,
                    nb_filters=nb_filters,
                    size_kernel=size_kernel_res,
                    rate_dilation=rate_dilation_res
                )
            for _ in range(nb_blocks_res)]
            self._conv_up = JukeboxModel._conv_transpose_dim(
                nb_dim_data=nb_dim_data,
                nb_filters=nb_filters,
                size_kernel=size_kernel_up,
                stride=stride_up,
            )
        def call(self, z: tf.Tensor) -> tf.Tensor:
            for block in self._blocks_res:
                z = block(z)
            x = self._conv_up(z)
            return x


    class Encoder(tf.keras.Model):
        """Encoder without quantisation"""
        def __init__(self,
            nb_dim_data: int,
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
                   nb_dim_data=nb_dim_data,
                   nb_filters=nb_filters,
                   size_kernel_down=size_kernel_down,
                   stride_down=stride_down,
                   nb_blocks_res=nb_blocks_res,
                   size_kernel_res=size_kernel_res,
                   rate_dilation_res=rate_dilation_res
               ) 
            for _ in range(nb_blocks_down)]
        def call(self, x: tf.Tensor) -> tf.Tensor:
            for block in self._blocks_down:
                x = block(x)
            return x


    class Decoder(tf.keras.Model):
        """Decoder"""
        def __init__(self,
            nb_dim_data: int,
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
                   nb_dim_data=nb_dim_data,
                   nb_filters=nb_filters,
                   size_kernel_up=size_kernel_up,
                   stride_up=stride_up,
                   nb_blocks_res=nb_blocks_res,
                   size_kernel_res=size_kernel_res,
                   rate_dilation_res=rate_dilation_res
               ) 
            for _ in range(nb_blocks_up)]
            self._conv_proj = JukeboxModel._conv_dim(
                nb_dim_data=nb_dim_data,
                nb_filters=nb_channels_output,
                size_kernel=size_kernel_res,
                stride=1,
            )
        def call(self, z: tf.Tensor) -> tf.Tensor:
            h = z
            for block in self._blocks_up:
                h = block(h)
            x_hat = self._conv_proj(h)
            return x_hat


    def __init__(self,
        shape_input: Tuple[int, ...],
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
        self._shape_input = shape_input
        self._nb_levels = nb_levels
        nb_dim_data = len(self._shape_input) - 2
        self._encoders = [
            self.Encoder(
                nb_dim_data=nb_dim_data,
                nb_filters=nb_filters,
                nb_blocks_down=nb_blocks_sample[idx_level],
                size_kernel_down=size_kernel_sample,
                stride_down=stride_sample,
                nb_blocks_res=nb_blocks_res[idx_level],
                size_kernel_res=size_kernel_res,
                rate_dilation_res=rate_dilation_res 
            )
            for idx_level in range(self._nb_levels)
        ]
        self._decoders = [
            self.Decoder(
                nb_dim_data=nb_dim_data,
                nb_channels_output=self._shape_input[-1],
                nb_filters=nb_filters,
                nb_blocks_up=nb_blocks_sample[idx_level],
                size_kernel_up=size_kernel_sample,
                stride_up=stride_sample,
                nb_blocks_res=nb_blocks_res[idx_level],
                size_kernel_res=size_kernel_res,
                rate_dilation_res=rate_dilation_res 
            )
            for idx_level in range(self._nb_levels)
        ]
        # The shapes of the codes are different for each level,
        # only the last axis is quantised, which should be the same for all levels
        shapes_code = [encoder.compute_output_shape(self._shape_input) for encoder in self._encoders]
        tf.debugging.Assert(all(shape_code[-1] == shapes_code[0][-1] for shape_code in shapes_code), shapes_code)
        axes_quantise = [-1] # [2] if nb_dim_data == 1 else [3]
        self._codebook = VectorQuantiser(
            shape_code=shapes_code[-1],
            nb_embeddings=size_codebook,
            axes=axes_quantise,
            beta=beta_codebook
        )
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        xs_hat = None
        for idx_level in range(self.nb_levels):
            z = self._encoders[idx_level](x)
            z_q = self._codebook(z)['quantised']
            x_hat = self._decoders[idx_level](z_q)
            if idx_level == 0:
                xs_hat = tf.expand_dims(x_hat, axis=-1)
            else:
                xs_hat = tf.concat([xs_hat, tf.expand_dims(x_hat, axis=-1)], -1)
        # xs = tf.stack([x] * len(self._encoders), axis=-1)
        xs = tf.repeat(tf.expand_dims(x, axis=-1), repeats=self.nb_levels, axis=-1)
        # for each level, loss over x_hat and x
        self.add_loss(
            tf.math.reduce_mean(
                tf.math.reduce_sum(
                    tf.keras.metrics.mean_squared_error(xs, xs_hat),
                    axis=-1
                ),
                axis=None
            )
        )
        return xs_hat 

    @property
    def nb_levels(self) -> int:
        return self._nb_levels