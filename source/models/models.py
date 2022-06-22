from typing import Tuple, List, Dict
import re

import tensorflow as tf


class BaseConvDim(tf.keras.layers.Layer):
    """Utility variable-dimension Conv"""
    def __init__(self,
        nb_filters: int,
        size_kernel: int,
        rate_dilation: int = 1,
        stride: int = 1,
        activation: str = None
    ) -> None:
        super().__init__()
        self._nb_filters = nb_filters
        self._size_kernel = size_kernel
        self._rate_dilation = rate_dilation
        self._stride = stride
        self._activation = activation
        self._padding = 'same'
        #
        self._conv = None
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._conv(x)

class ConvDim(BaseConvDim):
    """Utility variable-dimension Conv"""
    def build(self, shape_input: Tuple[int, ...]) -> None:
        """_summary_

        Args:
            shape_input (Tuple[int, ...]): _description_

        Raises:
            NotImplementedError: _description_
        """
        if len(shape_input) == 3:
            self._conv = tf.keras.layers.Conv1D(
                self._nb_filters,
                kernel_size=self._size_kernel,
                dilation_rate=self._rate_dilation,
                strides=self._stride,
                padding=self._padding,
                activation=self._activation
            )
        elif len(shape_input) == 4:
            self._conv = tf.keras.layers.Conv2D(
                self._nb_filters,
                kernel_size=self._size_kernel,
                dilation_rate=self._rate_dilation,
                strides=self._stride,
                padding=self._padding,
                activation=self._activation
            )
        else:
            raise NotImplementedError()

class ConvDimTranspose(BaseConvDim):
    """Utility variable-dimension transposed Conv"""
    def build(self, shape_input: Tuple[int, ...]) -> None:
        """_summary_

        Args:
            shape_input (Tuple[int, ...]): _description_

        Raises:
            NotImplementedError: _description_
        """
        if len(shape_input) == 3:
            self._conv = tf.keras.layers.Conv1DTranspose(
                self._nb_filters,
                kernel_size=self._size_kernel,
                dilation_rate=self._rate_dilation,
                strides=self._stride,
                padding=self._padding,
                activation=self._activation
            )
        elif len(shape_input) == 4:
            self._conv = tf.keras.layers.Conv2DTranspose(
                self._nb_filters,
                kernel_size=self._size_kernel,
                dilation_rate=self._rate_dilation,
                strides=self._stride,
                padding=self._padding,
                activation=self._activation
            )
        else:
            raise NotImplementedError()



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

    def __init__(self, nb_embeddings: int, axes: Tuple[int], beta: float) -> None:
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
        self._nb_embeddings = nb_embeddings
        self._shape_embedding = None
        self._vecs_embedding = None

    def build(self, shape_input: Tuple[int, ...]) -> None:
        super(VectorQuantiser, self).build(shape_input)
        shape_embedding = (self._nb_embeddings,)
        shape_embedding = shape_embedding + tuple([shape_input[axis] for axis in self._axes])
        self._shape_embedding = shape_embedding
        self._vecs_embedding = self.add_weight(
            shape=self._shape_embedding,
            # initializer=lambda: tf.random.uniform(shape=self._shape_embedding),
            trainable=True,
            name="codebook"
        )

    def get_codebook(self) -> tf.Tensor:
        return tf.constant(self._vecs_embedding)

    def get_code_lookup(self, ids_embedding: tf.Tensor) -> tf.Tensor:
        return tf.nn.embedding_lookup(self._vecs_embedding, ids_embedding)

    def call(self, z_e: tf.Tensor) -> tf.Tensor:
        """_summary_

        Args:
            z_e (tf.Tensor): dim -> (batch, not quantised dim (time), quantised dim (channel)) 

        Returns:
            tf.Tensor: _description_
        """
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
        z_q = self.get_code_lookup(ids_embedding)
        # similarity as normalised dot product of latent and embedding vectors
        similarity = (dot_z_embedding / tf.sqrt(norm_sq_z)) / tf.sqrt(norm_sq_embedding)
        # fix: tf.norm does not interpret 1-uples as integers.
        axes = self._axes[0] if len(self._axes) == 1 else self._axes
        loss_commit = tf.identity(tf.reduce_mean(tf.norm(tf.stop_gradient(z_e) - z_q, axis=axes)), name='loss_commit')
        loss_codebook = tf.identity(tf.reduce_mean(tf.norm(z_e - tf.stop_gradient(z_q), axis=axes)), name='loss_codebook')
        loss_vq = tf.identity(loss_codebook + loss_commit * self._beta, name='loss_vq')
        self.add_loss(loss_vq)
        return {
            'quantised': z_q,
            'similarity': similarity,
            'ids_codebook': ids_embedding
        }

class JukeboxAutoEncoder(tf.keras.Model):
    """Jukebox model"""

    class ForwardResidualSubBlock(tf.keras.Model):
        """Frequently used block in the article"""
        def __init__(self,
                nb_filters: int,
                size_kernel: int,
                rate_dilation: int,
            ) -> None:
            super(JukeboxAutoEncoder.ForwardResidualSubBlock, self).__init__()
            self._convs = tf.keras.Sequential([
                ConvDim(
                    nb_filters=nb_filters,
                    size_kernel=size_kernel,
                    rate_dilation=rate_dilation,
                    activation='relu'
                ),
                ConvDim(
                    nb_filters=nb_filters,
                    size_kernel=size_kernel,
                    rate_dilation=1,
                    activation='relu'
                )
            ])
        def call(self, x: tf.Tensor) -> tf.Tensor:
            h = self._convs(x) + x
            return h

    class BackwardResidualSubBlock(tf.keras.Model):
        """Frequently used block in the article"""
        def __init__(self,
                nb_filters: int,
                size_kernel: int,
                rate_dilation: int,
            ) -> None:
            super(JukeboxAutoEncoder.BackwardResidualSubBlock, self).__init__()
            self._convs = tf.keras.Sequential([
                ConvDim(
                    nb_filters=nb_filters,
                    size_kernel=size_kernel,
                    rate_dilation=1,
                    activation='relu'
                ),
                ConvDim(
                    nb_filters=nb_filters,
                    size_kernel=size_kernel,
                    rate_dilation=rate_dilation,
                    activation='relu'
                )
            ])
        def call(self, x: tf.Tensor) -> tf.Tensor:
            h = self._convs(x) + x
            return h

    class DownsamplingBlock(tf.keras.Model):
        """Downsampling block in the encoder"""
        def __init__(self,
            nb_filters: int,
            size_kernel_down: int,
            stride_down: int,
            nb_blocks_res: int, 
            size_kernel_res: int,
            rate_dilation_res: int,
        ) -> None:
            super(JukeboxAutoEncoder.DownsamplingBlock, self).__init__()
            self._conv_down = ConvDim(
                nb_filters=nb_filters,
                size_kernel=size_kernel_down,
                stride=stride_down,
                activation='relu'
            )
            self._blocks_res = [
                JukeboxAutoEncoder.ForwardResidualSubBlock(
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
            nb_filters: int,
            size_kernel_up: int,
            stride_up: int,
            nb_blocks_res: int,
            size_kernel_res: int,
            rate_dilation_res: int,
        ) -> None:
            super(JukeboxAutoEncoder.UpsamplingBlock, self).__init__()
            self._blocks_res = [
                JukeboxAutoEncoder.BackwardResidualSubBlock(
                    nb_filters=nb_filters,
                    size_kernel=size_kernel_res,
                    rate_dilation=rate_dilation_res
                )
            for _ in range(nb_blocks_res)]
            self._conv_up = ConvDimTranspose(
                nb_filters=nb_filters,
                size_kernel=size_kernel_up,
                stride=stride_up,
                activation='relu'
            )
        def call(self, z: tf.Tensor) -> tf.Tensor:
            for block in self._blocks_res:
                z = block(z)
            x = self._conv_up(z)
            return x


    class Encoder(tf.keras.Model):
        """Encoder without quantisation"""
        def __init__(self,
            nb_filters: int,
            nb_blocks_down: int,
            size_kernel_down: int,
            stride_down: int,
            nb_blocks_res: int,
            size_kernel_res: int,
            rate_dilation_res: int
        ) -> None:
            super(JukeboxAutoEncoder.Encoder, self).__init__()
            self._blocks_down = [
               JukeboxAutoEncoder.DownsamplingBlock(
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
            nb_channels_output: int,
            nb_filters: int,
            nb_blocks_up: int,
            size_kernel_up: int,
            stride_up: int,
            nb_blocks_res: int,
            size_kernel_res: int,
            rate_dilation_res: int
        ) -> None:
            super(JukeboxAutoEncoder.Decoder, self).__init__()
            self._blocks_up = [
               JukeboxAutoEncoder.UpsamplingBlock(
                   nb_filters=nb_filters,
                   size_kernel_up=size_kernel_up,
                   stride_up=stride_up,
                   nb_blocks_res=nb_blocks_res,
                   size_kernel_res=size_kernel_res,
                   rate_dilation_res=rate_dilation_res
               ) 
            for _ in range(nb_blocks_up)]
            self._conv_proj = ConvDim(
                nb_filters=nb_channels_output,
                size_kernel=size_kernel_res,
                stride=1,
                activation=None
            )
        def call(self, z: tf.Tensor) -> tf.Tensor:
            h = z
            for block in self._blocks_up:
                h = block(h)
            x_hat = self._conv_proj(h)
            return x_hat


    def __init__(self,
        vector_quantiser: VectorQuantiser,
        nb_levels: int=3,
        nb_filters: int=32,
        nb_blocks_sample: List[int]=(3, 5, 7),
        size_kernel_sample: int=4,
        stride_sample: int=2,
        nb_blocks_res: List[int]=(8, 4, 4),
        size_kernel_res: int=3,
        rate_dilation_res: int=3,
    ) -> None:
        tf.debugging.assert_equal(tf.rank(nb_blocks_sample), tf.rank(nb_blocks_res))
        super(JukeboxAutoEncoder, self).__init__()
        self._encoders = [
            self.Encoder(
                nb_filters=nb_filters,
                nb_blocks_down=nb_blocks_sample[idx_level],
                size_kernel_down=size_kernel_sample,
                stride_down=stride_sample,
                nb_blocks_res=nb_blocks_res[idx_level],
                size_kernel_res=size_kernel_res,
                rate_dilation_res=rate_dilation_res 
            )
            for idx_level in range(nb_levels)
        ]
        self._fn_decoder = lambda nb_channels_output, i: self.Decoder(
            nb_channels_output=nb_channels_output,
            nb_filters=nb_filters,
            nb_blocks_up=nb_blocks_sample[i],
            size_kernel_up=size_kernel_sample,
            stride_up=stride_sample,
            nb_blocks_res=nb_blocks_res[i],
            size_kernel_res=size_kernel_res,
            rate_dilation_res=rate_dilation_res
        )
        self._decoders = None
        # The shapes of the codes are different for each level,
        # only the last axis is quantised, which should be the same for all levels
        self._vector_quantiser = vector_quantiser
        self._shape_input = None

    def build(self, shape_input: Tuple[int, ...]) -> None:
        """Build stuff"""
        self._decoders = [
            self._fn_decoder(shape_input[-1], idx_level)
            for idx_level in range(len(self._encoders))
        ]
        super(JukeboxAutoEncoder, self).build(shape_input)
        shapes_code = self.get_code_shapes(shape_input)
        tf.debugging.Assert(all(shape_code[-1] == shapes_code[0][-1] for shape_code in shapes_code), shapes_code)
        self._vector_quantiser.build(shapes_code[0])

    def encode(self, x: tf.Tensor) -> List[tf.Tensor]:
        """Encode single audio example x.

        Args:
            x (tf.Tensor): 3-D (batch, time, channel) or 4-D (batch, feature, time, channel) tensor.
            axis (int): Axis into which to insert the levels.

        Returns:
            tf.Tensor: _description_
        """
        return [encoder(x) for encoder in self._encoders]

    def get_code_shapes(self, shape_input: Tuple[int, ...]) -> List[tf.TensorShape]:
        return [encoder.compute_output_shape(shape_input) for encoder in self._encoders]

    def decode(self, zs: List[tf.Tensor]) -> List[tf.Tensor]:
        """Decode the multi-level latent codes zs.

        Args:
            zs (tf.Tensor): 3-D tensor or 4-D tensor

        Returns:
            List[tf.Tensor]: List of 3-D or 4-D tensors.
        """
        xs_hat = [
            decoder(zs[idx_level])
            for idx_level, decoder in enumerate(self._decoders)
        ]
        return xs_hat

    def quantise(self, zs: List[tf.Tensor]) -> Dict[str, List[tf.Tensor]]:
        """_summary_

        Args:
            zs (t 3-D tensor or 4-D tensor

        Returns:
            List[tf.Tensor]: List of 3-D or 4-D tensors.

        Returns:
            Dict[str, List[tf.Tensor]]: The quantise output tensor for each key.
        """
        quantise_vq = [self._vector_quantiser(z) for z in zs]
        keys_vq = quantise_vq[0].keys()
        quantise_dict_vq = dict([
            (key, [quantise_vq[idx_level][key] for idx_level in range(len(zs))])
            for key in keys_vq
        ])
        return quantise_dict_vq

    def call(self, x: tf.Tensor) -> List[tf.Tensor]:
        """Process input
        Returns auto-encoding of input
        """
        zs = self.encode(x)
        zs_q = self.quantise(zs)['quantised']
        xs_hat = self.decode(zs_q)
        # fix: reconstruction loss should not go not here
        # since we don't use .call in main Model,
        # this loss becomes out of scope.
        # loss_reconstruct = self.get_reconstruction_loss(x, tf.stack(xs_hat, axis=0), axis=0)
        # self.add_loss(loss_reconstruct)
        return xs_hat
    
    def get_reconstruction_loss(self, x: tf.Tensor, xs_hat: tf.Tensor, axis: int) -> tf.Tensor:
        # for each level, loss over x_hat and x
        return tf.identity(tf.reduce_mean(
            tf.math.reduce_sum(
                tf.keras.metrics.mean_squared_error(x, xs_hat),
                axis=axis
            ),
            axis=None
        ), name="loss_reconstruction")

class CouplingTranscoder(tf.keras.Model):
    """Coupling solver"""
    def __init__(self,
        vector_quantiser: VectorQuantiser,
        conv_nb_filters: int,
        conv_size_kernel: int,
        att_nb_heads: int
    ) -> None:
        super(CouplingTranscoder, self).__init__()
        self._att_nb_heads = att_nb_heads
        #
        self._att = None
        self._conv = ConvDim(
                nb_filters=conv_nb_filters,
                size_kernel=conv_size_kernel,
                stride=1,
                activation='relu'
        )
        #
        self._vector_quantiser = vector_quantiser
        #
        # self._loss_tracker = tf.keras.metrics.Mean(name='loss_coupling')

    def build(self, shape_input: Tuple[int, ...]) -> None:
        # self._att = tf.keras.layers.MultiHeadAttention(num_heads=self._att_nb_heads, key_dim=)
        self._vector_quantiser.build(shape_input)

    def call(self, e_a: tf.Tensor) -> tf.Tensor:
        e_b = e_a
        return e_b

    def compute_loss(self, e_a: tf.Tensor, e_b: tf.Tensor, e_b_hat: tf.Tensor, sample_weight: tf.Tensor) -> tf.Tensor:
        z_q_b = self._vector_quantiser.get_code_lookup(e_b)
        z_q_b_hat = self._vector_quantiser.get_code_lookup(e_b_hat)
        loss_coupling = tf.reduce_mean(tf.keras.metrics.mean_squared_error(z_q_b, z_q_b_hat))
        loss_coupling += tf.add_n(self.losses) 
        loss_coupling = tf.identity(loss_coupling, name='loss_coupling') 
        # self.loss_tracker.update_state(loss_coupling)
        return loss_coupling
    
    # def reset_metrics(self) -> None:
        # self._loss_tracker.reset_states()

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return [self._loss_tracker]


class CouplingSolver(tf.keras.Model):
    """Coupling solver"""
    def __init__(self,
        vector_quantiser: VectorQuantiser,
        nb_levels: int,
        conv_nb_filters: int,
        conv_size_kernel: int,
        att_nb_heads: int 
    ) -> None:
        super(CouplingSolver, self).__init__()
        #
        self._transcoders = [
            CouplingTranscoder(
                vector_quantiser,
                conv_nb_filters,
                conv_size_kernel,
                att_nb_heads
            ) for _ in range(nb_levels)]
        #

    def call(self, es_a: List[tf.Tensor]) -> List[tf.Tensor]:
        """_summary_

        Args:
            e_a (tf.Tensor): _description_

        Returns:
            tf.Tensor: _description_
        """
        es_b = [self._transcoders[idx_level](es_a[idx_level]) for idx_level in range(len(self._transcoders))]
        return es_b

class SomethingModel(tf.keras.Model):
    def __init__(self, vector_quantiser: VectorQuantiser, auto_encoder: JukeboxAutoEncoder, coupling_solver: CouplingSolver) -> None:
        super(SomethingModel, self).__init__()
        self._vector_quantiser = vector_quantiser
        self._auto_encoder = auto_encoder
        self._coupling_solver = coupling_solver

    def build(self, shape_input: Tuple[int, ...]) -> None:
        # super(SomethingModel, self).build(shape_input)
        # vector_quantiser is built by the other sub-models.
        self._auto_encoder.build(shape_input)
        shapes_code = self._auto_encoder.get_code_shapes(shape_input)
        self._coupling_solver.build(shapes_code)
    
    def call(self, x_a: tf.Tensor) -> tf.Tensor:
        zs_a = self._auto_encoder.encode(x_a)
        zs_dict_vq_a = self._auto_encoder.quantise(zs_a)
        es_a = zs_dict_vq_a['ids_codebook']

        es_b = self._coupling_solver(es_a)
        zs_q_b = [self._vector_quantiser.get_code_lookup(e_b) for e_b in es_b]
        xs_b_hat = self._auto_encoder.decode(zs_q_b)

        #
        zs_q_a = zs_dict_vq_a['quantised']
        xs_hat_a = self._auto_encoder.decode(zs_q_a)
        loss_reconstruct_a = self._auto_encoder.get_reconstruction_loss(x_a, tf.stack(xs_hat_a, axis=0), axis=0)
        self.add_loss(loss_reconstruct_a)
        return xs_b_hat 

    def compute_loss(self, x_a, x_b, x_b_hat, sample_weight) -> tf.Tensor:
        # TODO: match name of loss
        _PAT_LOSS_NAME = re.compile(r'/loss_(?P<name>\w)(_\d)?:\d$')
        loss = tf.add_n(self.losses)
        for loss in self.losses:
            m_loss = re.search(_PAT_LOSS_NAME, loss.name)
            if m_loss is not None:
                name_loss = m_loss.group(0)
        return loss
