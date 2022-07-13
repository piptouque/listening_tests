from typing import Optional

import tensorflow as tf


class BaseConvDim(tf.keras.layers.Layer):
    """Utility variable-dimension Conv"""

    def __init__(self,
                 nb_filters: int,
                 size_kernel: int,
                 rate_dilation: int = 1,
                 stride: int = 1,
                 activation: Optional[str] = None
                 ) -> None:
        super().__init__()
        self._nb_filters = nb_filters
        self._size_kernel = size_kernel
        self._rate_dilation = rate_dilation
        self._stride = stride
        self._activation = activation
        self._padding = 'same'
        #
        self._conv: tf.keras.layers.Layer = lambda x: None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._conv(x)


class ConvDim(BaseConvDim):
    """Utility variable-dimension Conv"""

    def build(self, shape_input: tuple[int, ...]) -> None:
        """_summary_

        Args:
            shape_input (tuple[int, ...]): _description_

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

    def build(self, shape_input: tuple[int, ...]) -> None:
        """_summary_

        Args:
            shape_input (tuple[int, ...]): _description_

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
                 causal: bool = False,
                 use_bias: bool = True
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

    def __init__(self, nb_embeddings: int, axes: tuple[int], beta: float) -> None:
        """
            axis: 0 -> first non-batch dim

            ---
            _vecs_embedding: dim -> (nb_embeddings, not quantised dim)
        """
        super(VectorQuantiser, self).__init__()
        self._axes = tuple(axes)
        self._axes_embedding = tuple(range(1, len(self._axes)+1))
        self._beta = beta
        #
        self._nb_embeddings = nb_embeddings
        self._shape_embedding = None
        self._vecs_embedding = None

    def get_codebook(self) -> tf.Tensor:
        return tf.constant(self._vecs_embedding)

    def lookup_code(self, e: tf.Tensor) -> tf.Tensor:
        return tf.nn.embedding_lookup(self._vecs_embedding, e)

    @property
    def axes_quantisation(self) -> list[int]:
        return self._axes

    @property
    def nb_embeddings(self) -> int:
        return self._nb_embeddings

    @staticmethod
    def quantisation_codebook_loss(z_q: tf.Tensor, z: tf.Tensor, axes: list[int]) -> tf.Tensor:
        # fix: tf.norm does not interpret 1-uples as integers.
        axes = axes[0] if len(axes) == 1 else axes
        loss_codebook = tf.identity(
            tf.norm(z - tf.stop_gradient(z_q), axis=axes), name='loss_codebook')
        return loss_codebook

    @staticmethod
    def quantisation_commit_loss(z_q: tf.Tensor, z: tf.Tensor, axes: list[int]) -> tf.Tensor:
        # fix: tf.norm does not interpret 1-uples as integers.
        axes = axes[0] if len(axes) == 1 else axes
        loss_commit = tf.identity(
            tf.norm(tf.stop_gradient(z) - z_q, axis=axes), name='loss_commit')
        return loss_commit

    @classmethod
    def quantisation_loss(cls, z_q: tf.Tensor, z: tf.Tensor, beta: float, axes: list[int]) -> tf.Tensor:
        # fix: tf.norm does not interpret 1-uples as integers.
        loss_codebook = cls.quantisation_codebook_loss(z_q, z, axes=axes)
        loss_commit = cls.quantisation_commit_loss(z_q, z, axes=axes)
        loss_vq = tf.identity(
            loss_codebook + loss_commit * beta, name='loss_vq')
        return loss_vq

    @staticmethod
    def codebook_dissimilarity_loss(e_hat: tf.Tensor, e: tf.Tensor, nb_embeddings: int) -> tf.Tensor:
        # get 'one-hot' representation
        # then cosine similarity
        return tf.identity(
            1 + tf.keras.losses.cosine_similarity(
                tf.one_hot(e_hat, depth=nb_embeddings,
                           on_value=1.0, off_value=0.0),
                tf.one_hot(e, depth=nb_embeddings,
                           on_value=1.0, off_value=0.0)
            ),
            name="loss_similarity_codebook"
        )

    def quantise(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """_summary_

        Args:
    z (tf.Tensor): dim -> (batch, not quantised dim (time), quantised dim (channel))

        Returns:
            tf.Tensor: _description_
        """
        # tf.debugging.assert_equal(tf.shape(z_e)[1:], self._shape_code[1:])
        # dim -> (batch, not quantised dim, nb_embeddings)
        dot_z_embedding = tf.tensordot(z, self._vecs_embedding, axes=[
            self._axes, self._axes_embedding])
        norm_sq_z = tf.reduce_sum(tf.square(z), axis=self._axes)
        # dim -> (batch, not quantised dim, 1)
        norm_sq_embedding = tf.reduce_sum(
            tf.square(self._vecs_embedding), axis=self._axes_embedding)
        # Some brodcast operations
        norm_sq_z = tf.expand_dims(norm_sq_z, axis=-1)
        # DOT NOT CHANGE ORDER OF OPERATIONS
        # dim -> (batch, not quantised dim, nb_embeddings)
        # Euclidean distance as: dist(x,y) = norm(x)^2 + norm(y)^2  - 2<x,y>
        dist_quantisation = norm_sq_z + \
            (- 2 * dot_z_embedding + norm_sq_embedding)
        e = tf.argmin(dist_quantisation, axis=-1)
        z_q = self.lookup_code(e)
        # similarity as normalised dot product of latent and embedding vectors
        similarity = (dot_z_embedding / tf.sqrt(norm_sq_z)) / \
            tf.sqrt(norm_sq_embedding)
        return e, z_q, similarity

    def build(self, shape_input: tuple[int, ...]) -> None:
        shape_embedding = (self._nb_embeddings,)
        shape_embedding = shape_embedding + \
            tuple([shape_input[axis] for axis in self._axes])
        self._shape_embedding = shape_embedding
        self._vecs_embedding = self.add_weight(
            shape=self._shape_embedding,
            # initializer=lambda: tf.random.uniform(shape=self._shape_embedding),
            trainable=True,
            name="codebook"
        )
        super(VectorQuantiser, self).build(shape_input)

    def call(self, z: tf.Tensor) -> tf.Tensor:
        """_summary_

        Args:
            z (tf.Tensor): dim -> (batch, not quantised dim (time), quantised dim (channel)) 

        Returns:
            tf.Tensor: _description_
        """
        e, z_q, sim = self.quantise(z)
        # not here
        loss_vq = self.quantisation_loss(
            z_q, z, self._beta, self._axes)
        self.add_loss(loss_vq)
        return e, z_q, sim
