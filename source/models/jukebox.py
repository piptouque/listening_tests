from typing import Callable
import tensorflow as tf

from models.common import VectorQuantiser, ConvDim, ConvDimTranspose

class JukeboxAutoEncoder(tf.keras.Model):
    """Jukebox model"""

    def __init__(self,
                 vector_quantiser: VectorQuantiser,
                 *,
                 nb_filters: int=32,
                 nb_blocks_sample: tuple[int, ...]=(3, 5, 7),
                 size_kernel_sample: int=4,
                 stride_sample: int=2,
                 nb_blocks_res: tuple[int, ...]=(8, 4, 4),
                 size_kernel_res: int=3,
                 rate_dilation_res: int=3,
                 ) -> None:
        tf.debugging.assert_equal(len(nb_blocks_sample), len(nb_blocks_res))
        nb_levels = len(nb_blocks_sample)
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
            for idx_level in tf.range(nb_levels)
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

    def encode(self, x: tf.Tensor) -> list[tf.Tensor]:
        """Encode single audio example x.

        Args:
            x (tf.Tensor): 3-D (batch, time, channel) or 4-D (batch, feature, time, channel) tensor.
            axis (int): Axis into which to insert the levels.

        Returns:
            list[tf.Tensor]: _description_
        """
        return [encoder(x) for encoder in self._encoders]

    def get_code_shapes(self, shape_input: tf.TensorShape) -> list[tf.TensorShape]:
        shapes_code = [encoder.compute_output_shape(
            shape_input) for encoder in self._encoders]
        tf.debugging.Assert(all(
            shape_code[-1] == shapes_code[0][-1] for shape_code in shapes_code), shapes_code)
        return shapes_code

    @tf.function
    def decode(self, zs: tf.TensorArray) -> tf.TensorArray:
        """Decode the multi-level latent codes zs.

        Args:
            zs (tf.Tensor): 3-D tensor or 4-D tensor

        Returns:
            tf.TensorArray: list of 3-D or 4-D tensors.
        """
        xs_hat = tf.TensorArray(zs.dtype, size=zs.size(), infer_shape=True)
        for idx_level, decoder in enumerate(self._decoders):
            xs_hat = xs_hat.write(idx_level, decoder(zs.read(idx_level)))
        return xs_hat

    @tf.function
    def quantise(self, zs: tf.TensorArray) -> tuple[tf.TensorArray, tf.TensorArray, tf.TensorArray]:
        """_summary_

        Args:
            zs (t 3-D tensor or 4-D tensor

        Returns:
            tf.TensorArray: List of 3-D or 4-D tensors.

        Returns:
            dict[str, tf.TensorArray]: The quantise output tensor for each key.
        """
        # _make_arr = lambda: tf.TensorArray(zs.dtype, size=zs.size(), infer_shape=False)
        idx_level = 0
        z = zs.read(idx_level)
        e, z_q, sim = self._vector_quantiser.quantise(z)
        #
        es = tf.TensorArray(e.dtype, size=zs.size(), infer_shape=False)
        zs_q = tf.TensorArray(z.dtype, size=zs.size(), infer_shape=False)
        sims = tf.TensorArray(sim.dtype, size=zs.size(), infer_shape=False)
        #
        es = es.write(idx_level, e)
        zs_q = zs_q.write(idx_level, z_q)
        sims = sims.write(idx_level, sim)
        for idx_level in tf.range(1, zs.size()):
            es = es.write(idx_level, e)
            zs_q = zs_q.write(idx_level, z_q)
            sims = sims.write(idx_level, sim)
        return es, zs_q, sims

    @tf.function
    def lookup_code(self, es: tf.TensorArray) -> tf.TensorArray:
        zs_q = tf.TensorArray(es.dtype, size=es.size(), infer_shape=False)
        for idx_level in tf.range(es.size()):
            zs_q = zs_q.write(
                idx_level, self._vector_quantiser.lookup_code(es.read(idx_level)))
        return zs_q

    @property
    def axes_quantisation(self) -> list[int]:
        return self._vector_quantiser.axes_quantisation

    @property
    def nb_embeddings(self) -> int:
        return self._vector_quantiser.nb_embeddings

    def build(self, shape_input: tf.TensorShape) -> None:
        """Build stuff"""
        self._decoders = [
            self._fn_decoder(shape_input[-1], idx_level)
            for idx_level in tf.range(len(self._encoders))
        ]
        shapes_code = self.get_code_shapes(shape_input)
        self._vector_quantiser.build(shapes_code[0])
        super(JukeboxAutoEncoder, self).build(shape_input)

    def call(self, x: tf.Tensor) -> tf.TensorArray:
        """Process input
        Returns auto-encoding of input
        """
        # fix: reconstruction loss should not go not here
        # since we don't use .call in main Model,
        # this loss becomes out of scope.
        # we don't add quantisation loss either.
        zs = self.encode(x)
        es, zs_q, sims = self.quantise(zs)
        print(type(zs))
        print(type(zs_q))
        # loss_vq = self.get_quantisation_loss( zs_q, zs, self._vector_quantiser._beta, self.axes_quantisation)
        # self.add_loss(loss_vq)
        xs_hat = self.decode(zs_q)
        return xs_hat

    @staticmethod
    def _apply_loss_vec(fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], ys: tf.TensorArray, ys_truth: tf.TensorArray) -> tf.Tensor:
        tf.assert_equal(ys.size(), ys_truth.size())
        return tf.reduce_sum(tf.stack([fn(ys.read(idx_level), ys_truth.read(idx_level)) for idx_level in tf.range(ys.size())], 0), 0)

    @classmethod
    def get_reconstruction_loss(cls, xs_hat: tf.TensorArray, x: tf.Tensor) -> tf.Tensor:
        # for each level, loss over x_hat and x

        # FIXME: bit of a hack there, check that it works.
        # second argument xs_hat not used.
        return tf.identity(
            cls._apply_loss_vec(
                lambda y, _: tf.square(x - y),
                xs_hat,
                tf.TensorArray(xs_hat.dtype, size=xs_hat.size())
            ),
            name="loss_reconstruction"
        )

    @classmethod
    def get_code_reconstruction_loss(cls, zs_hat: tf.TensorArray, zs: tf.TensorArray) -> tf.Tensor:
        return tf.identity(
            cls._apply_loss_vec(
                lambda y, y_truth: tf.square(y_truth - y), zs_hat, zs),
            name="loss_code_reconstruction"
        )

    @classmethod
    def get_quantisation_codebook_loss(cls, zs_q: tf.TensorArray, zs: tf.TensorArray, axes: list[int]) -> tf.Tensor:
        return cls._apply_loss_vec(
            lambda y, y_truth:
                VectorQuantiser.get_quantisation_codebook_loss(
                    y,
                    y_truth,
                    axes=axes
                ),
            zs_q,
            zs
        )

    @classmethod
    def get_quantisation_commit_loss(cls, zs_q: tf.TensorArray, zs: tf.TensorArray, axes: list[int]) -> tf.Tensor:
        return cls._apply_loss_vec(
            lambda y, y_truth:
                VectorQuantiser.get_quantisation_commit_loss(
                    y,
                    y_truth,
                    axes=axes
                ),
            zs_q,
            zs
        )

    @classmethod
    def get_quantisation_loss(cls, zs_q: tf.TensorArray, zs: tf.TensorArray, beta: float, axes: list[int]) -> tf.Tensor:
        return cls._apply_loss_vec(
            lambda y, y_truth:
                VectorQuantiser.get_quantisation_loss(
                    y,
                    y_truth,
                    beta=beta,
                    axes=axes
                ),
            zs_q,
            zs
        )

    @classmethod
    def get_codebook_dissimilarity_loss(cls, es_hat: tf.TensorArray, es: tf.TensorArray, nb_embeddings: int) -> tf.Tensor:
        return cls._apply_loss_vec(
            lambda y, y_truth:
                VectorQuantiser.get_codebook_dissimilarity_loss(
                    y,
                    y_truth,
                    nb_embeddings=nb_embeddings
                ),
            es_hat,
            es
        )

    class ForwardResidualSubBlock(tf.keras.layers.Layer):
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

    class BackwardResidualSubBlock(tf.keras.layers.Layer):
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

    class DownsamplingBlock(tf.keras.layers.Layer):
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
                for _ in tf.range(nb_blocks_res)]

        def call(self, x: tf.Tensor) -> tf.Tensor:
            x_down = self._conv_down(x)
            for block in self._blocks_res:
                x_down = block(x_down)
            z = x_down
            return z

    class UpsamplingBlock(tf.keras.layers.Layer):
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
                for _ in tf.range(nb_blocks_res)]
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
                for _ in tf.range(nb_blocks_down)]

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
                for _ in tf.range(nb_blocks_up)]
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
