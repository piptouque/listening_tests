from typing import Callable
import tensorflow as tf

from models.common import VectorQuantiser, ConvDim, ConvDimTranspose


class JukeboxAutoEncoder(tf.keras.Model):
    """Jukebox model"""

    def __init__(self,
                 vector_quantiser: VectorQuantiser,
                 *,
                 nb_filters: int = 32,
                 nb_blocks_sample: tuple[int, ...] = (3, 5, 7),
                 size_kernel_sample: int = 4,
                 stride_sample: int = 2,
                 nb_blocks_res: tuple[int, ...] = (8, 4, 4),
                 size_kernel_res: int = 3,
                 rate_dilation_res: int = 3,
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

    def decode(self, zs: list[tf.Tensor]) -> list[tf.Tensor]:
        """Decode the multi-level latent codes zs.

        Args:
            zs (tf.Tensor): 3-D tensor or 4-D tensor Returns:
            list[tf.Tensor]: list of 3-D or 4-D tensors.
        """
        return [decoder(zs[idx_level]) for idx_level, decoder in enumerate(self._decoders)]

    def quantise(self, zs: list[tf.Tensor]) -> tuple[list[tf.Tensor], list[tf.Tensor], list[tf.Tensor]]:
        """_summary_

        Args:
            zs (list[tf.Tensor]): _description_

        Returns:
            tuple[list[tf.Tensor], list[tf.Tensor], list[tf.Tensor]]: (ids, quantised, similarity). Lists of 3-D or 4-D tensors.
        """
        # qs_tuple = [(z, z, z) for z in zs]
        qs_tuple = [self._vector_quantiser.quantise(z) for z in zs]
        qs = tuple([
            [
                qs_tuple[idx_level][idx_arg]
                for idx_level in range(len(zs))
            ]
            for idx_arg in range(len(qs_tuple[0]))
        ])
        return qs

    def lookup_code(self, es: list[tf.Tensor]) -> list[tf.Tensor]:
        return [self._vector_quantiser.lookup_code(e) for e in es]

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
            for idx_level in range(len(self._encoders))
        ]
        shapes_code = self.get_code_shapes(shape_input)
        self._vector_quantiser.build(shapes_code[0])
        super(JukeboxAutoEncoder, self).build(shape_input)

    def call(self, x: tf.Tensor) -> list[tf.Tensor]:
        """Process input
        Returns auto-encoding of input
        """
        # fix: reconstruction loss should not go not here
        # since we don't use .call in main Model,
        # this loss becomes out of scope.
        # we don't add quantisation loss either.
        zs = self.encode(x)
        _, zs_q, _ = self.quantise(zs)
        # FIXME: not tested with reduce_sum
        # loss_vq = self.quantisation_loss(zs_q, zs, self._vector_quantiser._beta, self.axes_quantisation)
        # self.add_loss(tf.reduce_sum(tf.stack([tf.reduce_mean(l) for l in loss_vq], 0), 0))
        xs_hat = self.decode(zs_q)
        return xs_hat

    @staticmethod
    def _apply_loss_vec(
        fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        ys: list[tf.Tensor],
        ys_truth: list[tf.Tensor],
        name: str = None
    ) -> list[tf.Tensor]:
        # for each level, loss over x_hat and x
        tf.assert_equal(len(ys), len(ys_truth))
        return [tf.identity(fn(y, ys_truth[idx_level]), name=name) for idx_level, y in enumerate(ys)]

    @classmethod
    def reconstruction_loss(cls, xs_hat: list[tf.Tensor], x: tf.Tensor) -> list[tf.Tensor]:
        return cls._apply_loss_vec(
            lambda y, y_truth: tf.square(y_truth - y),
            xs_hat,
            [x] * len(xs_hat),
            name="loss_reconstruction"
        )

    @classmethod
    def code_reconstruction_loss(cls, zs_hat: list[tf.Tensor], zs: list[tf.Tensor]) -> list[tf.Tensor]:
        return cls._apply_loss_vec(
            lambda y, y_truth: tf.square(y_truth - y),
            zs_hat, zs,
            name="loss_code_reconstruction"
        )

    @classmethod
    def quantisation_codebook_loss(cls, zs_q: list[tf.Tensor], zs: list[tf.Tensor], axes: list[int]) -> list[tf.Tensor]:
        return cls._apply_loss_vec(
            lambda y, y_truth:
                VectorQuantiser.quantisation_codebook_loss(
                    y,
                    y_truth,
                    axes=axes
                ),
            zs_q, zs
        )

    @classmethod
    def quantisation_commit_loss(cls, zs_q: list[tf.Tensor], zs: list[tf.Tensor], axes: list[int]) -> list[tf.Tensor]:
        return cls._apply_loss_vec(
            lambda y, y_truth:
                VectorQuantiser.quantisation_commit_loss(
                    y,
                    y_truth,
                    axes=axes
                ),
            zs_q, zs
        )

    @classmethod
    def quantisation_loss(cls, zs_q: list[tf.Tensor], zs: list[tf.Tensor], beta: float, axes: list[int]) -> list[tf.Tensor]:
        return cls._apply_loss_vec(
            lambda y, y_truth:
                VectorQuantiser.quantisation_loss(
                    y,
                    y_truth,
                    beta=beta,
                    axes=axes
                ),
            zs_q, zs
        )

    @classmethod
    def codebook_dissimilarity_loss(cls, es_hat: list[tf.Tensor], es: list[tf.Tensor], nb_embeddings: int) -> list[tf.Tensor]:
        return cls._apply_loss_vec(
            lambda y, y_truth:
                VectorQuantiser.codebook_dissimilarity_loss(
                    y,
                    y_truth,
                    nb_embeddings=nb_embeddings
                ),
            es_hat, es
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
                for _ in range(nb_blocks_res)]

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
