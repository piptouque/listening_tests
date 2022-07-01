from typing import Optional, Union, Callable

import tensorflow as tf

from models.common import VectorQuantiser


_RATE_DROPOUT = 0.1


class Transformer(tf.keras.Model):
    _EPSILON = 1e-6
    _MAX_LENGTH_TOKENS = 128

    def __init__(self,
                 size_vocab_input: int,
                 size_vocab_target: int,
                 nb_blocks: int,
                 size_model: int,
                 size_feedforward: int,
                 nb_heads: int,
                 *,
                 rate_dropout=_RATE_DROPOUT
                 ) -> None:
        super(Transformer, self).__init__()
        self._encoder = Transformer.Encoder(
            size_vocab_input=size_vocab_input,
            nb_blocks=nb_blocks,
            size_model=size_model,
            size_feedforward=size_feedforward,
            nb_heads=nb_heads,
            rate_dropout=rate_dropout
        )
        self._decoder = Transformer.Decoder(
            size_vocab_target=size_vocab_target,
            nb_blocks=nb_blocks,
            size_model=size_model,
            size_feedforward=size_feedforward,
            nb_heads=nb_heads,
            rate_dropout=rate_dropout
        )
        self._layer_fit_output = tf.keras.layers.Dense(size_vocab_target)

    def call(self,
             inputs: tuple[tf.Tensor, ...],
             *args,
             return_weights: bool = False,
             **kwargs
             ) -> tf.Tensor:
        input_, target = inputs
        mask_padding, mask_look_ahead = self._make_masks(input_, target)
        output_encoder = self._encoder(
            input_,
            *args,
            mask_padding=mask_padding,
            **kwargs
        )
        output_decoder, weights_att = self._decoder(
            [target, output_encoder],
            *args,
            mask_look_ahead=mask_look_ahead,
            mask_padding=mask_padding,
            **kwargs
        )
        output = self._layer_fit_output(output_decoder)
        if return_weights:
            return output, weights_att
        else:
            return output

    @staticmethod
    def mask_loss(fn_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], value_mask: int = 0) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        # mask the padding token (0)
        def _loss_masked(y: tf.Tensor, y_truth: tf.Tensor) -> tf.Tensor:
            mask = tf.math.logical_not(tf.math.equal(y_truth, value_mask))
            loss_ = fn_loss(y, y_truth)
            loss_ *= tf.cast(msk, loss_.dtype)
            return loss_ / tf.reduce_sum(mask)
        return _loss_masked

    @staticmethod
    def mask_accuracy(fn_accuracy: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], value_mask: int = 0) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        def _accuracy_masked(y: tf.Tensor, y_truth: tf.Tensor) -> tf.Tensor:
            acc_ = fn_accuracy(y, y_truth)
            mask = tf.math.logical_not(tf.math.equal(y_truth, value_mask))
            acc_ = tf.math.logical_and(acc_, mask)
            acc_ = tf.cast(acc_, tf.float32)
            mask = tf.cast(mask, tf.float32)
            return acc_ / tf.reduce_sum(mask)
        return _accuracy_masked

    @staticmethod
    def make_positional_encoding(nb_embeddings: int, size_model: int) -> tf.Tensor:
        pos = tf.expand_dims(tf.range(nb_embeddings), -1)
        ids_pos = tf.expand_dims(tf.range(size_model), 0)
        angles = pos / tf.math.pow(
            10_000, (2 * (ids_pos//2)) // tf.cast(size_model, tf.float32)
        )
        angles[..., ::2] = tf.sin(angles[..., ::2])
        angles[..., 1::2] = tf.cos(angles[..., 1::2])
        encoding_pos = tf.cast(tf.expand_dims(angles, 0), tf.float32)
        return encoding_pos

    @staticmethod
    def _make_padding_mask(input_: tf.Tensor) -> tf.Tensor:
        seq = input_
        mask_padding = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions to add to the attention logits
        mask_padding = tf.expand_dims(tf.expand_dims(mask_padding, 1), 2)
        return mask_padding

    @classmethod
    def _make_masks(cls, input_: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        mask_padding = cls._make_padding_mask(input_)
        #
        mask_padding_dec_target = cls._make_padding_mask(target)
        #
        size = target.shape[-2]
        mask_look_ahead = 1 - \
            tf.linalg.band_part(tf.ones((size, size)),
                                num_lower=-1, num_upper=0)
        mask_look_ahead = tf.maximum(mask_look_ahead, mask_padding_dec_target)
        return mask_padding, mask_look_ahead

    class PointwiseFeedForwardLayer(tf.keras.layers.Layer):
        def __init__(self,
                     size_model: int,
                     size_feedforward: int
                     ) -> None:
            super(Transformer.PointwiseFeedForwardLayer, self).__init__()
            self._layer = tf.keras.Sequential([
                tf.keras.layers.Dense(size_feedforward, activation='relu'),
                tf.keras.layers.Dense(size_model)
            ])

        def call(self, x: tf.Tensor) -> tf.Tensor:
            return self._layer(x)

    class EncoderBlock(tf.keras.layers.Layer):

        def __init__(self,
                     size_model: int,
                     size_feedforward: int,
                     nb_heads: int,
                     *,
                     rate_dropout: float = _RATE_DROPOUT
                     ) -> None:
            super(Transformer.EncoderBlock, self).__init__()
            self._att_self = tf.keras.layers.MultiHeadAttention(
                key_dim=size_model,
                num_heads=nb_heads,
                dropout=rate_dropout
            )
            self._feedforward = Transformer.PointwiseFeedForwardLayer(
                size_model=size_model,
                size_feedforward=size_feedforward
            )
            #
            self._dropout = tf.keras.layers.Dropout(rate_dropout)
            #
            self._norm_1 = tf.keras.layers.LayerNormalization(
                epsilon=Transformer._EPSILON)
            self._norm_2 = tf.keras.layers.LayerNormalization(
                epsilon=Transformer._EPSILON)

        def build(self, shape_input: tf.TensorShape) -> None:
            # _build_from_signature has to be called with MultiHeadAttention
            # see: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
            # pylint: disable=protected-access
            self._att_self._build_from_signature(shape_input, shape_input)
            super(Transformer.EncoderBlock, self).build(shape_input)

        def call(self, x, *, training: bool, attention_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
            h_1 = self._att_self(
                x, x, attention_mask=attention_mask, training=training)
            o_1 = self._norm_1(x + h_1)
            #
            h_2 = self._feedforward(h_1)
            h_2 = self._dropout(h_2, training=training)
            o_2 = self._norm_2(o_1 + h_2)
            return o_2

    class DecoderLayer(tf.keras.layers.Layer):
        def __init__(self,
                     size_model: int,
                     size_feedforward: int,
                     nb_heads: int,
                     *,
                     rate_dropout: float = _RATE_DROPOUT
                     ) -> None:
            super(Transformer.DecoderLayer, self).__init__()
            self._att_self = tf.keras.layers.MultiHeadAttention(
                key_dim=size_model,
                num_heads=nb_heads,
                dropout=rate_dropout
            )
            self._att_cross_io = tf.keras.layers.MultiHeadAttention(
                key_dim=size_model,
                num_heads=nb_heads,
                dropout=rate_dropout
            )
            self._feedforward = Transformer.PointwiseFeedForwardLayer(
                size_model=size_model,
                size_feedforward=size_feedforward
            )
            #
            self._dropout = tf.keras.layers.Dropout(rate_dropout)
            self._norm_1 = tf.keras.layers.LayerNormalization(
                epsilon=Transformer._EPSILON)
            self._norm_2 = tf.keras.layers.LayerNormalization(
                epsilon=Transformer._EPSILON)
            self._norm_3 = tf.keras.layers.LayerNormalization(
                epsilon=Transformer._EPSILON)

        def build(self, shape_input: tf.TensorShape, shape_output_encoder: tf.TensorShape) -> None:
            # pylint: disable=protected-access
            self._att_self._build_from_signature(shape_input, shape_input)
            # pylint: disable=protected-access
            self._att_cross_io._build_from_signature(
                shape_output_encoder, shape_output_encoder, shape_input)
            super(Transformer.DecoderLayer, self).build(shape_input)

        def call(self,
                 inputs: tuple[tf.Tensor, ...],
                 *,
                 training: bool,
                 mask_look_ahead: Optional[tf.Tensor] = None,
                 mask_padding: Optional[tf.Tensor] = None
                 ) -> tf.Tensor:
            x, output_encoder = inputs
            h_1, weights_h_1 = self._att_self(
                x, x, x, attention_mask=mask_look_ahead, training=training)
            o_1 = self._norm_1(h_1 + x)
            #
            h_2, weights_h_2 = self._att_cross_io(
                output_encoder, output_encoder, o_1, attention_mask=mask_padding, training=training)
            o_2 = self._norm_2(h_2 + o_1)
            #
            h_3 = self._feedforward(o_2)
            h_3 = self._dropout(h_3, training=training)
            o_3 = self._norm_3(h_3 + o_2)
            return o_3, weights_h_1, weights_h_2

    class Encoder(tf.keras.layers.Layer):
        def __init__(self,
                     size_vocab_input: int,
                     nb_blocks: int,
                     size_model: int,
                     size_feedforward: int,
                     nb_heads: int,
                     *,
                     rate_dropout: float = _RATE_DROPOUT
                     ) -> None:
            super(Transformer.Encoder, self).__init__()
            self._size_model = size_model
            self._embedding = tf.keras.layers.Embedding(
                size_vocab_input, self._size_model)
            self._layer = tf.keras.Sequential([
                Transformer.EncoderLayer(
                    size_model=self._size_model,
                    nb_heads=nb_heads,
                    size_feedforward=size_feedforward,
                    rate_dropout=rate_dropout
                ) for _ in tf.range(nb_blocks)]
            )
            #
            self._offset_pos = Transformer.make_positional_encoding(
                Transformer._MAX_LENGTH_TOKENS, self._size_model)
            self._dropout = tf.keras.layers.Dropout(rate_dropout)

        def call(self,
                 x: tf.Tensor,
                 *,
                 training: bool,
                 mask: Optional[tf.Tensor] = None
                 ) -> tf.Tensor:
            # encode positional information
            size_seq = x.shape[-2]
            h = self._embedding(
                x) * tf.math.sqrt(tf.cast(self._size_model, tf.float32))
            h = h + self._offset_pos[..., :size_seq, :]
            #
            h = self._dropout(h, training=training)
            o = self._layer(h, training=training, mask=mask)
            return o

    class Decoder(tf.keras.layers.Layer):
        def __init__(self,
                     size_vocab_target: int,
                     nb_blocks: int,
                     size_model: int,
                     size_feedforward: int,
                     nb_heads: int,
                     *,
                     rate_dropout: float = _RATE_DROPOUT
                     ) -> None:
            super(Transformer.Decoder, self).__init__()
            self._size_model = size_model
            self._blocks_decoder = [
                Transformer.DecoderLayer(
                    size_model=self._size_model,
                    nb_heads=nb_heads,
                    size_feedforward=size_feedforward,
                    rate_dropout=rate_dropout
                ) for _ in tf.range(nb_blocks)
            ]
            #
            self._embedding = tf.keras.layers.Embedding(
                size_vocab_target, self._size_model)
            self._offset_pos = Transformer.make_positional_encoding(
                Transformer._MAX_LENGTH_TOKENS, self._size_model)
            self._dropout = tf.keras.layers.Dropout(rate_dropout)

        def call(self,
                 inputs: tuple[tf.Tensor, tf.Tensor],
                 *,
                 training: bool,
                 mask_look_ahead: Optional[tf.Tensor] = None,
                 mask_padding: Optional[tf.Tensor] = None
                 ) -> tf.Tensor:
            x, output_encoder = inputs
            # TODO: check which is the time axis!!
            size_seq = x.shape[-2]
            weights_att = {}
            #
            h = self._embedding(
                x) * tf.math.sqrt(tf.cast(self._size_model, tf.float32))
            h = h + self._pos_encoding[..., :size_seq, :]
            h = self._dropout(h, training=training)
            #
            for idx_block, block in enumerate(self._blocks_decoder):
                h, weights_block_1, weights_block_2 = block(
                    h,
                    output_encoder,
                    training=training,
                    look_ahead_mask=mask_look_ahead,
                    mask_padding=mask_padding
                )
                weights_att[idx_block] = (
                    weights_block_1, weights_block_2)
            output_decoder = h
            return output_decoder, weights_att


class CouplingSolver(tf.keras.Model):
    """Coupling solver"""

    _TOKEN_PAD = 0
    _TOKEN_START = 1
    _TOKEN_END = 2

    def __init__(self,
                 vector_quantiser: VectorQuantiser,
                 *,
                 nb_blocks: int,
                 size_model: int,
                 size_feedforward: int,
                 nb_heads: int
                 ) -> None:
        super(CouplingSolver, self).__init__()
        #
        self._vector_quantiser = vector_quantiser
        #
        self._transformers = None
        #
        self._fn_transformer = lambda: Transformer(
            size_vocab_input=self._vector_quantiser.nb_embeddings,
            size_vocab_target=self._vector_quantiser.nb_embeddings,
            nb_blocks=nb_blocks,
            size_model=size_model,
            size_feedforward=size_feedforward,
            nb_heads=nb_heads
        )

    def build(self, shapes_input: list[tuple[int, ...]]) -> None:
        nb_levels = len(shapes_input)
        self._transformers = [self._fn_transformer()
                              for _ in tf.range(nb_levels)]
        for idx_level, transcoder in enumerate(self._transformers):
            transcoder.build(shapes_input[idx_level])
        super(CouplingSolver, self).build([shapes_input, shapes_input])

    def _predict_training(self, es_a: tf.TensorArray, es_b: tf.TensorArray) -> tf.TensorArray:
        es_b_hat = tf.TensorArray(es_a.dtype, size=es_a.size())
        for idx_level, transformer in tf.range(self._transformers):
            # The 0, 1, 2 values are reserved
            # resp. for 'padding', 'start' and 'end' sequence tokens
            # so we must pass the indices starting at 3,
            # and substract it back afterwards.
            input_ = self._offset_sequence(es_a.read(idx_level))
            target = self._offset_sequence(es_b.read(idx_level))
            output, _ = transformer([input_, target], training=True)
            output = self._unoffset_sequence(output)
            es_b_hat = es_b_hat.write(idx_level, output)
        return es_b_hat

    def _predict_inference(self, es_a: tf.TensorArray) -> tf.TensorArray:
        es_b = tf.TensorArray(es_a.dtype, size=es_a.size())
        for idx_level, transformer in tf.range(self._transformers):
            input_ = self._offset_sequence(es_a.read(idx_level))
            #
            size_input = input_.shape[1]
            arr_output = tf.TensorArray(dtype=input_.dtype, size=size_input)
            # add a first 'start' token first
            arr_output = arr_output.write(0, self._TOKEN_START)
            #
            for i in tf.range(size_input):
                output = tf.transpose(arr_output[:i+1].stack())
                preds = transformer([input_, output], training=False)
                # We are actually only interested in the token that was predicted last, 
                # select the last token from the seq_len dimension
                pred_last = preds[..., -1, :]
                # pred_last: (batch_size, 1, nb_embeddings)
                id_pred = tf.argmax(pred_last, axis=-1)
                # id_pred: (batch_size, 1)
                arr_output = arr_output.write(i+1, id_pred[..., 0])
            # compute attention weights outside of loop
            # _, weights_att = transformer([input_, output], training=False, return_weights=True)
            # no need to add a final 'end' token.
            output = arr_output.stack()
            # also remove first 'start' token added.
            output = output[..., 1:, :]
            output = self._unoffset_sequence(output.read(idx_level))
            es_b_hat = es_b_hat.write(idx_level, output)
        return es_b

    def call(self,
             inputs: Union[tf.TensorArray, tuple[tf.TensorArray, tf.TensorArray]],
             training: bool
             ) -> tf.TensorArray:
        es_b_hat = None
        if training:
            es_a, es_b = inputs
            es_b_hat = self._predict_training(es_a, es_b)
        else:
            es_a = inputs
            es_b_hat = self._predict_inference(es_a)
        return es_b_hat

    @classmethod
    def _get_sequence_offset(cls) -> tf.Tensor:
        return tf.math.reduce_max([cls._TOKEN_PAD, cls._TOKEN_START, cls._TOKEN_END])

    @classmethod
    def _offset_sequence(cls, seq: tf.Tensor) -> tf.Tensor:
        return tf.math.add(seq, cls._get_sequence_offset())

    @classmethod
    def _unoffset_sequence(cls, seq: tf.Tensor) -> tf.Tensor:
        return tf.math.add(seq, cls._get_sequence_offset())
