from typing import Optional, Union, Callable

import tensorflow as tf
import tensorflow_probability as tfp

from models.common import VectorQuantiser


_RATE_DROPOUT = 0.1


class Transformer(tf.keras.Model):
    _MAX_LENGTH_TOKENS = 512
    _EPSILON = 1e-6

    def __init__(self,
                 size_vocab_input: int,
                 size_vocab_target: int,
                 nb_blocks: int,
                 size_model: int,
                 size_feedforward: int,
                 nb_heads: int,
                 *,
                 rate_dropout=_RATE_DROPOUT) -> None:
        super(Transformer, self).__init__()
        self._embedding_encoder = Transformer.PositionalEmbedding(
            size_vocab=size_vocab_input,
            size_model=size_model
        )
        self._embedding_decoder = Transformer.PositionalEmbedding(
            size_vocab=size_vocab_target,
            size_model=size_model
        )
        self._encoder = Transformer.Encoder(
            nb_blocks=nb_blocks,
            size_model=size_model,
            size_feedforward=size_feedforward,
            nb_heads=nb_heads,
            rate_dropout=rate_dropout
        )
        self._decoder = Transformer.Decoder(
            nb_blocks=nb_blocks,
            size_model=size_model,
            size_feedforward=size_feedforward,
            nb_heads=nb_heads,
            rate_dropout=rate_dropout
        )
        self._layer_fit_output = tf.keras.Sequential()
        if size_vocab_input != size_vocab_target:
            self._layer_fit_output.add(
                tf.keras.layers.Permute((-2, -1)))
            self._layer_fit_output.add(
                tf.keras.layers.Dense(size_vocab_target))

            self._layer_fit_output.add(
                tf.keras.layers.Permute((-2, -1))
            )
        else:
            self._layer_fit_output = tf.keras.layers.Lambda(lambda x: x)


    def call(self,
             inputs: tuple[tf.Tensor, tf.Tensor],
             *,
             training: bool,
             return_weights: bool = False,
             ) -> tf.Tensor:
        """_summary_

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]):
                - [0]: Input tokens (batch, time)
                - [1]: Target tokens (batch, time)
            return_weights (bool, optional): _description_. Defaults to False.

        Returns:
            tf.Tensor: Predictions (batch, time, size_vocab_target)
        """
        input_, target = inputs
        mask_lin_padding, mask_look_ahead = self._make_masks(input_, target)
        #
        input_ = self._embedding_encoder(input_)
        target = self._embedding_decoder(target)
        # input_, target: (batch, size_seq_{input, target}, size_model)
        size_seq_input = input_.shape[-2]
        size_seq_target = target.shape[-2]
        #
        mask_padding_encoder = tf.stack([mask_lin_padding] * size_seq_input, -2)
        mask_padding_decoder = tf.stack([mask_lin_padding] * size_seq_target, -2)
        #
        output_encoder = self._encoder(
            input_,
            mask_padding=tf.cast(mask_padding_encoder, input_.dtype),
            training=training
        )
        output_decoder, weights_att = self._decoder(
            [target, output_encoder],
            mask_look_ahead=tf.cast(mask_look_ahead, target.dtype),
            mask_padding=tf.cast(mask_padding_decoder, target.dtype),
            training=training

        )
        output = self._layer_fit_output(output_decoder)
        if return_weights:
            return output, weights_att
        else:
            return output

    @staticmethod
    def get_masked_loss(fn_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], *, value_mask: int = 0) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        # mask the padding token (0)
        def _loss_masked(y: tf.Tensor, y_truth: tf.Tensor) -> tf.Tensor:
            mask = tf.math.logical_not(tf.math.equal(y_truth, value_mask))
            loss_ = fn_loss(y, y_truth)
            loss_ *= tf.cast(mask, loss_.dtype)
            return loss_ / tf.reduce_sum(mask)
        return _loss_masked

    @staticmethod
    def get_masked_loss_sparse(fn_loss_sparse: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], *, value_mask: int = 0) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        # mask the padding token (0)
        def _loss_masked(y: tf.Tensor, y_truth: tf.Tensor) -> tf.Tensor:
            y_truth_val = tf.argmax(y_truth, axis=-1)
            mask = tf.math.logical_not(tf.math.equal(y_truth_val, value_mask))
            loss_ = fn_loss_sparse(y, y_truth)
            loss_ *= tf.cast(mask, loss_.dtype)
            return loss_ / tf.reduce_sum(tf.cast(mask, loss_.dtype))
        return _loss_masked

    @staticmethod
    def get_masked_accuracy(fn_accuracy: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], *, value_mask: int = 0) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        def _accuracy_masked(y: tf.Tensor, y_truth: tf.Tensor) -> tf.Tensor:
            acc_ = fn_accuracy(y, y_truth)
            mask = tf.math.logical_not(tf.math.equal(y_truth, value_mask))
            acc_ = tf.math.logical_and(acc_, mask)
            acc_ = tf.cast(acc_, tf.float32)
            mask = tf.cast(mask, tf.float32)
            return acc_ / tf.reduce_sum(mask)
        return _accuracy_masked

    @staticmethod
    def _make_padding_mask_lin(input_: tf.Tensor) -> tf.Tensor:
        """_summary_

        Args:
            input_ (tf.Tensor): _description_

        Returns:
            tf.Tensor: (batch, size_seq)
        """
        mask_padding = tf.math.not_equal(input_, 0)
        return mask_padding

    @staticmethod
    def _make_look_ahead_mask(size_seq: int) -> tf.Tensor:
        """_summary_

        Args:
            size_seq (int): _description_

        Returns:
            tf.Tensor: (1, size_seq, size_seq)
        """
        mask_look_ahead = tfp.math.fill_triangular(
            tf.ones((size_seq * (size_seq+1)) // 2, dtype=tf.bool), upper=False)
        # add batch dim to look-ahead mask
        mask_look_ahead = tf.expand_dims(mask_look_ahead, 0)
        return mask_look_ahead

    @classmethod
    def _make_masks(cls, input_: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # tf.assert_equal(input_.shape[-1], target.shape[-1])
        mask_lin_padding = cls._make_padding_mask_lin(input_)
        #
        size_seq_target = target.shape[-1]
        mask_look_ahead = cls._make_look_ahead_mask(size_seq_target)
        mask_lin_padding_target = cls._make_padding_mask_lin(target)
        mask_look_ahead = tf.logical_and(
            mask_look_ahead,
            tf.stack([mask_lin_padding_target] * size_seq_target, -1))
        # broadcast last dim of padding mask to input sequence dim
        return mask_lin_padding, mask_look_ahead

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

        def call(self, x: tf.Tensor, *, training: bool, mask_padding: Optional[tf.Tensor] = None) -> tf.Tensor:
            h_1 = self._att_self(
                x, x, attention_mask=mask_padding, training=training)
            o_1 = self._norm_1(x + h_1)
            #
            h_2 = self._feedforward(h_1)
            h_2 = self._dropout(h_2, training=training)
            o_2 = self._norm_2(o_1 + h_2)
            return o_2

    class DecoderBlock(tf.keras.layers.Layer):
        def __init__(self,
                     size_model: int,
                     size_feedforward: int,
                     nb_heads: int,
                     *,
                     rate_dropout: float = _RATE_DROPOUT
                     ) -> None:
            super(Transformer.DecoderBlock, self).__init__()
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

        def build(self, shapes_input: tuple[tf.TensorShape, tf.TensorShape]) -> None:
            shape_input, shape_output_encoder = shapes_input
            # pylint: disable=protected-access
            self._att_self._build_from_signature(shape_input, shape_input)
            # pylint: disable=protected-access
            self._att_cross_io._build_from_signature(
                shape_input, shape_output_encoder)
            super(Transformer.DecoderBlock, self).build(shape_input)

        def call(self,
                 inputs: tuple[tf.Tensor, tf.Tensor],
                 *,
                 training: bool,
                 mask_look_ahead: Optional[tf.Tensor] = None,
                 mask_padding: Optional[tf.Tensor] = None
                 ) -> tf.Tensor:
            x, output_encoder = inputs
            h_1, weights_h_1 = self._att_self(
                x, x,
                attention_mask=mask_look_ahead, training=training,
                return_attention_scores=True
            )
            o_1 = self._norm_1(h_1 + x)
            #
            h_2, weights_h_2 = self._att_cross_io(
                o_1, output_encoder,
                attention_mask=mask_padding, training=training,
                return_attention_scores=True)
            o_2 = self._norm_2(h_2 + o_1)
            #
            h_3 = self._feedforward(o_2)
            h_3 = self._dropout(h_3, training=training)
            o_3 = self._norm_3(h_3 + o_2)
            return o_3, weights_h_1, weights_h_2

    class PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, size_vocab: int, size_model: int) -> None:
            super(Transformer.PositionalEmbedding, self).__init__()
            self._size_model = size_model
            self._embedding = tf.keras.layers.Embedding(
                size_vocab, self._size_model
            )
            self._offset_pos = self.make_positional_encoding(
                Transformer._MAX_LENGTH_TOKENS, self._size_model)

        def call(self, x: tf.Tensor) -> tf.Tensor:
            """encode positional information
            Args:
                x (tf.Tensor): _description_

            Returns:
                tf.Tensor: _description_
            """
            size_seq = x.shape[-1]
            # pylint: disable=protected-access
            tf.assert_less(size_seq+1, Transformer._MAX_LENGTH_TOKENS,
                           "Max sequence length exceeded.")
            h = self._embedding(
                x) * tf.math.sqrt(tf.cast(self._size_model, tf.float32))
            o = h + self._offset_pos[..., :size_seq, :]
            return o

        @ staticmethod
        def make_positional_encoding(nb_embeddings: int, size_model: int) -> tf.Tensor:
            pos = tf.expand_dims(tf.range(nb_embeddings), -1)
            ids_pos = tf.expand_dims(tf.range(size_model), 0)
            angles = pos / tf.math.pow(
                10_000, (2 * (ids_pos//2)) // size_model
            )
            pred_even = tf.expand_dims(tf.math.equal(
                tf.range(angles.shape[0]) % 2, 0), -1)
            angles = tf.where(pred_even,
                              x=tf.cos(angles), y=tf.sin(angles))
            encoding_pos = tf.cast(tf.expand_dims(angles, 0), tf.float32)
            return encoding_pos

    class Encoder(tf.keras.layers.Layer):
        def __init__(self,
                     nb_blocks: int,
                     size_model: int,
                     size_feedforward: int,
                     nb_heads: int,
                     *,
                     rate_dropout: float = _RATE_DROPOUT
                     ) -> None:
            super(Transformer.Encoder, self).__init__()
            self._blocks = [Transformer.EncoderBlock(
                size_model=size_model,
                nb_heads=nb_heads,
                size_feedforward=size_feedforward,
                rate_dropout=rate_dropout
            ) for _ in range(nb_blocks)
            ]
            #
            self._dropout = tf.keras.layers.Dropout(rate_dropout)

        def call(self,
                 x: tf.Tensor,
                 *,
                 training: bool,
                 mask_padding: Optional[tf.Tensor] = None
                 ) -> tf.Tensor:
            h = self._dropout(x, training=training)
            for block in self._blocks:
                h = block(h, training=training, mask_padding=mask_padding)
            return h

    class Decoder(tf.keras.layers.Layer):
        def __init__(self,
                     nb_blocks: int,
                     size_model: int,
                     size_feedforward: int,
                     nb_heads: int,
                     *,
                     rate_dropout: float = _RATE_DROPOUT
                     ) -> None:
            super(Transformer.Decoder, self).__init__()
            self._blocks_decoder = [
                Transformer.DecoderBlock(
                    size_model=size_model,
                    nb_heads=nb_heads,
                    size_feedforward=size_feedforward,
                    rate_dropout=rate_dropout
                ) for _ in range(nb_blocks)
            ]
            #
            self._dropout = tf.keras.layers.Dropout(rate_dropout)

        def call(self,
                 inputs: tuple[tf.Tensor, tf.Tensor],
                 *,
                 training: bool,
                 mask_look_ahead: Optional[tf.Tensor] = None,
                 mask_padding: Optional[tf.Tensor] = None
                 ) -> tf.Tensor:
            x, output_encoder = inputs
            weights_att = {}
            #
            h = self._dropout(x, training=training)
            #
            for idx_block, block in enumerate(self._blocks_decoder):
                h, weights_block_1, weights_block_2 = block(
                    [h, output_encoder],
                    training=training,
                    mask_look_ahead=mask_look_ahead,
                    mask_padding=mask_padding
                )
                weights_att[idx_block] = (
                    weights_block_1, weights_block_2)
            output_decoder = h
            return output_decoder, weights_att


class CouplingSolver(tf.keras.Model):

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
        self._fn_transformer = lambda: Transformer(
            size_vocab_input=self._vector_quantiser.nb_embeddings,
            size_vocab_target=self._vector_quantiser.nb_embeddings,
            nb_blocks=nb_blocks,
            size_model=size_model,
            size_feedforward=size_feedforward,
            nb_heads=nb_heads
        )
        self._transformer = self._fn_transformer()

    def build(self, shapes_input: list[tuple[int, ...]]) -> None:
        nb_levels = len(shapes_input)
        # Can't use build() with integer-typed tensors.
        # Have to use call() with placeholder data
        # Have to fully define them and set the batch dim
        placeholders = [tf.keras.Input(shape[1:], dtype=tf.int64)
                        for shape in shapes_input]
        for idx_level in range(nb_levels):
            input_ = self.preprocess_input(placeholders[idx_level])
        _ = self._transformer([input_, input_])

    def _predict_training(self, es_a: list[tf.Tensor], es_b: list[tf.Tensor]) -> list[tf.Tensor]:
        preds_output_level = [None] * len(es_b)
        for idx_level, e_a in enumerate(es_a):
            # The 0, 1, 2 values are reserved
            # resp. for 'padding', 'start' and 'end' sequence tokens
            # so we must pass the indices starting at 3,
            # and substract it back afterwards.
            input_ = self.preprocess_input(e_a)
            target = self.preprocess_input(es_b[idx_level])
            target_roll = target[..., :-1]
            input_roll = input_[..., 1:]
            preds_output = self._transformer([input_roll, target_roll], training=True)
            # preds_output: (batch, time, nb_embeddings)
            # output = tf.argmax(preds_output, axis=-1)
            # add back the first prediction (ignored)
            target_first = tf.one_hot(tf.expand_dims(target[..., 0], -1), depth=preds_output.shape[-1], axis=-1)
            preds_output = tf.concat([target_first, preds_output], -2)
            # output = self._postprocess_output(output, shape=e_a.shape)
            preds_output_level[idx_level] = preds_output
        return preds_output_level

    def _predict_inference(self, es_a: list[tf.Tensor]) -> list[tf.Tensor]:
        es_b_hat = [None] * len(es_a)
        for idx_level, e_a in enumerate(es_a):
            input_ = self.preprocess_input(e_a)
            #
            size_seq = input_.shape[-1]
            size_batch = input_.shape[0]
            arr_output = tf.TensorArray(dtype=input_.dtype, size=size_seq)
            # add a first 'start' token first
            output_first = tf.expand_dims(tf.convert_to_tensor(self._TOKEN_START), 0)
            if input_.shape.is_fully_defined():
                output_first = tf.broadcast_to(output_first, [input_.shape[0]])
            output_first = tf.cast(output_first, input_.dtype)
            arr_output = arr_output.write(0, output_first)
            #
            for idx_token in range(size_seq):
                preds_output = tf.transpose(arr_output.stack()[:idx_token+1])
                preds = self._transformer([input_, preds_output], training=False)
                # We are actually only interested in the token that was predicted last, 
                # select the last token from the seq_len dimension
                pred_last = preds[..., -1, :]
                # pred_last: (batch_size, nb_embeddings)
                id_pred = tf.argmax(pred_last, axis=-1)
                # id_pred: (batch_size,)
                arr_output = arr_output.write(idx_token+1, id_pred)
            # compute attention weights outside of loop
            # _, weights_att = transformer([input_, output], training=False, return_weights=True)
            # no need to add a final 'end' token.
            preds_output = tf.transpose(arr_output.stack())
            # also remove first 'start' token added.
            preds_output = preds_output[..., 1:, :]
            e_b_hat = self.postprocess_output(preds_output, e_a.shape)
            es_b_hat[idx_level] = e_b_hat
        return es_b_hat

    def call(self,
             inputs: Union[list[tf.Tensor], tuple[list[tf.Tensor], list[tf.Tensor]]],
             training: bool) -> list[tf.Tensor]:
        es_b_hat = None
        if training:
            es_a, es_b = inputs
            es_b_hat = self._predict_training(es_a, es_b)
        else:
            es_a = inputs
            es_b_hat = self._predict_inference(es_a)
        return es_b_hat

    @classmethod
    def _transcoding_loss(cls, target: tf.Tensor, preds_output: tf.Tensor) -> tf.Tensor:
        return tf.keras.metrics.sparse_categorical_crossentropy(target, preds_output, from_logits=False, axis=-1)

    @classmethod
    def _masked_transcoding_loss(cls, target: tf.Tensor, preds_output: tf.Tensor) -> tf.Tensor:
        fn_loss = cls._transcoding_loss
        fn_loss = Transformer.get_masked_loss_sparse(fn_loss, value_mask=cls._TOKEN_PAD)
        return fn_loss(target, preds_output)

    @classmethod
    def coupling_loss(cls, targets: list[tf.Tensor], preds_output_level: list[tf.Tensor]) -> list[tf.Tensor]:
        return [cls._masked_transcoding_loss(targets[idx_level], preds_output) for idx_level, preds_output in enumerate(preds_output_level)]

    @ classmethod
    def preprocess_input(cls, e: tf.Tensor) -> tf.Tensor:
        input_ = cls._offset_sequence(e)
        # flatten all but batch
        nb_els = tf.math.reduce_prod(tf.convert_to_tensor(input_.shape[1:]))
        input_ = tf.reshape(input_, [-1, nb_els])
        return input_

    @ classmethod
    def postprocess_output(cls, output: tf.Tensor, shape: tuple[int]) -> tf.Tensor:
        # infer batch dim
        shape_reshape = [-1, *shape[1:]]
        # revert flatten
        e = tf.reshape(output, shape_reshape)
        e = cls._unoffset_sequence(e)
        return e

    @ classmethod
    def _get_sequence_offset(cls) -> tf.Tensor:
        return tf.cast(tf.math.reduce_max([cls._TOKEN_PAD, cls._TOKEN_START, cls._TOKEN_END]), tf.int32)

    @ classmethod
    def _offset_sequence(cls, seq: tf.Tensor) -> tf.Tensor:
        return seq + tf.cast(cls._get_sequence_offset(), seq.dtype)

    @ classmethod
    def _unoffset_sequence(cls, seq: tf.Tensor) -> tf.Tensor:
        return seq - tf.cast(cls._get_sequence_offset(), seq.dtype)
