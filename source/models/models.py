from typing import Callable

import tensorflow as tf

from models.jukebox import JukeboxAutoEncoder
from models.transformer import CouplingSolver


class SomethingModel(tf.keras.Model):
    def __init__(self,
                 auto_encoder: JukeboxAutoEncoder,
                 coupling_solver: CouplingSolver,
                 *,
                 gamma_reconstruction: float,
                 gamma_quantisation_codebook: float,
                 gamma_quantisation_commit: float,
                 ) -> None:
        super(SomethingModel, self).__init__()
        self._auto_encoder = auto_encoder
        self._coupling_solver = coupling_solver
        #
        self._vector_quantiser = auto_encoder._vector_quantiser
        self._gamma_reconstruction = gamma_reconstruction
        self._gamma_quantisation_codebook = gamma_quantisation_codebook
        self._gamma_quantisation_commit = gamma_quantisation_commit
        #
        self._trackers_loss = {
            'reconstruction': tf.keras.metrics.Mean(name='loss_reconstruction'),
            'quantisation_codebook': tf.keras.metrics.Mean(name='quantisation_codebook'),
            'quantisation_commit': tf.keras.metrics.Mean(name='quantisation_commit'),
            'coupling': tf.keras.metrics.Mean('loss_coupling')
        }

        self._caches_loss = {
            'reconstruction': None,
            'quantisation_codebook': None,
            'quantisation_commit': None,
            'coupling': None
        }

    def build(self, shape_input: tf.TensorShape) -> None:
        self._auto_encoder.build(shape_input)
        super(SomethingModel, self).build(shape_input)

    def call(self, x_a: tf.Tensor) -> tf.Tensor:
        return self.predict_inference(x_a)

    @classmethod
    def _get_merged_loss(cls, fn_loss: Callable[[tf.Tensor, tf.Tensor, Ellipsis], list[tf.Tensor]]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        def _loss(y: tf.Tensor, y_truth: tf.Tensor, *args, **kwargs) -> tf.Tensor:
            losses = fn_loss(y, y_truth, *args, **kwargs)
            return tf.reduce_sum(tf.stack([tf.reduce_mean(loss) for loss in losses], 0), 0)
        return _loss

    def predict_inference(self,
                          x_a: tf.Tensor
                          ) -> tf.Tensor:
        zs_a = self._auto_encoder.encode(x_a)
        es_a, zs_q_a, _ = self._auto_encoder.quantise(zs_a)
        #
        es_b_hat = self._coupling_solver(es_a, training=False)
        # es_b_hat = es_a
        zs_q_b_hat = self._auto_encoder.lookup_code(es_b_hat)
        xs_b_hat = self._auto_encoder.decode(zs_q_b_hat)
        #
        # LOSSES
        xs_a_hat = self._auto_encoder.decode(zs_q_a)
        loss_reconstruction_a = self._get_merged_loss(JukeboxAutoEncoder.reconstruction_loss)(
            xs_a_hat, x_a)
        loss_quantisation_codebook = self._get_merged_loss(JukeboxAutoEncoder.quantisation_codebook_loss)(
            zs_q_a, zs_a, self._auto_encoder.axes_quantisation)
        loss_quantisation_commit = self._get_merged_loss(JukeboxAutoEncoder.quantisation_commit_loss)(
            zs_q_a, zs_a, self._auto_encoder.axes_quantisation)
        #
        self._caches_loss['reconstruction'] = loss_reconstruction_a
        self._caches_loss['quantisation_codebook'] = loss_quantisation_codebook
        self._caches_loss['quantisation_commit'] = loss_quantisation_commit
        #
        for name_loss, value_loss in self._caches_loss.items():
            # update all BUT coupling (which will be None)
            if value_loss is not None:
                self._trackers_loss[name_loss].update_state(value_loss)
        # Can't use metrics.Mean object for gradient computation!!
        return [xs_b_hat, zs_q_b_hat, es_b_hat]

    def train_step(self, data: tuple[tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
        x_a, x_b = data

        with tf.GradientTape() as tape:
            # pylint: disable=not-context-manager
            with tape.stop_recording():
                # don't apply gradient on this encoding
                # it's taken care of elsewhere.
                zs_b = self._auto_encoder.encode(x_b)
                es_b, _, _ = self._auto_encoder.quantise(zs_b)
            zs_a = self._auto_encoder.encode(x_a)
            es_a, zs_q_a, _ = self._auto_encoder.quantise(zs_a)
            #
            es_b_hat = self._coupling_solver(es_a, es_b, training=True)
            # es_b_hat = es_a
            zs_q_b_hat = self._auto_encoder.lookup_code(es_b_hat)
            xs_b_hat = self._auto_encoder.decode(zs_q_b_hat)
            #
            # LOSSES
            xs_a_hat = self._auto_encoder.decode(zs_q_a)
            loss_reconstruction_a = self._get_merged_loss(JukeboxAutoEncoder.reconstruction_loss)(
                xs_a_hat, x_a)
            loss_quantisation_codebook = self._get_merged_loss(JukeboxAutoEncoder.quantisation_codebook_loss)(
                zs_q_a, zs_a, self._auto_encoder.axes_quantisation)
            loss_quantisation_commit = self._get_merged_loss(JukeboxAutoEncoder.quantisation_commit_loss)(
                zs_q_a, zs_a, self._auto_encoder.axes_quantisation)
            loss_coupling_b = self._(JukeboxAutoEncoder.codebook_dissimilarity_loss)(
                es_b_hat, es_b, self._auto_encoder.nb_embeddings)
            #
            self._caches_loss['reconstruction'] = loss_reconstruction_a
            self._caches_loss['quantisation_codebook'] = loss_quantisation_codebook
            self._caches_loss['quantisation_commit'] = loss_quantisation_commit
            self._caches_loss['coupling'] = loss_coupling_b
            #
            # Can't use metrics.Mean object for gradient computation!!
            loss_reconstruction_a = self._caches_loss['reconstruction']
            loss_quantisation_codebook = self._caches_loss['quantisation_codebook']
            loss_quantisation_commit = self._caches_loss['quantisation_commit']
            loss_coupling_b = self._caches_loss['coupling']
            # self.add_loss(self._gamma_reconstruction * loss_reconstruction_a)
            # self.add_loss(self._gamma_quantisation_codebook * loss_quantisation_codebook)
            # self.add_loss(self._gamma_quantisation_commit * loss_quantisation_commit)
            loss = loss_coupling_b + \
                self._gamma_reconstruction * loss_reconstruction_a + \
                self._gamma_quantisation_codebook * loss_quantisation_codebook + \
                self._gamma_quantisation_commit * loss_quantisation_commit

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Reset stored losses
        for key, _ in self._caches_loss:
            self._caches_loss[key] = None

        # Update metrics
        self.compiled_metrics.update_state(x_b, xs_b_hat)
        # Compute our own metrics
        return {m.name: m.result() for m in self.metrics}

    def reset_metrics(self):
        for tracker in self._trackers_loss.values():
            tracker.reset_states()

    @ property
    def metrics(self):
        return self._trackers_loss.values()
