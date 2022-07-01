#
from typing import Union

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
        self.tracker_loss_coupling = tf.keras.metrics.Mean(
            name='loss_coupling')
        self.tracker_loss_reconstruction = tf.keras.metrics.Mean(
            name='loss_reconstruction')
        self.tracker_loss_quantisation_codebook = tf.keras.metrics.Mean(
            name='loss_quantisation_codebook')
        self.tracker_loss_quantisation_commit = tf.keras.metrics.Mean(
            name='loss_quantisation_commit')

        self.cache_losses = {
            'reconstruction': None,
            'quantisation_codebook': None,
            'quantisation_commit': None
        }

    def build(self, shape_input: tf.TensorShape) -> None:
        self._auto_encoder.build(shape_input)
        super(SomethingModel, self).build(shape_input)

    def predict_inference(self,
                          # inputs: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
                          x_a: tf.Tensor
                          ) -> tf.Tensor:
        zs_a = self._auto_encoder.encode(x_a)
        es_a, zs_q_a, _ = self._auto_encoder.quantise(zs_a)
        #
        es_b_hat = self._coupling_solver.call(es_a)
        zs_q_b_hat = self._auto_encoder.lookup_code(es_b_hat)
        xs_b_hat = self._auto_encoder.decode(zs_q_b_hat)
        #
        # LOSSES
        xs_a_hat = self._auto_encoder.decode(zs_q_a)
        loss_reconstruction_a = JukeboxAutoEncoder.get_reconstruction_loss(
            xs_a_hat, x_a)
        loss_quantisation_codebook = JukeboxAutoEncoder.get_quantisation_codebook_loss(
            zs_q_a, zs_a, self._auto_encoder.axes_quantisation)
        loss_quantisation_commit = JukeboxAutoEncoder.get_quantisation_commit_loss(
            zs_q_a, zs_a, self._auto_encoder.axes_quantisation)
        #
        self.cache_losses['reconstruction'] = loss_reconstruction_a
        self.cache_losses['quantisation_codebook'] = loss_quantisation_codebook
        self.cache_losses['quantisation_commit'] = loss_quantisation_commit
        #
        # Can't use metrics.Mean object for gradient computation!!
        self.tracker_loss_reconstruction.update_state(loss_reconstruction_a)
        self.tracker_loss_quantisation_codebook.update_state(
            loss_quantisation_codebook)
        self.tracker_loss_quantisation_commit.update_state(
            loss_quantisation_commit)
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
            zs_q_b_hat = self._auto_encoder.lookup_code(es_b_hat)
            xs_b_hat = self._auto_encoder.decode(zs_q_b_hat)
            #
            # LOSSES
            xs_a_hat = self._auto_encoder.decode(zs_q_a)
            loss_reconstruction_a = JukeboxAutoEncoder.get_reconstruction_loss(
                xs_a_hat, x_a)
            loss_quantisation_codebook = JukeboxAutoEncoder.get_quantisation_codebook_loss(
                zs_q_a, zs_a, self._auto_encoder.axes_quantisation)
            loss_quantisation_commit = JukeboxAutoEncoder.get_quantisation_commit_loss(
                zs_q_a, zs_a, self._auto_encoder.axes_quantisation)
            #
            self.cache_losses['reconstruction'] = loss_reconstruction_a
            self.cache_losses['quantisation_codebook'] = loss_quantisation_codebook
            self.cache_losses['quantisation_commit'] = loss_quantisation_commit
            #
            # Can't use metrics.Mean object for gradient computation!!
            self.tracker_loss_reconstruction.update_state(
                loss_reconstruction_a)
            self.tracker_loss_quantisation_codebook.update_state(
                loss_quantisation_codebook)
            self.tracker_loss_quantisation_commit.update_state(
                loss_quantisation_commit)
            #
            # Coupling loss!
            loss_coupling_b = JukeboxAutoEncoder.get_codebook_dissimilarity_loss(
                es_b_hat, es_b, self._auto_encoder.nb_embeddings)
            self.tracker_loss_coupling.update_state(loss_coupling_b)
            #
            loss_reconstruction_a = self.cache_losses['reconstruction']
            loss_quantisation_codebook = self.cache_losses['quantisation_codebook']
            loss_quantisation_commit = self.cache_losses['quantisation_commit']
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
        for key, _ in self.cache_losses:
            self.cache_losses[key] = None

        # Update metrics
        self.compiled_metrics.update_state(x_b, xs_b_hat)
        # Compute our own metrics
        return {m.name: m.result() for m in self.metrics}

    def reset_metrics(self):
        self.tracker_loss_coupling.reset_states()
        self.tracker_loss_reconstruction.reset_states()
        self.tracker_loss_quantisation_codebook.reset_states()
        self.tracker_loss_quantisation_commit.reset_states()

    @property
    def metrics(self):
        return [
            self.tracker_loss_coupling,
            self.tracker_loss_reconstruction,
            self.tracker_loss_quantisation_codebook,
            self.tracker_loss_quantisation_commit
        ]
