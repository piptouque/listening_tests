
import tensorflow as tf
import tensorflow_io as tfio

import librosa

from typing import Tuple, Dict


class AudioPreprocessor(tf.keras.Model):
    """Has to be `run_eagerly`'d, because it uses librosa functions.
    """

    def __init__(self, config: object):
        super(AudioPreprocessor, self).__init__()
        self._config = config.audio

    @staticmethod
    def _compute_annotation_spectrum(x_ann: tf.Tensor, size_win: int, stride_win: int) -> tf.Tensor:
        x_ann_spec = tf.convert_to_tensor(librosa.util.frame(
            x_ann.numpy(),
            frame_length=size_win,
            hop_length=stride_win,
            axis=0
        ), dtype=x_ann.dtype)
        return x_ann_spec

    @classmethod
    def compute_spectrum(
            cls,
            x_audio: tf.Tensor,
            x_ann: Dict[str, tf.Tensor],
            nb_freqs: int,
            size_win: int,
            stride_win: int) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """_summary_

        Args:
            x_audio (tf.Tensor): _description_
            x_ann (Dict[str, tf.Tensor]): _description_
            nb_freqs (int): _description_
            size_win (int): _description_
            stride_win (int): _description_

        Returns:
            Tuple[tf.Tensor, Dict[str, tf.Tensor]]: _description_
        """
        #
        # Resample the annotations to the audio sample rate
        print(x_audio.shape)
        x_spec = tf.expand_dims(tfio.audio.spectrogram(
            tf.squeeze(x_audio, axis=-1),
            nfft=nb_freqs,
            window=size_win,
            stride=stride_win),
            axis=-1
        )
        print(x_spec.shape)
        x_ann_spec = dict()
        # Important:
        # Tensors inside dicts are not Eager tensors,
        # so we have to use py_function
        # see: https://github.com/tensorflow/tensorflow/issues/32842#issuecomment-536649134
        x_ann_spec['x_play'] = tf.py_function(cls._compute_annotation_spectrum, [
                                              x_ann['x_play'], size_win, stride_win], Tout=x_ann['x_play'].dtype)
        x_ann_spec['y_play'] = tf.py_function(cls._compute_annotation_spectrum, [
                                              x_ann['y_play'], size_win, stride_win], Tout=x_ann['y_play'].dtype)
        x_ann_spec['dir_play'] = tf.py_function(cls._compute_annotation_spectrum, [
                                                x_ann['dir_play'], size_win, stride_win], Tout=x_ann['dir_play'].dtype)
        print(x_ann_spec['x_play'].shape)

        return (x_spec, x_ann_spec)

    def call(self, x_audio: tf.Tensor, x_ann: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """_summary_

        Args:
            x_audio (tf.Tensor): _description_
            x_ann (Dict[str, tf.Tensor]): _description_

        Returns:
            Tuple[tf.Tensor, Dict[str, tf.Tensor]]: _description_
        """
        return self.compute_spectrum(x_audio, x_ann, self._config.nb_freqs, self._config.size_win, self._config.stride_win)
