from typing import Tuple, Dict

import tensorflow as tf
import tensorflow_io as tfio

import librosa


def _compute_annotation_spectrum(x_ann: tf.Tensor, size_win: int, stride_win: int) -> tf.Tensor:
    """Has to be `run_eagerly`'d, because it uses librosa functions.
    """
    x_ann_spec = tf.convert_to_tensor(librosa.util.frame(
        x_ann.numpy(),
        frame_length=size_win,
        hop_length=stride_win,
        axis=0
    ), dtype=x_ann.dtype)
    return x_ann_spec

def preprocess_audio(
        x_audio: tf.Tensor,
        x_ann: Dict[str, tf.Tensor],
        config: object
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """_summary_

    Args:
        x_audio (tf.Tensor): _description_
        x_ann (Dict[str, tf.Tensor]): _description_
        config (object): _description_

    Returns:
        Tuple[tf.Tensor, Dict[str, tf.Tensor]]: _description_
    """
    #
    x_ann['dir_play'] = _compute_change_play(
        x_ann['dir_play'], config.size_kernel, config.sigma)
    #
    x_spec, x_ann_spec = _compute_spectrum(x_audio, x_ann,
                                                config.nb_freqs,
                                                config.size_win,
                                                config.stride_win
                                                )

    x_mel = tf.expand_dims(
        tfio.audio.melscale(
            tf.squeeze(x_spec, axis=-1),
            fmin=config.freq_min,
            fmax=config.freq_max,
            rate=config.rate_sample,
            mels=config.nb_mel_freqs
        ),
        axis=-1
    )
    #
    return x_mel, x_ann_spec

def _compute_spectrum(
        x_audio: tf.Tensor,
        x_ann: Dict[str, tf.Tensor],
        nb_freqs: int,
        size_win: int,
        stride_win: int
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
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
    x_spec = tf.expand_dims(
        tfio.audio.spectrogram(
            tf.squeeze(x_audio, axis=-1),
            nfft=nb_freqs,
            window=size_win,
            stride=stride_win
        ),
        axis=-1
    )
    x_ann_spec = dict()
    # Important:
    # Tensors inside dicts are not Eager tensors,
    # so we have to use py_function
    # see: https://github.com/tensorflow/tensorflow/issues/32842#issuecomment-536649134
    x_ann_spec['x_play'] = tf.py_function(_compute_annotation_spectrum, [
        x_ann['x_play'], size_win, stride_win], Tout=x_ann['x_play'].dtype)
    x_ann_spec['y_play'] = tf.py_function(_compute_annotation_spectrum, [
        x_ann['y_play'], size_win, stride_win], Tout=x_ann['y_play'].dtype)
    x_ann_spec['dir_play'] = tf.py_function(_compute_annotation_spectrum, [
        x_ann['dir_play'], size_win, stride_win], Tout=x_ann['dir_play'].dtype)

    return (x_spec, x_ann_spec)

def _compute_change_play(x_dir_play: tf.Tensor, size_kernel: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """_summary_

    Args:
        x_ann_dir_play (tf.Tensor): _description_

    Returns:
        tf.Tensor: _description_
    """
    x_change_play = x_dir_play - tf.roll(x_dir_play, shift=1, axis=-2)
    x_change_play = tf.cast(tf.abs(x_change_play), tf.float32)
    # gaussian smoothing
    x_change_play = tf.squeeze(tfio.experimental.filter.gaussian(
        tf.reshape(x_change_play, [1, 1, *x_change_play.get_shape().as_list()]), ksize=size_kernel, sigma=sigma), axis=(0, 1))
    return x_change_play
