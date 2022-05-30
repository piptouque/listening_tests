from typing import Tuple, Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

import librosa

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
        x_ann['dir_play'], config.annotations.size_kernel, config.annotations.sigma)
    #
    x_spec, x_ann_spec = _compute_spectrum(x_audio, x_ann,
                                           config.audio.freq.nb_freqs,
                                           config.audio.time.size_win,
                                           config.audio.time.stride_win)

    x_mel = tf.stack([
        tfio.audio.melscale(
            el_spec,
            fmin=config.audio.freq.freq_min,
            fmax=config.audio.freq.freq_max,
            rate=config.audio.time.rate_sample,
            mels=config.audio.freq.nb_freqs_mel
        ) for el_spec in tf.unstack(x_spec, axis=-1)],
        axis=-1
    )
    #
    return x_mel, x_ann_spec

def preprocess_audio_features(x_audio: tf.Tensor, config: object) -> tf.Tensor:
    """
    """
    x_features = _compute_audio_features(x_audio,
        rate_sample=config.audio.time.rate_sample,
        size_win=config.audio.time.size_win,
        stride_win=config.audio.time.stride_win,
        freq_min=config.audio.freq.freq_min,
        freq_max=config.audio.freq.freq_max,
        nb_mfcc=config.audio.freq.nb_mfcc
    )
    _NAMES_FEATURES_USED = [
        'freq_0', 'prob_freq',
        'root_mean_square',
        'spectral_flatness', 'spectral_centroid'
    ]
    x_features_used = [x_features[name_feature_used] for name_feature_used in _NAMES_FEATURES_USED]
    return tf.stack(x_features_used, axis=0) 

def _compute_audio_features(
    x_audio: tf.Tensor,
    rate_sample: int,
    size_win: int,
    stride_win: int,
    freq_min: float,
    freq_max: float,
    nb_mfcc: int
) -> Dict[str, tf.Tensor]:
    _NAMES_FEATURE = [
        'mfcc', 'freq_0', 'prob_freq',
        'root_mean_square', 'zero_crossing_rate',
        'spectral_flatness', 'spectral_centroid'
    ]
    # our pre-processing assumes channel-first (after batch)
    # from [batch, ..., channel] to [batch, channel, ...]
    perm = [] # TODO
    perm_inv = [] # TODO
    x_features = tf.numpy_function(
        _compute_audio_features_numpy,
        [
            tf.transpose(x_audio, perm=perm),
            rate_sample,
            size_win,
            stride_win,
            freq_min,
            freq_max,
            nb_mfcc
        ],
        Tout=[x_audio.dtype] * len(_NAMES_FEATURE)
    )
    x_features = [tf.transpose(x_feature, perm=perm_inv) for x_feature in x_features]
    return dict(zip(_NAMES_FEATURE, x_features))


def _compute_audio_features_numpy(
    x_audio: np.ndarray,
    rate_sample: int,
    size_win: int,
    stride_win: int,
    freq_min: float,
    freq_max: float,
    nb_mfcc: int
) -> List[np.ndarray]:

    nb_freqs = size_win
    x_spec = librosa.stft(
            y=x_audio,
            n_fft=nb_freqs,
            win_length=size_win,
            hop_length=stride_win
    )

    x_mel = librosa.feature.melspectrogram(
        S=np.abs(x_spec)
    ).astype(x_audio.dtype)

    x_mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(np.abs(x_mel)**2),
        n_mfcc=nb_mfcc
    ).astype(x_audio.dtype)

    x_freq_0, _, x_prob_freq = librosa.pyin(
        y=x_audio,
        fmin=freq_min,
        fmax=freq_max,
        sr=rate_sample,
        frame_length=size_win,
        hop_length=stride_win
    )
    x_freq_0 = x_freq_0.astype(x_audio.dtype)
    x_prob_freq = x_prob_freq.astype(x_audio.dtype)

    x_rms = librosa.feature.rms(
        S=np.abs(x_spec),
        frame_length=size_win,
        hop_length=stride_win
    ).astype(x_audio.dtype)

    x_zcr = librosa.feature.zero_crossing_rate(
        y=x_audio,
        frame_length=size_win,
        hop_length=size_win
    ).astype(x_audio.dtype)

    x_flatness = librosa.feature.spectral_flatness(
        S=np.abs(x_spec),
    ).astype(x_audio.dtype)
    
    x_centroid = librosa.feature.spectral_centroid(
        S=np.abs(x_spec)
    ).astype(x_audio.dtype)

    return [
        x_mfcc,
        x_freq_0,
        x_prob_freq,
        x_rms,
        x_zcr,
        x_flatness,
        x_centroid
    ]
       

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
    x_spec = tf.stack([
        tfio.audio.spectrogram(
            el_audio,
            nfft=nb_freqs,
            window=size_win,
            stride=stride_win
        ) for el_audio in tf.unstack(x_audio, axis=-1)],
        axis=-1
    )
    #
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
