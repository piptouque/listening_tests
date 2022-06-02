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

class AudioFeatureExtractor(tf.keras.layers.Layer):
    """Simple audio feature extractor"""
    _NAMES_FEATURES_USED = [
        'freq_0', 'prob_freq',
        'root_mean_square',
        'spectral_flatness', 'spectral_centroid'
    ]
    def __init__(self,
        rate_sample: int,
        size_win: int,
        stride_win: int,
        freq_min: float,
        freq_max: float,
        nb_mfcc: int
    ) -> None:
        super(AudioFeatureExtractor, self).__init__() 

        self._rate_sample = rate_sample
        self._size_win = size_win
        self._stride_win = stride_win
        self._freq_min = freq_min
        self._freq_max = freq_max
        self._nb_mfcc = nb_mfcc
        #
        self._nb_freqs = self._size_win
    
    def call(self,
        x_audio: tf.Tensor
    ) -> tf.Tensor:
        """Compute feature tensor"""
        x_features = _compute_audio_features(
            x_audio,
            rate_sample=self._rate_sample,
            size_win=self._size_win,
            stride_win=self._stride_win,
            freq_min=self._freq_min,
            freq_max=self._freq_max,
            nb_freqs=self._nb_freqs,
            nb_mfcc=self._nb_mfcc
        )
        x_features_used = [x_features[name_feature_used] for name_feature_used in self._NAMES_FEATURES_USED]
        # All features should have the same shape
        # That way, they can be stacked in a new tensor.
        shapes = [feat.shape for feat in x_features_used]
        if all(s.is_fully_defined() for s in shapes):
            tf.debugging.Assert(all(s == shapes[0] for s in shapes), [shapes[0]])
        x_features_used = tf.stack(x_features_used, axis=0) 
        output_shape = self._compute_output_shape(x_audio.shape)
        x_features_used = tf.ensure_shape(x_features_used, shape=output_shape)
        return x_features_used

    def _compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute here so we don't loose the shape info due to preprocessing. 
        """
        tf.assert_greater(tf.size(input_shape), 1)
        output_shape = input_shape.as_list()
        # insert the feature dimension 
        # [..., time_1, channel] -> [..., feature, time_1, channel]
        output_shape.insert(-2, len(self._NAMES_FEATURES_USED))
        # and change (downsample) time dimension.
        # [..., feature, time_1, channel] -> [..., feature, time_2, channel]
        output_shape[-2] = output_shape[-2] // self._stride_win + 1
        output_shape = tf.TensorShape(output_shape)
        return output_shape


def _compute_audio_features(
    x_audio: tf.Tensor,
    rate_sample: int,
    size_win: int,
    stride_win: int,
    freq_min: float,
    freq_max: float,
    nb_freqs: int,
    nb_mfcc: int
) -> Dict[str, tf.Tensor]:
    _NAMES_FEATURES = [
        'mfcc', 'freq_0', 'prob_freq',
        'root_mean_square', 'zero_crossing_rate',
        'spectral_flatness', 'spectral_centroid'
    ]
    # our pre-processing assumes channel-first (no batch)
    # forward permutation: from [..., time, channel] to [channel, ..., time]
    perm = tf.roll(tf.range(tf.rank(x_audio)), shift=1, axis=0)
    x_features = tf.numpy_function(
        _compute_audio_features_numpy,
        [
            tf.transpose(x_audio, perm=perm),
            rate_sample,
            size_win,
            stride_win,
            freq_min,
            freq_max,
            nb_freqs,
            nb_mfcc
        ],
        Tout=[x_audio.dtype] * len(_NAMES_FEATURES)
    )
    # inverse permutation: from [channel, ..., feature, time] to [..., feature, time, channel]
    for idx, feat in enumerate(x_features):
        perm_inv = tf.roll(tf.range(tf.rank(feat)), shift=-1, axis=0)
        x_features[idx] = tf.transpose(feat, perm=perm_inv)
    return dict(zip(_NAMES_FEATURES, x_features))


def _compute_audio_features_numpy(
    x_audio: np.ndarray,
    rate_sample: int,
    size_win: int,
    stride_win: int,
    freq_min: float,
    freq_max: float,
    nb_freqs: int,
    nb_mfcc: int
) -> List[np.ndarray]:

    x_spec = librosa.stft(
        y=x_audio,
        n_fft=nb_freqs,
        win_length=size_win,
        hop_length=stride_win,
        center=True
    )
    x_spec = x_spec[:-1]
    print(x_spec.shape)
    
    x_mel = librosa.feature.melspectrogram(
        S=np.abs(x_spec),
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
        hop_length=stride_win,
        center=True
    )
    x_freq_0 = x_freq_0.astype(x_audio.dtype)[:-1]
    x_prob_freq = x_prob_freq.astype(x_audio.dtype)[:-1]

    # Functions in librosa.feature add a dimension
    # in second-to-last position.
    # we remove it to match the other features.
    x_rms = librosa.feature.rms(
        S=np.abs(x_spec),
        frame_length=size_win,
        hop_length=stride_win,
        center=True
    ).astype(x_audio.dtype)
    x_rms = np.squeeze(x_rms, axis=-2)

    x_zcr = librosa.feature.zero_crossing_rate(
        y=x_audio,
        frame_length=size_win,
        hop_length=size_win,
        center=True
    ).astype(x_audio.dtype)
    x_zcr = np.squeeze(x_zcr, axis=-2)[:-1]

    x_flatness = librosa.feature.spectral_flatness(
        S=np.abs(x_spec),
    ).astype(x_audio.dtype)
    x_flatness = np.squeeze(x_flatness, axis=-2)

    
    x_centroid = librosa.feature.spectral_centroid(
        S=np.abs(x_spec)
    ).astype(x_audio.dtype)
    x_centroid = np.squeeze(x_centroid, axis=-2)

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
