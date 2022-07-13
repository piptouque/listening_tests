from abc import abstractmethod
from typing import Tuple, Dict, List, Any

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


import librosa


class AudioFeatureExtractor(tf.keras.layers.Layer):
    """Simple audio feature extractor"""

    def __init__(self,
                 names_features: List[str],
                 rate_sample: int,
                 size_win: int,
                 stride_win: int,
                 freq_min: float,
                 freq_max: float,
                 nb_freqs_mel: int,
                 ) -> None:
        super(AudioFeatureExtractor, self).__init__()

        self._rate_sample = rate_sample
        self._size_win = size_win
        self._stride_win = stride_win
        self._freq_min = freq_min
        self._freq_max = freq_max
        self._nb_freqs_mel = nb_freqs_mel
        #
        self._names_features = names_features

    def call(self, x_audio: tf.Tensor) -> tf.Tensor:
        """Compute feature tensor"""
        x_features = self._extract_features(x_audio)
        output_shape = self.compute_output_shape(x_audio.shape)
        x_features = tf.ensure_shape(x_features, shape=output_shape)
        return x_features

    def get_config(self) -> Dict[str, Any]:
        config = super(AudioFeatureExtractor, self).get_config()
        config.update({
            'names_features': self._names_features,
            'rate_sample': self._rate_sample,
            'size_win': self._size_win,
            'stride_win': self._stride_win,
            'freq_min': self._freq_min,
            'freq_max': self._freq_max,
            'nb_freqs_mel': self._nb_freqs_mel
        })

    @abstractmethod
    def _extract_features(self, x_audio: tf.Tensor) -> tf.Tensor:
        return NotImplemented

    @abstractmethod
    def compute_output_shape(self, shape_input: tf.TensorShape) -> tf.TensorShape:
        return NotImplemented

    def _compute_audio_descriptors(
        self,
        x_audio: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        # order matters, must be the same as the output of *_numpy
        _NAMES_FEATURES = [
            'freq_0', 'prob_freq',
            'root_mean_square', 'zero_crossing_rate',
            'spectral_flatness', 'spectral_centroid'
        ]
        # our pre-processing assumes time is in the last dimension
        # forward permutation: from [batch, ..., time, channel] to [channel, bach, ..., time]
        perm = tf.roll(tf.range(tf.rank(x_audio)), shift=1, axis=0)
        x_features = tf.numpy_function(
            self._compute_audio_descriptors_numpy,
            [
                tf.transpose(x_audio, perm=perm),
                self._rate_sample,
                self._size_win,
                self._stride_win,
                self._freq_min,
                self._freq_max,
                self._size_win,
            ],
            Tout=[x_audio.dtype] * len(_NAMES_FEATURES)
        )

        # inverse first permutation,
        # and also switch time and feature dim
        # from [channel, batch, ..., feature, time] to [batch, ..., feature, time, channel]
        for idx, feat in enumerate(x_features):
            perm_inv = tf.range(tf.rank(feat))
            perm_inv = tf.concat(
                [perm_inv[:-2], [perm_inv[-1]], [perm_inv[-2]]], 0)
            perm_inv = tf.roll(perm_inv, shift=-1, axis=0)
            x_features[idx] = tf.transpose(feat, perm=perm_inv)
        return dict(zip(_NAMES_FEATURES, x_features))

    def _compute_audio_spectrum(
        self,
        x_audio: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        # our pre-processing assumes time is in the last dimension
        # forward permutation: from [batch,..., time, channel] to [channel, batch, ..., time]
        perm = tf.roll(tf.range(tf.rank(x_audio)), shift=1, axis=0)
        x_audio_trans = tf.transpose(x_audio, perm=perm)
        nb_freqs_fft = self._size_win
        x_stft = tf.signal.stft(x_audio_trans,
                                frame_length=self._size_win,
                                frame_step=self._stride_win,
                                fft_length=nb_freqs_fft,
                                pad_end=True
                                )
        m_mel = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self._nb_freqs_mel,
            num_spectrogram_bins=x_stft.shape[-1],
            sample_rate=self._rate_sample,
            lower_edge_hertz=self._freq_min,
            upper_edge_hertz=self._freq_max
        )
        x_spec_mel = tf.tensordot(tf.abs(x_stft), m_mel, axes=1)
        x_spec_mel.set_shape(x_stft.shape[:-1].concatenate(m_mel.shape[-1:]))
        # x_mfcc = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(x_spec_mel + 1e-6))
        # x_features = x_mfcc
        x_features = x_spec_mel
        # inverse permutation: from [channel, batch,..., feature, time] to [batch,..., time, feature, channel]
        perm_inv = tf.roll(tf.range(tf.rank(x_features)), shift=-1, axis=0)
        x_features = tf.transpose(x_features, perm=perm_inv)
        return dict(zip(self._names_features, [x_features]))

    @staticmethod
    def _compute_audio_descriptors_numpy(
        x_audio: np.ndarray,
        rate_sample: int,
        size_win: int,
        stride_win: int,
        freq_min: float,
        freq_max: float,
        nb_freqs: int,
    ) -> List[np.ndarray]:

        # librosa does not compute frame_length
        # as time // stride
        # so we need to remove an element (arbitrarily, the last one).
        # see: https://github.com/librosa/librosa/issues/1278
        x_spec = librosa.stft(
            y=x_audio,
            n_fft=nb_freqs,
            win_length=size_win,
            hop_length=stride_win,
            center=True
        )
        x_spec = x_spec[..., :-1]

        x_freq_0, _, x_prob_freq = librosa.pyin(
            y=x_audio,
            fmin=freq_min,
            fmax=freq_max,
            sr=rate_sample,
            frame_length=size_win,
            hop_length=stride_win,
            center=True
        )
        x_freq_0 = x_freq_0.astype(x_audio.dtype)[..., :-1]
        x_prob_freq = x_prob_freq.astype(x_audio.dtype)[..., :-1]

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
            hop_length=stride_win,
            center=True
        ).astype(x_audio.dtype)
        x_zcr = np.squeeze(x_zcr, axis=-2)[..., :-1]

        x_flatness = librosa.feature.spectral_flatness(
            S=np.abs(x_spec),
        ).astype(x_audio.dtype)
        x_flatness = np.squeeze(x_flatness, axis=-2)

        x_centroid = librosa.feature.spectral_centroid(
            S=np.abs(x_spec)
        ).astype(x_audio.dtype)
        x_centroid = np.squeeze(x_centroid, axis=-2)

        return [
            x_freq_0,
            x_prob_freq,
            x_rms,
            x_zcr,
            x_flatness,
            x_centroid
        ]


class AudioSpectrumExtractor(AudioFeatureExtractor):
    """For spectrum only"""

    def _extract_features(self, x_audio: tf.Tensor) -> tf.Tensor:
        """Compute feature tensor"""
        tf.assert_equal(len(self._names_features), 1)
        x_features = self._compute_audio_spectrum(x_audio)
        x_features = x_features[self._names_features[0]]
        return x_features

    def compute_output_shape(self, shape_input: tf.TensorShape) -> tf.TensorShape:
        """Compute here so we don't loose the shape info due to preprocessing.
        """
        shape_output = np.asarray(shape_input.as_list())
        # change (downsample) time dimension.
        # [..., time_1, channel] -> [..., time_2, channel]
        shape_output[-2] = shape_output[-2] // self._stride_win
        # insert the feature dimension AFTER the time
        # [..., time_2, channel] -> [..., feature, time_2, channel]
        shape_output = np.insert(shape_output, -1, self._nb_freqs_mel)
        shape_output = tf.TensorShape(shape_output)
        return shape_output


class AudioDescriptorsExtractor(AudioFeatureExtractor):
    """For descriptors only"""

    def _extract_features(self,
                          x_audio: tf.Tensor,
                          ) -> tf.Tensor:
        """Compute feature tensor"""
        x_features = self._compute_audio_descriptors(x_audio)
        x_features = [x_features[name_feature_used]
                      for name_feature_used in self._names_features]
        # All features should have the same shape
        # That way, they can be stacked in a new tensor.
        x_features = tf.stack(x_features, axis=-2)
        # remove the audio channel dimension,
        x_features = tf.squeeze(x_features, axis=-1)
        return x_features

    def compute_output_shape(self, shape_input: tf.TensorShape) -> tf.TensorShape:
        """Compute here so we don't loose the shape info due to preprocessing. 
        """
        # FIXME
        tf.assert_equal(shape_input[-1], 1)
        shape_output = np.asarray(shape_input.as_list())
        # change (downsample) time dimension.
        # [..., time_1, channel] -> [..., time_2, channel]
        shape_output[-2] = shape_output[-2] // self._stride_win
        # The new channel dimension (previously features) at the end
        shape_output[-1] = len(self._names_features)
        shape_output = tf.TensorShape(shape_output)
        return shape_output
