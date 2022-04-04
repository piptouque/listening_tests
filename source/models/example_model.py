import numpy as np
import numpy.typing as npt

import tensorflow as tf
import tensorflow_io as tfio

import librosa


class AudioPreprocessor(tf.keras.Model):
    """_summary_
    """

    def __init__(self, conf: object):
        super(AudioPreprocessor, self).__init__()
        self._conf = conf.audio
        self.l1 = tf.keras.layers.InputLayer()

    def call(self, inputs) -> tf.Tensor:
        x_audio = inputs['audio']
        x_ann = inputs['annotations']
        ann_x_play = x_ann['x_play']
        ann_y_play = x_ann['y_play']
        ann_dir_play = x_ann['dir_play']
        label = inputs['labels']
        #
        # Resample the annotations to the audio sample rate
        print(x_audio.shape)
        x_spec = tf.expand_dims(tfio.audio.spectrogram(
            tf.squeeze(x_audio, axis=-1),
            nfft=self._conf.nb_freqs,
            window=self._conf.size_win,
            stride=self._conf.stride_win),
            axis=-1
        )
        print(x_spec.shape)
        x_ann['x_play'] = tf.convert_to_tensor(librosa.util.frame(
            ann_x_play.numpy(),
            frame_length=self._conf.size_win,
            hop_length=self._conf.stride_win,
            axis=0
        ))
        print(x_ann['x_play'].shape)
        x_ann['y_play'] = tf.convert_to_tensor(librosa.util.frame(
            ann_y_play.numpy(),
            frame_length=self._conf.size_win,
            hop_length=self._conf.stride_win,
            axis=0
        ))
        x_ann['dir_play'] = tf.convert_to_tensor(librosa.util.frame(
            ann_dir_play.numpy(),
            frame_length=self._conf.size_win,
            hop_length=self._conf.stride_win,
            axis=0
        ))
        return (x_spec, x_ann, label)


class ExampleModel(tf.keras.Model):
    def __init__(self, conf: object):
        super(ExampleModel, self).__init__()

        self.trans = AudioPreprocessor(conf)

        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')
        #

    def call(self, inputs) -> tf.Tensor:
        x_spec, x_ann, label = self.trans(inputs)
        z = self.conv1(x_spec)
        z = self.flatten(z)
        z = self.d1(z)
        return self.d2(z)
