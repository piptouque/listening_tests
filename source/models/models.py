import tensorflow as tf

from typing import Dict


class ExampleModel(tf.keras.Model):
    def __init__(self, config: object):
        super(ExampleModel, self).__init__()

        self._config = config

        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')
        #

    def call(self, inputs: dict) -> tf.Tensor:
        x = inputs['spec']
        x_ann = inputs['ann_spec']
        label = inputs['labels']

        z = self.conv1(x)
        z = self.flatten(z)
        z = self.d1(z)
        return self.d2(z)
