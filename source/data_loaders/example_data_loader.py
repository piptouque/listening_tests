import tensorflow as tf
import tensorflow_datasets as tfds


class ExampleDatasetGenerator:
    def __init__(self, config: object):
        self._config = config.training
        # data = tfds.load("mnist", with_info=True)
        train_ds, test_ds = tfds.load('mnist', split=['train', 'test'])
        self.train_data = train_ds
        self.test_data = test_ds
        assert isinstance(self.train_data, tf.data.Dataset)

    def __call__(self):
        self.preprocess()
        return self.train_data, self.test_data

    def preprocess(self):
        self.train_data = self.train_data.map(
            ExampleDatasetGenerator.convert_types
        ).batch(self._config.batch_size)
        self.test_data = self.test_data.map(
            ExampleDatasetGenerator.convert_types
        ).batch(self._config.batch_size)

    @staticmethod
    def convert_types(batch):
        image, label = batch.values()
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
