import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from typing import Tuple

import abc


class ExampleDatasetGenerator:
    def __init__(self, config: object):
        self._config_dataset = config.dataset
        self._config_training = config.training
        # data = tfds.load("mnist", with_info=True)
        self.ds_train, self.ds_val = self.load_split_training(
            self._config_dataset.name, self._config_dataset.validation_split)
        assert isinstance(self.ds_train, tf.data.Dataset)

    def __call__(self):
        self.preprocess()
        return self.ds_train, self.ds_val

    @abc.abstractmethod
    def preprocess(self) -> None:
        """_summary_
        """
        pass

    @staticmethod
    def load_split_training(dataset_name: str, validation_split: float) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """_summary_

        Args:
            dataset_name (str): _description_
            validation_split (float): _description_

        Returns:
            Tuple[tf.Dataset, tf.Dataset]: _description_
        """
        """
        ds_train = tfds.load(dataset_name, split=[
            f'train[{int(np.round((1 - validation_split) * 100))}%:]'])
        ds_val = tfds.load(dataset_name, split=[
            f'train[:{int(np.round(validation_split * 100))}%]'])
        """
        ds_train, ds_val = tfds.load(
            dataset_name, split=['train', 'validation'])
        return ds_train, ds_val
