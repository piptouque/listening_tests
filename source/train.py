import os
import tensorflow as tf
import tensorflow_datasets as tfds

from data_loaders.example_data_loader import ExampleDatasetGenerator
from data_loaders.preprocessing import AudioPreprocessor
from models.models import ExampleModel
from utils.dirs import create_dirs
from utils.config import process_config
from utils.utils import get_args

import datasets.canonne_duos.canonne_duos


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    args = get_args()
    config = process_config(args)

    # create the experiments dirs
    create_dirs([config.save.path.log_dir,
                config.save.path.checkpoint_dir])

    # create tensorflow session
    # data = DataGenerator(config)
    # ds_train, ds_val = ExampleDatasetGenerator(config)()
    ds_train, ds_val = tfds.load('canonne_duos', split=[
                                 'train[90%:]', 'train[:10%]'])

    # Add preprocessing of data
    # preprocessor = AudioPreprocessor(config)
    # preprocessor.compile(run_eagerly=True)

    def _preprocess(inputs: dict) -> dict:
        x_spec, x_ann_spec = AudioPreprocessor.compute_spectrum(
            inputs['audio'], inputs['annotations'],
            config.audio.nb_freqs, config.audio.size_win, config.audio.stride_win)
        inputs['spec'] = x_spec
        inputs['ann_spec'] = x_ann_spec
        return inputs
    ds_train = ds_train.map(_preprocess).batch(
        config.training.size_batch).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.map(_preprocess).batch(config.training.size_batch)
    # create an instance of the model you want
    # model = ExampleModel(config)
    model = ExampleModel(config)
    model.compile(run_eagerly=config.debug.enabled)
    # Callbacks
    cb_log = tf.keras.callbacks.TensorBoard(
        config.save.path.log_dir,
        **vars(config.save.log)
    )
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            config.save.path.checkpoint_dir, '{epoch:02d}.hdf5'),
        save_best_only=True,
        monitor='val_acc',
        mode='min'
    )
    #
    model.fit(
        x=ds_train,
        validation_data=ds_val,
        callbacks=[
            cb_log
        ]
    )
