from typing import Tuple
import os
import sys
import tensorflow as tf
import tensorflow_datasets as tfds

from data_loaders.example_data_loader import ExampleDatasetGenerator
from models.models import JukeboxModel
from models.preprocessing import AudioFeatureExtractor
from utils.dirs import create_dirs
from utils.config import process_config
from utils.utils import get_args

import datasets.canonne_duos.canonne_duos


class Preprocessor(tf.keras.Model):
    def __init__(self, feature_extractor: tf.keras.Model, config: object) -> None:
        super(Preprocessor, self).__init__()
        self._feature_extractor = feature_extractor
        self._config = config

    def _preprocess_mnist(self, x: tf.Tensor) -> tf.Tensor:
        # x_audio = inputs['audio']
        # x_features = feature_extractor(x_audio)
        # x = x_features
        # y = x
        x_image = x
        x = tf.cast(x_image, tf.float32)
        return x

    def _preprocess_audio(self, x: tf.Tensor) -> tf.Tensor:
        x_audio = x
        x_features = self._feature_extractor(x_audio)
        x = x_features
        return x

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self._config.dataset.name == 'mnist':
            return self._preprocess_mnist(x)
        else:
            return self._preprocess_audio(x)
        

if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    args = get_args()
    config = process_config(args)

    # create the experiments dirs
    create_dirs([config.save.path.log_dir,
                config.save.path.checkpoint_dir])

    import manage_gpus as gpl
    try:
        id_gpu_locked = gpl.get_gpu_lock(soft=True)
        comp_device = f"/gpu:{id_gpu_locked}"
        # os.environ["CUDA_VISIBLE_DEVICES"]=""
    except gpl.NoGpuManager:
        print("no gpu manager available - will use all available GPUs", file=sys.stderr)
    except gpl.NoGpuAvailable:
        # there is no GPU available for locking, continue with CPU
        comp_device = "/cpu:0" 
        os.environ["CUDA_VISIBLE_DEVICES"]=""

    if "CUDA_VISIBLE_DEVICES" in os.environ and len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0:
        devices = tf.config.list_physical_devices('GPU')
        print(len(devices))
        tf.config.experimental.set_memory_growth(device=devices[0], enable=True)
    else:
        comp_device = "/cpu:0"

    with tf.device(comp_device):

        split = [
            tfds.core.ReadInstruction(
                'train',
                from_=int(100*config.dataset.validation_split),
                unit='%'
            ),
            tfds.core.ReadInstruction(
                'train',
                to=int(100*config.dataset.validation_split)
            )
        ]
        ds_train, ds_val = tfds.load(
            config.dataset.name,
            split=split,
            data_dir=config.dataset.path,
            download=True
        )
        
        feature_extractor = AudioFeatureExtractor.make(
            kind_feature=config.data.audio.features.kind,
            name_features=config.data.audio.features.names,
            rate_sample=config.data.audio.time.rate_sample,
            size_win=config.data.audio.time.size_win,
            stride_win=config.data.audio.time.stride_win,
            freq_min=config.data.audio.freq.freq_min,
            freq_max=config.data.audio.freq.freq_max,
            nb_mfcc=config.data.audio.freq.nb_mfcc
        )

        preprocessor = Preprocessor(feature_extractor, config)

        name_field = 'image' if config.dataset.name == 'mnist' else 'audio'
        # for now, process left and right channel separately
        # stereo example as two mono examples -> unbatch then batch again.
        ds_train = ds_train \
            .map(lambda inputs: (inputs[name_field], None)) \
            # .batch(config.training.size_batch) \
            # .prefetch(tf.data.AUTOTUNE)
            # .map( _preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
            # .unbatch() \
            # .shuffle(4) \
        ds_val = ds_val \
            .map(lambda inputs: (inputs[name_field], None)) \
            # .batch(config.training.size_batch) \
            # .prefetch(tf.data.AUTOTUNE)
            # .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
            # .unbatch() \
            # .shuffle(4) \
        if name_field == 'audio':
            ds_train = ds_train.unbatch()
            ds_val = ds_val.unbatch()
        ds_train = ds_train \
            .shuffle(4) \
            .batch(config.training.size_batch) \
            .prefetch(tf.data.AUTOTUNE)
        ds_val = ds_val \
            .batch(config.training.size_batch) \
            .prefetch(tf.data.AUTOTUNE)

        # Infer processed input shape
        shape_input = ds_train.element_spec[0].shape

        # create an instance of the model you want
        inputs = tf.keras.Input(shape=shape_input[1:])
        preprocessed_inputs = preprocessor(inputs)
        jukebox = JukeboxModel(**vars(config.model.jukebox))
        outputs = jukebox(preprocessed_inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            metrics=[],
            run_eagerly=config.debug.enabled
        )
        #
        model.build(shape_input)
        model.summary(expand_nested=False)

        # Callbacks
        cb_log = tf.keras.callbacks.TensorBoard(
            config.save.path.log_dir,
            **vars(config.save.log)
        )
        cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.save.path.checkpoint_dir, 'best_{epoch:02d}.tf'),
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            mode='min'
        )
        #
        model.fit(
            x=ds_train,
            validation_data=ds_val,
            callbacks=[cb_checkpoint, cb_log],
            epochs=config.training.nb_epochs,
            steps_per_epoch=None
        )
