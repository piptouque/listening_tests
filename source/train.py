from typing import Tuple
import os
import sys
import tensorflow as tf
import tensorflow_datasets as tfds

from data_loaders.example_data_loader import ExampleDatasetGenerator
from models.models import JukeboxAutoEncoder
from models.preprocessing import AudioFeatureExtractor
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
                to=int(100*config.dataset.validation_split),
                unit='%'
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
            nb_freqs_mel=config.data.audio.freq.nb_freqs_mel
        )

        def prep_ds(ds: tf.data.Dataset):
            """Reshaping the dataset before pre-processing.

            Args:
                ds (tf.data.Dataset): _description_

            Returns:
                _type_: _description_
            """
            name_field = 'audio'
            ds = ds.map(lambda inputs: inputs[name_field])
            #
            has_channel_dim = ds.element_spec.shape.rank >= 2
            is_monophonic = ds.element_spec.shape[-1] == 1
            if is_monophonic or not has_channel_dim:
                if is_monophonic:
                    ds = ds.map(lambda el: tf.squeeze(el, axis=[-1]))
                # fabricate polyphonic examples from two consecutive monophonic examples.
                ds = ds \
                    .batch(2, drop_remainder=True) \
                    .map(
                        lambda el: tf.transpose(el, perm=tf.roll(tf.range(tf.rank(el)), shift=-1, axis=0))
                    )
            # Split example between input and target
            ds_a = ds.map(lambda el: tuple(tf.unstack(tf.expand_dims(el, axis=-1), axis=-2)))
            ds_b = ds_a.map(lambda *inputs: inputs[::-1])
            ds = ds_a.concatenate(ds_b)
            return ds

        ds_train = prep_ds(ds_train)
        ds_val = prep_ds(ds_val) \

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
        preprocessed_inputs = feature_extractor(inputs)
        jukebox = JukeboxAutoEncoder(**vars(config.model.jukebox))
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
