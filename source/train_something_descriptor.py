import os
import sys
import tensorflow as tf
import tensorflow_datasets as tfds

from data_loaders.example_data_loader import ExampleDatasetGenerator
from models.models import JukeboxModel
from utils.dirs import create_dirs
from utils.config import process_config
from utils.utils import get_args
from utils.preprocessing import preprocess_audio_features

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
        id_gpu_locked = gpl.get_gpu_lock(soft=False)
        # os.environ["CUDA_VISIBLE_DEVICES"]=""
    except gpl.NoGpuManager:
        print("no gpu manager available - will use all available GPUs", file=sys.stderr)
    except gpl.NoGpuAvailable:
        # there is no GPU available for locking, continue with CPU
        comp_device = "/cpu:0" 
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        # nope.
        # print("No gpu available!")
        # exit(1)
    
    ds_train, ds_val = tfds.load(
        'canonne_duos',
        split=['train[90%:]', 'train[:10%]'],
        data_dir=config.dataset.path,
        download=False
        )

    def _preprocess(inputs: dict) -> dict:
        x_audio = inputs['audio']
        x_features = preprocess_audio_features(x_audio, config.data)
        x = tf.expand_dims(x_features[..., 0], axis=-1)
        y = x
        print(x.shape, y.shape)
        return x, y
    ds_train = ds_train.map(_preprocess).batch(
        config.training.size_batch).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.map(_preprocess).batch(config.training.size_batch)

    # Infer processed input shape
    shape_input = ds_train.element_spec[0].shape
    # Since the preprocessing makes us lose this info,
    # here's an hacky way to do it.
    example = ds_train.get_single_element()

    # create an instance of the model you want
    model = JukeboxModel(shape_input=shape_input, **vars(config.model.jukebox))
    model.compile(
        loss=tf.keras.losses.MSE(),
        run_eagerly=config.debug.enabled
    )
    #
    # Callbacks
    cb_log = tf.keras.callbacks.TensorBoard(
        config.save.path.log_dir,
        **vars(config.save.log)
    )
    print(os.path.join(config.save.path.checkpoint_dir, '{epoch:02d}.hdf5'),)
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
        ],
        batch_size=config.training.size_batch,
        epochs=config.training.nb_epochs,
        steps_per_epoch=None
    )
    model.build(shape_input)
    model.summary(expand_nested=True)
