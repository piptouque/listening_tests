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
    
    shape_preprocessed = ds_train.element_spec['audio'].shape

    feature_extractor = AudioFeatureExtractor(
        config.data.audio.time.rate_sample,
        config.data.audio.time.size_win,
        config.data.audio.time.stride_win,
        config.data.audio.freq.freq_min,
        config.data.audio.freq.freq_max,
        config.data.audio.freq.nb_mfcc
    )
    # feature_extractor.build(shape_preprocessed)

    def _preprocess(inputs: dict) -> dict:
        x_audio = inputs['audio']
        x_features = feature_extractor(x_audio)
        # for now, process left and right channel separately
        # add batch as the audio channel dim
        # And replace channel by the features as last dim.
        # [..., f, t_2, c] -> [c, ..., t_2, f]
        perm = tf.range(tf.rank(x_features))
        perm = tf.roll(perm, shift=1, axis=0)
        perm = tf.concat([perm[:-2], [perm[-1]], [perm[-2]]], 0)
        x_features = tf.transpose(x_features, perm=perm)
        x = x_features
        y = x
        return x, y
    
    # stereo example as two mono examples -> unbatch then btach again.
    ds_train = ds_train.map(_preprocess) \
        .unbatch() \
        .batch(config.training.size_batch) \
        .prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.map(_preprocess) \
        .unbatch() \
        .batch(config.training.size_batch)

    # Infer processed input shape
    shape_input = ds_train.element_spec[0].shape

    # create an instance of the model you want
    model = JukeboxModel(shape_input=shape_input, **vars(config.model.jukebox))
    model.compile(
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
    # model.build(shape_input)
    # model.summary(expand_nested=False)
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
