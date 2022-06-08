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

        ds_train, ds_val = tfds.load(
            'canonne_duos',
            split=['train[90%:]', 'train[:10%]'],
            data_dir=config.dataset.path,
            download=False
        )
        
        shape_preprocessed = ds_train.element_spec['audio'].shape

        feature_extractor = AudioFeatureExtractor(
            config.data.audio.kind,
            config.data.audio.time.rate_sample,
            config.data.audio.time.size_win,
            config.data.audio.time.stride_win,
            config.data.audio.freq.freq_min,
            config.data.audio.freq.freq_max,
            config.data.audio.freq.nb_mfcc
        )

        @tf.function
        def _preprocess(inputs: dict) -> Tuple[tf.Tensor, tf.Tensor]:
            x_audio = inputs['audio']
            x_features = feature_extractor(x_audio)
            x = x_features
            y = x
            return x, y
        
        # for now, process left and right channel separately
        # stereo example as two mono examples -> unbatch then batch again.
        ds_train = ds_train.map(_preprocess) \
            .unbatch() \
            .batch(config.training.size_batch) \

        # Infer processed input shape
        shape_input = ds_train.element_spec[0].shape

        # create an instance of the model you want
        model = JukeboxModel(shape_input=shape_input, **vars(config.model.jukebox))
        model.compile(
            run_eagerly=config.debug.enabled
        )
        #
        model.fit(
            x=ds_train,
            callbacks=[ ],
            batch_size=config.training.size_batch,
            epochs=config.training.nb_epochs,
            steps_per_epoch=None
        )
