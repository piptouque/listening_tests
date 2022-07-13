"""canonne_duos dataset."""

from pathlib import Path
import re
from typing import Callable, List

import librosa
import pandas as pd
import numpy as np
# import numpy.typing as npt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio
import dataclasses


_URL_DOWNLOAD = None
# TODO(canonne_duos): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """
@article{golvetFamiliarityCopresenceIncrease2021,
  title = {With, against, or without? {{Familiarity}} and Copresence Increase Interactional Dissensus and Relational Plasticity in Freely Improvising Duos.},
  shorttitle = {With, against, or Without?},
  author = {Golvet, Théo and Goupil, Louise and Saint-Germier, Pierre and Matuszewski,
      Benjamin and Assayag, Gérard and Nika, Jérôme and Canonne, Clément},
  date = {2021-09-02},
  journaltitle = {Psychology of Aesthetics, Creativity, and the Arts},
  issn = {1931-390X, 1931-3896},
  doi = {10.1037/aca0000422},
  url = {http://doi.apa.org/getdoi.cfm?doi=10.1037/aca0000422},
  urldate = {2022-02-28},
  abstract = {Agents engaged in creative joint actions might need to find a balance between the demands of doing something collectively, by adopting congruent and interacting behaviors, and the goal of delivering a creative output, which can eventually benefit from disagreements and autonomous behaviors. Here, we investigate this idea in the context of collective free improvisation – a paradigmatic example of group creativity in which musicians aim at creating music that is as complex and unprecedented as possible without relying on predefined plans or individual roles. Controlling for both the familiarity between the musicians and their physical co-presence, duos of improvisers were asked to freely improvise together and to individually annotate their performances with a digital interface, indicating at each time whether they were playing “with”, “against”, or “without” their partner. At an individual level, we found that musicians largely intended to converge with their co-improviser, making only occasional use of non-cooperative or non-interactive modes such as “playing against” or “playing without”. By contrast, at the group level, musicians tended to combine their relational intents in such a way as to create interactional dissensus. We also demonstrate that co-presence and familiarity act as interactional smoothers: they increase the agents’ overall level of relational plasticity and allow for the exploration of less cooperative behaviors. Overall, our findings suggest that relational intents might function as a primary resource for creative joint actions.},
  langid = {english}
}
"""
_HOMEPAGE = None

_NB_DUOS = 10
_NB_TAKES = 4
_NB_CHANNELS = 2

_RATE_AUDIO = 48000
# Wanted length (in seconds) for a single example.
# Will be increased to the next power of two.
_LENGTH_BLOCK_DESIRED = 2
# better to have a power of two.
_SIZE_BLOCK = int(2 ** np.ceil(np.log2(_RATE_AUDIO * _LENGTH_BLOCK_DESIRED)))
_DTYPE_AUDIO = tf.float32

_RATE_ANNOTATION = 4

_MAP_LABEL_INSTRUMENTS = {
    'piano': 0,
    'electric-guitar': 1,
    'bass-guitar': 2,
    'viola': 3,
    'cello': 4,
    'double-bass': 5,
    'voice': 6,
    'trumpet': 7,
    'saxophone': 8,
}

_MAP_LABELS_PLAYING = {
    'avec': 1,
    'sans': 0,
    'contre': -1
}

_PATTERN_FILE_AUDIO = r'Duo(?P<duo>\d+)_(?P<take>\d+)_(?P<channel>\d+)_(?P<instrument>(\w|\-)+).wav'
_PATTERN_DIR_DUO = r'duo_(?P<duo>\d+)'
_PATTERN_DIR_TAKE = r'take_(?P<take>\d+)'


@dataclasses.dataclass
class CanonneDuosConfig(tfds.core.BuilderConfig):
    kind_split: str = 'joined'


class CanonneDuos(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for canonne_duos dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    TODO
    """
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        CanonneDuosConfig(
            name='joined',
            description='Tracks from the same take and duo are joined in a single example',
            kind_split='joined'
        )
    ]
    # pytype: enable=wrong-keyword-args

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(canonne_duos): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            metadata=tfds.core.MetadataDict({
                'rate_sample': _RATE_AUDIO,
                'rate_annotation': _RATE_ANNOTATION
            }),
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'audio': tfds.features.Audio(shape=(_SIZE_BLOCK, _NB_CHANNELS), file_format='wav', sample_rate=_RATE_AUDIO, dtype=tf.float32),
                'annotations': {
                    'x_play': tfds.features.Tensor(shape=(_SIZE_BLOCK, _NB_CHANNELS), dtype=tf.dtypes.float32),
                    'y_play': tfds.features.Tensor(shape=(_SIZE_BLOCK, _NB_CHANNELS), dtype=tf.dtypes.float32),
                    'dir_play': tfds.features.Tensor(shape=(_SIZE_BLOCK, _NB_CHANNELS), dtype=tf.dtypes.int8),
                },
                'labels': {
                    'idx_duo': tfds.features.ClassLabel(num_classes=_NB_DUOS+1),
                    'idx_take': tfds.features.ClassLabel(num_classes=_NB_TAKES+1),
                    'idx_channel': tfds.features.Tensor(shape=(_NB_CHANNELS,), dtype=tf.dtypes.int8),
                    'idx_instrument': tfds.features.Tensor(shape=(_NB_CHANNELS,), dtype=tf.dtypes.int8)
                }
                # 'label': tfds.features.ClassLabel(names=['no', 'yes']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('audio', 'labels'),  # Set to `None` to disable
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(canonne_duos): Downloads the data and defines the splits
        if dl_manager.manual_dir is None:
            path = dl_manager.download_and_extract(_URL_DOWNLOAD)
        else:
            path = dl_manager.manual_dir

        return {
            'train': self._generate_examples(
                path_audio=path / 'audio' / 'train',
                path_ann=path / 'annotations'
            ),
            'test': self._generate_examples(
                path_audio=path / 'audio' / 'test',
                path_ann=path / 'annotations'
            )
        }

    def _generate_examples(self, path_audio: Path, path_ann: Path):
        """Yields examples."""
        # TODO(canonne_duos): Yields (key, example) tuples from the dataset
        pat_file_audio = re.compile(_PATTERN_FILE_AUDIO)
        pat_dir_duo = re.compile(_PATTERN_DIR_DUO)
        pat_dir_take = re.compile(_PATTERN_DIR_TAKE)
        ann_dict = self._extract_annotations(
            path_ann / 'dt_duo_positions.csv')
        for path_dir_duo in path_audio.iterdir():
            m_dir_duo = pat_dir_duo.match(path_dir_duo.name)
            if m_dir_duo is not None:
                g_dir_duo = m_dir_duo.groupdict()
                idx_dir_duo = int(g_dir_duo['duo'])
                for path_dir_take in path_dir_duo.iterdir():
                    m_dir_take = pat_dir_take.match(path_dir_take.name)
                    if m_dir_take is not None:
                        g_dir_take = m_dir_take.groupdict()
                        idx_dir_take = int(g_dir_take['take'])
                        list_channel_examples = [None] * _NB_CHANNELS
                        for path_file_audio in path_dir_take.iterdir():
                            if path_file_audio.is_file():
                                # print('', flush=True)
                                # print(path_file_audio.name, flush=False)
                                m_file = pat_file_audio.match(
                                    path_file_audio.name)
                                if m_file is not None:
                                    # print('...okay', flush=True)
                                    #  Figure out the labels and annotations
                                    g_file = m_file.groupdict()
                                    label_instrument = g_file['instrument']
                                    idx_instrument = _MAP_LABEL_INSTRUMENTS[label_instrument]
                                    # ids have to be [1, max], idx=0 is a placeholder
                                    idx_duo = int(g_file['duo'])
                                    idx_take = int(g_file['take'])
                                    idx_channel = int(g_file['channel'])
                                    df_ann = ann_dict[idx_duo][idx_take]
                                    # some safety check
                                    assert idx_duo == idx_dir_duo,   "Inconsistent naming/directory scheme"
                                    assert idx_take == idx_dir_take, "Inconsistent naming/directory scheme"
                                    #
                                    example_whole = self._load_example_unsplit(
                                        path_file_audio, df_ann, idx_duo, idx_take, idx_channel, idx_instrument)
                                    #  Divide the audio into blocks of required length
                                    list_example_split = self._split_example(
                                        example_whole)
                                    # add to total channels
                                    list_channel_examples[idx_channel -
                                                          1] = list_example_split
                        nb_splits = len(list_channel_examples[0])
                        # Swap splits <-> channels in list of lists' dimensions.
                        list_channel_examples_swapped = [
                            [[] for _b in range(_NB_CHANNELS)] for _a in range(nb_splits)]
                        for channel, list_examples_split in enumerate(list_channel_examples):
                            assert list_examples_split is not None, f'Example Duo {idx_duo} / Take {idx_take} -- Channel {channel+1} was not found!'
                            for idx_split, example in enumerate(list_examples_split):
                                list_channel_examples_swapped[idx_split][channel] = example
                        # merge all channels in a single example of each split
                        list_examples_merged = [{
                            'audio': np.concatenate([example['audio'] for example in list_examples], axis=-1),
                            'annotations': {
                                'x_play': np.concatenate([example['annotations']['x_play'] for example in list_examples], axis=-1),
                                'y_play': np.concatenate([example['annotations']['y_play'] for example in list_examples], axis=-1),
                                'dir_play': np.concatenate([example['annotations']['dir_play'] for example in list_examples], axis=-1),
                            },
                            'labels': {
                                'idx_duo': list_examples[0]['labels']['idx_duo'],
                                'idx_take': list_examples[0]['labels']['idx_take'],
                                'idx_channel': np.stack([example['labels']['idx_channel'] for example in list_examples], axis=-1),
                                'idx_instrument': np.stack([example['labels']['idx_instrument'] for example in list_examples], axis=-1)
                            }
                        } for list_examples in list_channel_examples_swapped]
                        for idx_split, list_example_split in enumerate(list_examples_merged):
                            yield f'{path_dir_duo.name}_{idx_split}', list_example_split

    def _split_example(self, example_whole: dict) -> List[dict]:
        x_audio_split = self._split_droplast(
            example_whole['audio'], _SIZE_BLOCK)
        ann_x_play_split = self._split_droplast(
            example_whole['annotations']['x_play'], _SIZE_BLOCK)
        ann_y_play_split = self._split_droplast(
            example_whole['annotations']['y_play'], _SIZE_BLOCK)
        ann_dir_play_split = self._split_droplast(
            example_whole['annotations']['dir_play'], _SIZE_BLOCK)
        list_examples_split = []
        for i in range(x_audio_split.shape[0]):
            list_examples_split.append({
                'audio': np.expand_dims(x_audio_split[i], axis=-1),
                'annotations': {
                    'x_play': np.expand_dims(ann_x_play_split[i], axis=-1),
                    'y_play': np.expand_dims(ann_y_play_split[i], axis=-1),
                    'dir_play': np.expand_dims(ann_dir_play_split[i], axis=-1),
                },
                'labels': example_whole['labels']
            })
        return list_examples_split

    def _load_example_unsplit(self,
                              path_file_audio: Path,
                              df_ann: pd.DataFrame,
                              idx_duo: int,
                              idx_take: int,
                              idx_channel: int,
                              idx_instrument: int
                              ) -> dict:
        #  Figure out the labels and annotations
        #
        x_audio = tfio.audio.decode_wav(
            path_file_audio.read_bytes(), dtype=tf.int16)
        # trancode to float
        arr_audio = librosa.util.buf_to_float(
            x_audio.numpy(), dtype=np.float32)
        x_audio = tf.convert_to_tensor(arr_audio)
        #
        ann_t_play = df_ann.get('time').to_numpy()
        ann_x_play = df_ann.get(
            f'x{idx_channel}').to_numpy().astype(np.float32)
        ann_y_play = df_ann.get(
            f'y{idx_channel}').to_numpy().astype(np.float32)
        ann_dir_play = df_ann.get(
            f'zone{idx_channel}').map(_MAP_LABELS_PLAYING).to_numpy().astype(np.int8)
        # Resample the annotations to the audio sample rate
        ann_x_play = self._fit_annotations(
            ann_x_play, ann_t_play, x_audio)
        ann_y_play = self._fit_annotations(
            ann_y_play, ann_t_play, x_audio)
        ann_dir_play = self._fit_annotations(
            ann_dir_play, ann_t_play, x_audio)
        return {
            'audio': x_audio,
            'annotations': {
                'x_play': ann_x_play,
                'y_play': ann_y_play,
                'dir_play': ann_dir_play
            },
            'labels': {
                'idx_duo': np.int8(idx_duo),
                'idx_take': np.int8(idx_take),
                'idx_channel': np.int8(idx_channel),
                'idx_instrument': np.int8(idx_instrument)
            }
        }

    @ staticmethod
    def _extract_annotations(path_ann: Path) -> dict:
        """_summary_

        Args:
            path_ann (Path): _description_

        Returns:
            dict: _description_
        """
        df_ann = pd.read_csv(path_ann)
        dict_ann = dict()

        def _choose_duo_take(duo: int, take: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
            return lambda df: np.logical_and(df['duo'] == duo, df['take'] == take)
        nb_duos = df_ann['duo'].max()
        nb_takes = df_ann['take'].max()
        assert nb_duos <= _NB_DUOS
        assert nb_takes <= _NB_TAKES
        for duo in range(1, nb_duos+1):
            if duo not in dict_ann.keys():
                dict_ann[duo] = dict()
            for take in range(1, nb_takes+1):
                dict_ann[duo][take] = df_ann.loc[_choose_duo_take(duo, take)]
        return dict_ann

    @ staticmethod
    def _split_droplast(x: np.ndarray, size_blocks: int) -> np.ndarray:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            np.ndarray: _description_
        """
        # Taken from: https://stackoverflow.com/a/65507867
        nb_blocks = int(tf.size(x) / size_blocks)
        x_split = x[:nb_blocks * size_blocks]
        x_split = np.reshape(x_split, (nb_blocks, size_blocks))
        return x_split

    @ staticmethod
    def _fit_annotations(x_ann: np.ndarray, t_ann: np.ndarray, x_audio: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x_ann (np.ndarray): _description_
            t_ann (npt.NDArray[float]): _description_
            x_audio (npt.NDArray[_DTYPE_AUDIO]): _description_

        Returns:
            np.ndarray: _description_
        """
        # 'resample' annotation so that it matches audio sample frequency.
        x_ann_fit = x_ann.repeat(_RATE_AUDIO // _RATE_ANNOTATION)
        # add zeros before the first time of annotation.
        x_ann_fit = np.concatenate(
            (np.zeros(int(_RATE_AUDIO * t_ann[0]), dtype=x_ann_fit.dtype), x_ann_fit))
        x_ann_fit.resize(x_audio.shape)
        return x_ann_fit
