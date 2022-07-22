"""canonne_trios dataset."""

import re
from pathlib import Path
import dataclasses
from typing import Any, Optional

import librosa
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_io as tfio

# TODO(canonne_trios): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(canonne_trios): BibTeX citation
_CITATION = """
@article{goupil_emergent_2021,
	title = {Emergent Shared Intentions Support Coordination During Collective Musical Improvisations},
	volume = {45},
	issn = {0364-0213, 1551-6709},
	url = {https://onlinelibrary.wiley.com/doi/10.1111/cogs.12932},
	doi = {10.1111/cogs.12932},
	number = {1},
	journaltitle = {Cognitive Science},
	shortjournal = {Cogn Sci},
	author = {Goupil, Louise and Wolf, Thomas and Saint‐Germier, Pierre and Aucouturier, Jean‐Julien and Canonne, Clément},
	urldate = {2022-02-28},
	date = {2021-01},
	langid = {english},
	keywords = {Musicology, {SocialSciences}},
}
"""
_HOMEPAGE = None
_URL_DOWNLOAD = None

_DO_BUILD_DESCRIPTOR = False
_DTYPE_AUDIO = tf.dtypes.float32


def _next_pow_2(a): return int(2 ** np.ceil(np.log2(a)))


_NB_TRIOS = 12
_MAX_NB_TAKES = 16
_NB_CHANNELS = 3

# in Hertz
_RATE_AUDIO = 48000
#
_SIZE_WINDOW_DESCRIPTOR = 1024
_STRIDE_WINDOW_DESCRIPTOR = 256
_FREQ_MIN_DESCRIPTOR = 40
_FREQ_MAX_DESCRIPTOR = 12000

_NAMES_DESCRIPTORS = [
    'freq_0', 'prob_freq',
    'root_mean_square', 'zero_crossing_rate',
    'spectral_flatness', 'spectral_centroid'
]
# Wanted length (in seconds) for a single example.
# Will be increased to the next power of two.
_LENGTH_BLOCK_DESIRED = 2
# better to have a power of two.
_SIZE_EXAMPLE_AUDIO = _next_pow_2(_RATE_AUDIO * _LENGTH_BLOCK_DESIRED)
_SIZE_BLOCK_DESCRIPTOR = _SIZE_EXAMPLE_AUDIO // _STRIDE_WINDOW_DESCRIPTOR

_LABELS_INSTRUMENTS = [
    'piano',
    'prepared-piano',
    'bass-guitar',
    'electric-guitar',
    'double-bass',
    'cello',
    'viola',
    'voice',
    'flute',
    'duduk',
    'bass-clarinet',
    'voice-clarinet',
    'saxophone',
    'baryton-saxophone',
    'tenor-saxophone',
    'alto-saxophone',
    'trumpet',
    'drums',
    'electronics',
]
# Source: Supplementary Materials and Results, S.1.1
_MAP_TRIO_MUSICIANS: dict[int, tuple[int, int, int]] = {
    1: (1, 2, 3),
    2: (4, 5, 2),
    3: (6, 4, 3),
    4: (7, 8, 9),
    5: (9, 5, 10),
    6: (7, 6, 11),
    7: (8, 12, 13),
    8: (13, 14, 15),
    9: (13, 16, 17),
    10: (18, 19, 20),
    11: (1, 21, 20),
    12: (11, 17, 18)
}

_MAP_MUSICIAN_INSTRUMENTS: dict[int, list[str]] = {
    1: ['bass-clarinet'],
    2: ['alto-saxophone'],
    3: ['drums'],
    4: ['voice-clarinet'],
    5: ['prepared-piano'],
    6: ['tenor-saxophone'],
    7: ['electric-guitar'],
    8: ['electronics'],
    9: ['drums'],
    10: ['trumpet'],
    11: ['drums'],
    12: ['prepared-piano'],
    13: ['baryton-saxophone', 'duduk'],
    14: ['alto-saxophone', 'piano'],
    15: ['alto-saxophone'],
    16: ['double-bass'],
    17: ['alto-saxophone'],
    18: ['flute'],
    19: ['soprano-saxophone'],
    20: ['double-bass'],
    21: ['trumpet']
}

_PATTERN_FILE_AUDIO = r'trio(?P<trio>\d+)_take(?P<take>\d+)_member(?P<member>\d+).wav'
_PATTERN_DIR_TRIO = r'booth_(?P<trio>\d+)'
_PATTERN_DIR_TAKE = r'take_(?P<take>\d+)'


@dataclasses.dataclass
class CanonneTriosConfig(tfds.core.BuilderConfig):
    kind_split: str = 'joined'


class CanonneTrios(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for canonne_trios dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        # pylint: enable=wrong-keyword-args
        CanonneTriosConfig(
            name='joined',
            description='Tracks from the same take and trio are joined in a single example',
            kind_split='joined'
        )
    ]
    # pytype: enable=wrong-keyword-args
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    TODO
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            metadata=tfds.core.MetadataDict({
                'rate_sample': _RATE_AUDIO,
                'size_window': _SIZE_WINDOW_DESCRIPTOR,
                'stride_window': _STRIDE_WINDOW_DESCRIPTOR
            }),
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'audio': tfds.features.Audio(
                    shape=(_SIZE_EXAMPLE_AUDIO, _NB_CHANNELS),
                    file_format='wav',
                    sample_rate=_RATE_AUDIO, dtype=_DTYPE_AUDIO
                ),
                'descriptors': dict(zip(
                    _NAMES_DESCRIPTORS,
                    [tfds.features.Tensor(
                        shape=(_SIZE_BLOCK_DESCRIPTOR, _NB_CHANNELS),
                        dtype=_DTYPE_AUDIO)
                     for _ in range(len(_NAMES_DESCRIPTORS))]
                )),
                'labels': {
                    'idx_trio': tfds.features.ClassLabel(num_classes=_NB_TRIOS+1),
                    'idx_take': tfds.features.ClassLabel(num_classes=_MAX_NB_TAKES+1),
                    'flags_instrument': tfds.features.Tensor(
                        shape=(len(_LABELS_INSTRUMENTS), _NB_CHANNELS),
                        dtype=tf.bool
                    )
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
        # TODO(canonne_trios): Downloads the data and defines the splits
        if dl_manager.manual_dir is None:
            path = dl_manager.download_and_extract(_URL_DOWNLOAD)
        else:
            path = dl_manager.manual_dir
        return {
            'train': self._generate_examples(
                path_audio=path / 'audio' / 'train'
            ),
            'test': self._generate_examples(
                path_audio=path / 'audio' / 'test'
            )
        }

    def _generate_examples(self, path_audio: Path):
        """Yields examples."""
        pat_dir_trio = re.compile(_PATTERN_DIR_TRIO)
        pat_dir_take = re.compile(_PATTERN_DIR_TAKE)
        for path_dir_trio in path_audio.iterdir():
            m_dir_trio = pat_dir_trio.match(path_dir_trio.name)
            if m_dir_trio is None:
                continue
            g_dir_trio = m_dir_trio.groupdict()
            idx_dir_trio = g_dir_trio['trio']
            for path_dir_take in path_dir_trio.iterdir():
                if not path_dir_take.is_dir():
                    continue
                m_dir_take = pat_dir_take.match(path_dir_take.name)
                if m_dir_take is None:
                    continue
                g_dir_take = m_dir_take.groupdict()
                idx_dir_take = int(g_dir_take['take'])
                list_channel_examples = [None] * _NB_CHANNELS
                ###
                for path_file_audio in path_dir_trio.iterdir():
                    if not path_file_audio.is_file():
                        continue
                    example_whole = self._load_example_unsplit(
                        path_file_audio,
                        idx_dir_trio=idx_dir_trio,
                        idx_dir_take=idx_dir_take)
                    #  Divide the audio into blocks of required length
                    list_example_split = self._split_example(
                        example_whole)
                    # compute descriptors
                    for dict_example in list_example_split:
                        if _DO_BUILD_DESCRIPTOR:
                            list_descriptors = self.compute_audio_descriptors(
                                dict_example['audio'],
                                rate_sample=_RATE_AUDIO,
                                size_win=_SIZE_WINDOW_DESCRIPTOR,
                                stride_win=_STRIDE_WINDOW_DESCRIPTOR,
                                freq_min=_FREQ_MIN_DESCRIPTOR,
                                freq_max=_FREQ_MAX_DESCRIPTOR
                            )
                        else:
                            # Bogus data
                            list_descriptors = [
                                np.zeros(
                                    _SIZE_BLOCK_DESCRIPTOR, dtype=dict_example['audio'].dtype)
                                for _ in range(len(_NAMES_DESCRIPTORS))
                            ]
                        dict_example['descriptors'] = dict(zip(_NAMES_DESCRIPTORS, [
                            np.expand_dims(descr, axis=-1) for descr in list_descriptors]))
                    idx_channel = idx_dir_trio
                    # add to total channels
                    list_channel_examples[idx_channel-1] = list_example_split
                nb_splits = len(list_channel_examples[0])
                assert np.all(
                    [len(channel_examples) == nb_splits for channel_examples in list_channel_examples])
                # Swap splits <-> channels in list of lists' dimensions.
                list_channel_examples_swapped = [
                    [[] for _b in range(_NB_CHANNELS)] for _a in range(nb_splits)]
                for channel, list_examples_split in enumerate(list_channel_examples):
                    assert list_examples_split is not None, f'Example Trio {idx_dir_trio} / Take {idx_dir_take} -- Channel {channel+1} was not found!'
                    for idx_split, example in enumerate(list_examples_split):
                        list_channel_examples_swapped[idx_split][channel] = example
                # merge all channels in a single example of each split
                list_examples_merged = [{
                    'audio': np.stack([example['audio'] for example in list_examples], axis=-1),
                    'labels': {
                        'idx_trio': list_examples[0]['labels']['idx_duo'],
                        'idx_take': list_examples[0]['labels']['idx_take'],
                        'flags_instrument': np.stack([example['labels']['instrument']['flags'] for example in list_examples], axis=-1)
                    },
                    'descriptors': dict(zip(_NAMES_DESCRIPTORS, [
                        np.concatenate(
                            [example['descriptors'][name_descriptor] for example in list_examples], axis=-1
                        )
                        for name_descriptor in _NAMES_DESCRIPTORS
                    ]))
                } for list_examples in list_channel_examples_swapped]
                for idx_split, list_example_split in enumerate(list_examples_merged):
                    yield f'{path_dir_trio.name}_{path_dir_take.name}_{idx_split}', list_example_split

    @staticmethod
    def _try_load_example_unsplit(path_file_audio: Path,
                                  idx_dir_trio: int,
                                  idx_dir_take: int,
                                  ) -> Optional[dict[str, Any]]:
        pat_file_audio = re.compile(_PATTERN_FILE_AUDIO)
        if not path_file_audio.is_file():
            return None
        m_file = pat_file_audio.match(
            path_file_audio.name)
        if m_file is None:
            return None
        #  Figure out the labels and annotations
        g_file = m_file.groupdict()
        idx_trio = g_file['trio']
        idx_take = g_file['take']
        idx_take = int(g_file['take'])
        idx_member = g_file['member']
        # some safety checking
        assert idx_trio == idx_dir_trio
        assert idx_take == idx_dir_take
        # idx_booth starts at 1, _MAP_* booth starts at 0
        idx_musician = _MAP_TRIO_MUSICIANS[idx_trio][idx_member-1]
        labels_instruments = _MAP_MUSICIAN_INSTRUMENTS[idx_musician]
        #
        # ids have to be [2, max], idx=0 is a placeholder
        #  Figure out the labels and annotations
        #
        x_audio = tfio.audio.decode_wav(
            path_file_audio.read_bytes(), dtype=tf.int16)
        # trancode to float
        x_audio = librosa.util.buf_to_float(
            x_audio.numpy(), dtype=_DTYPE_AUDIO)
        flags_instruments = np.isin(_LABELS_INSTRUMENTS, labels_instruments)
        #
        return {
            'audio': x_audio,
            'labels': {
                'idx_trio': np.int8(idx_trio),
                'idx_take': np.int8(idx_take),
                'instruments': {
                    'flag': flags_instruments
                }
            }
        }

    @classmethod
    def _split_example(cls, example_whole: dict) -> list[dict]:
        x_audio_split = cls._split_droplast(
            example_whole['audio'], _SIZE_EXAMPLE_AUDIO)
        list_examples_split = []
        for i in range(x_audio_split.shape[0]):
            list_examples_split.append({
                'audio': x_audio_split[i],
                'labels': example_whole['labels']
            })
        return list_examples_split

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
