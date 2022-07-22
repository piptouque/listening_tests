"""canonne_duos dataset."""

from pathlib import Path
import re
import dataclasses
from typing import Any, Callable, List, Optional

import librosa
import pandas as pd
import numpy as np
# import numpy.typing as npt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

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
  langid = {english}
}
"""
_HOMEPAGE = None
_URL_DOWNLOAD = None

_DO_BUILD_DESCRIPTOR = False
_DTYPE_AUDIO = tf.dtypes.float32


def _next_pow_2(a): return int(2 ** np.ceil(np.log2(a)))


_NB_DUOS = 10
_MAX_NB_TAKES = 4
_NB_CHANNELS = 2

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


_RATE_ANNOTATION = 4


_MAP_LABEL_INSTRUMENTS = {
    'piano': 1,
    'prepared-piano': 2,
    'bass-guitar': 3,
    'electric-guitar': 4,
    'double-bass': 5,
    'cello': 6,
    'viola': 7,
    'voice': 8,
    'flute': 9,
    'duduk': 10,
    'bass-clarinet': 11,
    'voice-clarinet': 12,
    'saxophone': 13,
    'baryton-saxophone': 14,
    'tenor-saxophone': 15,
    'alto-saxophone': 16,
    'trumpet': 17,
    'drums': 18,
    'electronics': 19,
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

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        # pylint: disable=wrong-keyword-args
        CanonneDuosConfig(
            name='joined',
            description='Tracks from the same take and duo are joined in a single example',
            kind_split='joined'
        )
    ]
    # pylint: enable=wrong-keyword-args
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
                'rate_annotation': _RATE_ANNOTATION,
                'size_window': _SIZE_WINDOW_DESCRIPTOR,
                'stride_window': _STRIDE_WINDOW_DESCRIPTOR
            }),
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'audio': tfds.features.Audio(shape=(_SIZE_EXAMPLE_AUDIO, _NB_CHANNELS), file_format='wav', sample_rate=_RATE_AUDIO, dtype=tf.float32),
                'descriptors': dict(zip(
                    _NAMES_DESCRIPTORS,
                    [tfds.features.Tensor(shape=(_SIZE_BLOCK_DESCRIPTOR, _NB_CHANNELS),
                                          dtype=tf.dtypes.float32) for _ in range(len(_NAMES_DESCRIPTORS))]
                )),
                'annotations': {
                    'x_play': tfds.features.Tensor(shape=(_SIZE_EXAMPLE_AUDIO, _NB_CHANNELS), dtype=tf.dtypes.float32),
                    'y_play': tfds.features.Tensor(shape=(_SIZE_EXAMPLE_AUDIO, _NB_CHANNELS), dtype=tf.dtypes.float32),
                    'dir_play': tfds.features.Tensor(shape=(_SIZE_EXAMPLE_AUDIO, _NB_CHANNELS), dtype=tf.dtypes.int8),
                },
                'labels': {
                    'idx_duo': tfds.features.ClassLabel(num_classes=_NB_DUOS+1),
                    'idx_take': tfds.features.ClassLabel(num_classes=_MAX_NB_TAKES+1),
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
        pat_dir_duo = re.compile(_PATTERN_DIR_DUO)
        pat_dir_take = re.compile(_PATTERN_DIR_TAKE)
        ann_dict = self._extract_annotations(
            path_ann / 'dt_duo_positions.csv')
        for path_dir_duo in path_audio.iterdir():
            if not path_dir_duo.is_dir():
                continue
            m_dir_duo = pat_dir_duo.match(path_dir_duo.name)
            if m_dir_duo is None:
                continue
            g_dir_duo = m_dir_duo.groupdict()
            idx_dir_duo = int(g_dir_duo['duo'])
            for path_dir_take in path_dir_duo.iterdir():
                if not path_dir_take.is_dir():
                    continue
                m_dir_take = pat_dir_take.match(path_dir_take.name)
                if m_dir_take is None:
                    continue
                g_dir_take = m_dir_take.groupdict()
                idx_dir_take = int(g_dir_take['take'])
                list_channel_examples = [None] * _NB_CHANNELS
                for path_file_audio in path_dir_take.iterdir():
                    if not path_file_audio.is_file():
                        continue
                    example_whole = self._load_example_unsplit(
                        path_file_audio,
                        ann_dict=ann_dict,
                        idx_dir_duo=idx_dir_duo,
                        idx_dir_take=idx_dir_take)
                    if example_whole is None:
                        continue
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
                            list_descriptors = [
                                np.zeros(
                                    _SIZE_BLOCK_DESCRIPTOR, dtype=dict_example['audio'].dtype)
                                for _ in range(len(_NAMES_DESCRIPTORS))
                            ]
                        dict_example['descriptors'] = dict(zip(_NAMES_DESCRIPTORS, [
                            np.expand_dims(descr, axis=-1) for descr in list_descriptors]))
                    # add to total channels
                    idx_channel = example_whole['labels']['idx_channel']
                    list_channel_examples[idx_channel - 1] = list_example_split
                nb_splits = len(list_channel_examples[0])
                assert np.all([len(channel_examples) == nb_splits for channel_examples in list_channel_examples])
                # Swap splits <-> channels in list of lists' dimensions.
                list_channel_examples_swapped = [
                    [[] for _b in range(_NB_CHANNELS)] for _a in range(nb_splits)]
                for channel, list_examples_split in enumerate(list_channel_examples):
                    assert list_examples_split is not None, f'Example Duo {idx_dir_duo} / Take {idx_dir_take} -- Channel {channel+1} was not found!'
                    for idx_split, example in enumerate(list_examples_split):
                        list_channel_examples_swapped[idx_split][channel] = example
                # merge all channels in a single example of each split
                list_examples_merged = [{
                    'audio': np.stack([example['audio'] for example in list_examples], axis=-1),
                    'annotations': {
                        'x_play': np.stack([example['annotations']['x_play'] for example in list_examples], axis=-1),
                        'y_play': np.stack([example['annotations']['y_play'] for example in list_examples], axis=-1),
                        'dir_play': np.stack([example['annotations']['dir_play'] for example in list_examples], axis=-1),
                    },
                    'labels': {
                        'idx_duo': list_examples[0]['labels']['idx_duo'],
                        'idx_take': list_examples[0]['labels']['idx_take'],
                        'idx_channel': np.stack([example['labels']['idx_channel'] for example in list_examples], axis=-1),
                        'idx_instrument': np.stack([example['labels']['idx_instrument'] for example in list_examples], axis=-1)
                    },
                    'descriptors': dict(zip(_NAMES_DESCRIPTORS, [
                        np.concatenate(
                            [example['descriptors'][name_descriptor] for example in list_examples], axis=-1
                        )
                        for name_descriptor in _NAMES_DESCRIPTORS
                    ]))
                } for list_examples in list_channel_examples_swapped]
                for idx_split, list_example_split in enumerate(list_examples_merged):
                    yield f'{path_dir_duo.name}_{path_dir_take.name}_{idx_split}', list_example_split

    @classmethod
    def _split_example(cls, example_whole: dict) -> List[dict]:
        x_audio_split = cls._split_droplast(
            example_whole['audio'], _SIZE_EXAMPLE_AUDIO)
        ann_x_play_split = cls._split_droplast(
            example_whole['annotations']['x_play'], _SIZE_EXAMPLE_AUDIO)
        ann_y_play_split = cls._split_droplast(
            example_whole['annotations']['y_play'], _SIZE_EXAMPLE_AUDIO)
        ann_dir_play_split = cls._split_droplast(
            example_whole['annotations']['dir_play'], _SIZE_EXAMPLE_AUDIO)
        list_examples_split = []
        for i in range(x_audio_split.shape[0]):
            list_examples_split.append({
                'audio': x_audio_split[i],
                'annotations': {
                    'x_play': ann_x_play_split[i],
                    'y_play': ann_y_play_split[i],
                    'dir_play': ann_dir_play_split[i],
                },
                'labels': example_whole['labels']
            })
        return list_examples_split

    @classmethod
    def _load_example_unsplit(cls,
                              path_file_audio: Path,
                              ann_dict: dict[str, Any],
                              idx_dir_duo: int,
                              idx_dir_take: int,
                              ) -> Optional[dict[str, Any]]:

        pat_file_audio = re.compile(_PATTERN_FILE_AUDIO)
        if not path_file_audio.is_file():
            return None
        m_file = pat_file_audio.match(path_file_audio.name)
        if m_file is None:
            return None
        #  Figure out the labels and annotations
        g_file = m_file.groupdict()
        label_instrument = g_file['instrument']
        idx_instrument = _MAP_LABEL_INSTRUMENTS[label_instrument]
        # ids have to be [2, max], idx=0 is a placeholder
        idx_duo = int(g_file['duo'])
        idx_take = int(g_file['take'])
        idx_channel = int(g_file['channel'])
        df_ann = ann_dict[idx_duo][idx_take]
        # some safety check
        assert idx_duo == idx_dir_duo,   "Inconsistent naming/directory scheme"
        assert idx_take == idx_dir_take, "Inconsistent naming/directory scheme"
        #
        x_audio = tfio.audio.decode_wav(
            path_file_audio.read_bytes(), dtype=tf.int16)
        # trancode to float
        x_audio = librosa.util.buf_to_float(
            x_audio.numpy(), dtype=np.float32)
        #
        ann_t_play = df_ann.get('time').to_numpy()
        ann_x_play = df_ann.get(
            f'x{idx_channel}').to_numpy().astype(np.float32)
        ann_y_play = df_ann.get(
            f'y{idx_channel}').to_numpy().astype(np.float32)
        ann_dir_play = df_ann.get(
            f'zone{idx_channel}').map(_MAP_LABELS_PLAYING).to_numpy().astype(np.int8)
        # Resample the annotations to the audio sample rate
        ann_x_play = cls._fit_annotations(
            ann_x_play, ann_t_play, x_audio)
        ann_y_play = cls._fit_annotations(
            ann_y_play, ann_t_play, x_audio)
        ann_dir_play = cls._fit_annotations(
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
        assert nb_takes <= _MAX_NB_TAKES
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

    @classmethod
    def compute_audio_descriptors(
            cls,
            x_audio: np.ndarray,
            *,
            rate_sample: int,
            size_win: int,
            stride_win: int,
            freq_min: float,
            freq_max: float) -> list[np.ndarray]:
        # librosa expects time as last axis
        x_audio = np.transpose(x_audio)
        x_descriptor = cls.compute_audio_descriptors_numpy(
            x_audio,
            rate_sample=rate_sample,
            size_win=size_win,
            stride_win=stride_win,
            freq_min=freq_min,
            freq_max=freq_max
        )
        x_descriptor = np.transpose(x_descriptor)
        return x_descriptor

    @staticmethod
    def compute_audio_descriptors_numpy(
        x_audio: np.ndarray,
        *,
        rate_sample: int,
        size_win: int,
        stride_win: int,
        freq_min: float,
        freq_max: float,
    ) -> list[np.ndarray]:

        # librosa does not compute frame_length
        # as time // stride
        # so we need to remove an element (arbitrarily, the last one).
        # see: https://github.com/librosa/librosa/issues/1278
        x_spec = librosa.stft(
            y=x_audio,
            n_fft=size_win,
            win_length=size_win,
            hop_length=stride_win,
            center=True
        )
        x_spec = x_spec[..., :-1]

        x_freq_0, _, x_prob_freq = librosa.pyin(
            y=x_audio,
            fmin=freq_min,
            fmax=freq_max,
            sr=rate_sample,
            frame_length=size_win,
            hop_length=stride_win,
            center=True
        )
        x_freq_0 = x_freq_0.astype(x_audio.dtype)[..., :-1]
        x_prob_freq = x_prob_freq.astype(x_audio.dtype)[..., :-1]

        # Functions in librosa.feature add a dimension
        # in second-to-last position.
        # we remove it to match the other features.
        x_rms = librosa.feature.rms(
            S=np.abs(x_spec),
            frame_length=size_win,
            hop_length=stride_win,
            center=True
        ).astype(x_audio.dtype)
        x_rms = np.squeeze(x_rms, axis=-2)

        x_zcr = librosa.feature.zero_crossing_rate(
            y=x_audio,
            frame_length=size_win,
            hop_length=stride_win,
            center=True
        ).astype(x_audio.dtype)
        x_zcr = np.squeeze(x_zcr, axis=-2)[..., :-1]

        x_flatness = librosa.feature.spectral_flatness(
            S=np.abs(x_spec),
        ).astype(x_audio.dtype)
        x_flatness = np.squeeze(x_flatness, axis=-2)

        x_centroid = librosa.feature.spectral_centroid(
            S=np.abs(x_spec)
        ).astype(x_audio.dtype)
        x_centroid = np.squeeze(x_centroid, axis=-2)

        return [
            x_freq_0,
            x_prob_freq,
            x_rms,
            x_zcr,
            x_flatness,
            x_centroid
        ]
