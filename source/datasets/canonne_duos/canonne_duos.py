"""canonne_duos dataset."""

from pathlib import Path
import re
from typing import Callable, Tuple, List

import pandas as pd
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio


_URL_DOWNLOAD = "https://todo-data-url"
# TODO(canonne_duos): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(canonne_duos): BibTeX citation
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

_NB_DUOS = 10
_NB_TAKES = 4
_NB_INTERPRETS = 2

_RATE_AUDIO = 48000
_SIZE_BLOCKS = _RATE_AUDIO * 10
_DTYPE_AUDIO = tf.int16

_RATE_ANNOTATION = 4

_LABEL_INSTRUMENTS = [
    'piano',
    'electric-guitar',
    'bass-guitar',
    'viola',
    'cello',
    'double-bass',
    'voice',
    'trumpet',
    'saxophone',
]

_MAP_LABELS_PLAYING = {
    "avec": 1,
    "sans": 0,
    "contre": -1
}

_PATTERN_AUDIOFILE = r'Duo(?P<duo>\d+)_(?P<take>\d+)_(?P<interpret>\d+)_(?P<instrument>\w+).wav'


class CanonneDuos(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for canonne_duos dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    TODO
    """
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(canonne_duos): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'audio': tfds.features.Audio(shape=(None,), file_format='wav', sample_rate=_RATE_AUDIO),
                'annotation': {
                    'x_play': tfds.features.Tensor(shape=(None,), dtype=tf.dtypes.float32),
                    'y_play': tfds.features.Tensor(shape=(None,), dtype=tf.dtypes.float32),
                    'dir_play': tfds.features.Tensor(shape=(None,), dtype=tf.dtypes.int8),
                },
                'label': {
                    'idx_duo': tfds.features.ClassLabel(num_classes=_NB_DUOS),
                    'idx_take': tfds.features.ClassLabel(num_classes=_NB_TAKES),
                    'idx_interpret': tfds.features.ClassLabel(num_classes=_NB_INTERPRETS),
                    'instrument': tfds.features.ClassLabel(names=_LABEL_INSTRUMENTS)
                }
                # 'label': tfds.features.ClassLabel(names=['no', 'yes']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('audio', 'label'),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
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
        pat_file = re.compile(_PATTERN_AUDIOFILE)
        ann_dict = self._extract_annotations(
            path_ann / 'dt_duo_positions.csv')
        for p in path_audio.iterdir():
            if p.is_file():
                m_file = pat_file.match(p.name)
                if m_file is not None:
                    #
                    audio = tfio.audio.decode_wav(
                        p.read_bytes(), dtype=_DTYPE_AUDIO)
                    #  Figure out the annotations
                    g_file = m_file.groupdict()
                    label_instrument = g_file['instrument']
                    # ids have to be [0, max-1]
                    idx_duo = int(g_file['duo'])-1
                    idx_take = int(g_file['take'])-1
                    idx_interpret = int(g_file['interpret'])-1
                    df_ann = ann_dict[idx_duo+1][idx_take+1]
                    #
                    t_play = df_ann.get('time').to_numpy()
                    x_play = df_ann.get(
                        f'x{idx_interpret+1}').to_numpy().astype(np.float32)
                    y_play = df_ann.get(
                        f'y{idx_interpret+1}').to_numpy().astype(np.float32)
                    dir_play = df_ann.get(
                        f'zone{idx_interpret+1}').map(_MAP_LABELS_PLAYING).to_numpy().astype(np.int8)
                    # Resample the annotations to the audio sample rate
                    x_play = self._fit_annotations(x_play, t_play, audio)
                    y_play = self._fit_annotations(y_play, t_play, audio)
                    dir_play = self._fit_annotations(dir_play, t_play, audio)
                    #  Split the audio into blocks of required length
                    audio_split = self._split_droplast(
                        audio, _SIZE_BLOCKS)
                    x_play_split = self._split_droplast(
                        x_play, _SIZE_BLOCKS)
                    y_play_split = self._split_droplast(
                        y_play, _SIZE_BLOCKS)
                    dir_play_split = self._split_droplast(
                        dir_play, _SIZE_BLOCKS)
                    for i in range(audio_split.shape[0]):
                        yield f'{p.name}_{i}', {
                            'audio': audio_split[i],
                            'annotation': {
                                'x_play': x_play_split[i],
                                'y_play': y_play_split[i],
                                'dir_play': dir_play_split[i]
                            },
                            'label': {
                                'idx_duo': idx_duo,
                                'idx_take': idx_take,
                                'idx_interpret': idx_interpret,
                                'instrument': label_instrument
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
    def _fit_annotations(x_ann: np.ndarray, t_ann: npt.NDArray[float], x_audio: npt.NDArray[_DTYPE_AUDIO]) -> np.ndarray:
        """_summary_

        Args:
            x_ann (np.ndarray): _description_
            t_ann (npt.NDArray[float]): _description_
            x_audio (npt.NDArray[_DTYPE_AUDIO]): _description_

        Returns:
            np.ndarray: _description_
        """
        x_ann_fit = x_ann.repeat(_RATE_AUDIO // _RATE_ANNOTATION)
        x_ann_fit = np.concatenate(
            (np.zeros(int(_RATE_AUDIO * t_ann[0]), dtype=x_ann_fit.dtype), x_ann_fit))
        x_ann_fit.resize(x_audio.shape)
        return x_ann_fit
