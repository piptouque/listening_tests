import tensorflow as tf

from .example_model import ExampleModel

from typing import Optional


class SaveManager:
    """Saving and loading model checkpoints
    """

    def __init__(self, config: object, model: ExampleModel, epoch_counter: Optional[tf.Variable] = None, **kwargs):
        self._config = config
        self._checkpoint = tf.train.Checkpoint(model, **kwargs)
        self._checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self._checkpoint,
            directory=self._config.path.checkpoint_dir,
            step_counter=epoch_counter,
            **vars(self._config.checkpoint)
        )

    def save(self) -> str:
        return self._checkpoint_manager.save()

    def load(self, save_path: str) -> bool:
        found = False
        try:
            self._checkpoint.restore(save_path)
            found = True
        except tf.errors.NotFoundError:
            found = False
        return found

    def load_last(self) -> bool:
        found = False
        if self._checkpoint_manager.latest_checkpoint:
            status = self._checkpoint.restore(
                self._checkpoint_manager.latest_checkpoint)
            try:
                status.assert_consumed()
            except AssertionError:
                found = False
            found = True
        return found
