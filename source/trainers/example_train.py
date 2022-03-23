import tensorflow as tf
from tqdm import tqdm

from source.models.example_model import ExampleModel
from source.utils.logger import Logger
from source.models.save_manager import SaveManager


class ExampleTrainer:
    def __init__(self, config: object,  model: ExampleModel, logger: Logger, train_data: tf.Tensor, test_data: tf.Tensor, ):
        self._model = model
        self._train_data = train_data
        self._test_data = test_data
        self._config = config.training
        self.optimizer = tf.optimizers.Adam(
            learning_rate=self._config.learning_rate)

        self._train_loss = tf.keras.metrics.Mean(name='train_loss')
        self._train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        self._test_loss = tf.keras.metrics.Mean(name='test_loss')
        self._test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

        self.epoch_counter = tf.Variable(0, trainable=False)

        self.save_manager = SaveManager(
            config=config.save,
            model=model,
            epoch_counter=self.epoch_counter,
            optimizer=self.optimizer)
        self._logger = logger

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self._model(x, training=True)
            loss = self._model.loss_object(y, predictions)
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self._model.trainable_variables))

        self._train_loss(loss)
        self._train_accuracy(y, predictions)

    def train_epoch(self, epoch: int):
        loop = tqdm(range(self._config.num_iter_per_epoch))
        for _, (batch_x, batch_y) in zip(loop, self._train_data):
            self.train_step(batch_x, batch_y)
        self.epoch_counter.assign(epoch)

        summaries_dict = {
            'loss': {
                'type': 'scalar',
                'value': self._train_loss.result(),
            },
            'acc': {
                'type': 'scalar',
                'value': self._train_accuracy.result()
            }
        }

        self._logger.summarize(epoch, summaries_dict=summaries_dict)
        self.save_manager.save()

    @tf.function
    def test_step(self, x, y):
        predictions = self._model(x, training=False)
        t_loss = self._model.loss_object(y, predictions)

        self._test_loss(t_loss)
        self._test_accuracy(y, predictions)

    def test_epoch(self, epoch: int):
        for (batch_x, batch_y) in self._test_data:
            self.test_step(batch_x, batch_y)

    def train(self):
        template_train = "Epoch: {} | Train Loss: {}, Train Accuracy: {}"
        template_test = "         | Test Loss:  {}, Test Accuracy: {}"
        epochs = self._config.num_epochs
        for epoch in range(1, epochs+1):
            self._train_loss.reset_states()
            self._train_accuracy.reset_states()
            self._test_loss.reset_states()
            self._test_accuracy.reset_states()

            self.train_epoch(epoch)
            self.test_epoch(epoch)
