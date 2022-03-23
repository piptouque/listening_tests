import tensorflow as tf
import os


class Logger:
    """Logging using Tensorboard.
    """

    def __init__(self, config: object):
        self._config = config
        # self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"), self.sess.graph)
        # self.test_summary_writer = tf.summary.FileWriter( os.path.join(self.config.summary_dir, "test"))
        self.train_summary_writer = tf.summary.create_file_writer(
            os.path.join(self._config.path.summary_dir, "train"))
        self.test_summary_writer = tf.summary.create_file_writer(
            os.path.join(self._config.path.summary_dir, "test"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        if summaries_dict is not None:
            with summary_writer.as_default(step=step):
                for tag, el in summaries_dict.items():
                    getattr(tf.summary, el['type'])(tag, el['value'])
            summary_writer.flush()
