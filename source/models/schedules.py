
import tensorflow as tf

class TransformerLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
        size_model: int,
        nb_steps_warmup: int
    ) -> None:
        super(TransformerLearningRateSchedule, self).__init__()
        self._size_mdel = size_model
        self._nb_steps_warmup = nb_steps_warmup

    def __call__(self, step: int) -> float:
        m_1 = tf.math.rsqrt(step)
        m_2 = step * (self._nb_steps_warmup ** -1.5)
        rate_learning = tf.math.rsqrt(self.size_model) * tf.math.minimum(m_1, m_2)
        return rate_learning