
import tensorflow as tf

class PolyMeanSqueredError()
class PowerSpectrumLoss(tf.keras.losses.Loss):
    def __init__(self,
        frame_length: int,
        step_frame: int,
        fft_length: int = None,
        window_fn=tf.signal.hann_window,
        pad_end=False,
        reduction=tf.keras.losses.Reduction.AUTO,
        name=None
    ) -> None:
        super(PowerSpectrumLoss, self).__init__(reduction=reduction, name=name)
        def my_stft(x: tf.Tensor) -> tf.Tensor: 
            return tf.signal.stft(
                x,
                frame_length=frame_length,
                frame_step=step_frame,
                fft_length=fft_length,
                window_fn=window_fn,
                pad_end=pad_end
            )
        self._fn_stft = my_stft
        
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.keras.losses.mean_squared_error(
            self._fn_stft(y_pred), self._fn_stft(y_true)
        )
