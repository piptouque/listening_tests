import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os

from source.utils.utils import get_project_root

class PlantsTrainer:
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config


        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics=['accuracy']

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    def train(self):
# %%
        root_path = get_project_root()
        check_path = os.path.join(root_path, "experiments", self.config["exp_name"],
                                    "checkpoint", self.config["checkpoint_name"])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path,
                                                        verbose=1, save_best_only=True)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

        history = self.model.fit(self.train_data,
                            epochs=self.config["num_epochs"],
                            callbacks = [checkpoint, callback],
                            validation_data=self.val_data)
        return history


