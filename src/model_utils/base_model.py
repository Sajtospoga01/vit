import tensorflow as tf
import keras
from keras.models import Model
import itertools


class ModelGenerator(Model):
    """
    Custom model class with custom training loop
    """
    # metrics = None

    def __init__(self, *args, **kwargs):
        super(ModelGenerator, self).__init__(*args, **kwargs)

        self.n_classes = None
        self.base_model = None
        self.levels = None
        self.loss_fn = None
        self.optimizer = None
        self.acc_metric = keras.metrics.CategoricalAccuracy(name="accuracy")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.eval_acc_metric = keras.metrics.CategoricalAccuracy(name="val_accuracy")
        self.eval_loss_tracker = keras.metrics.Mean(name="val_loss")
        self.eval_recall = keras.metrics.Recall(name="val_recall")
        self.eval_precision = keras.metrics.Precision(name="val_precision")
        self.backup_logs = None

    def get_backup_logs(self):
        """
        returns the eval metric values from the model
        """
        return self.backup_logs

    def compile(self, *args, **kwargs):
        """
        Compiles the model
        """
        self.loss_fn = kwargs["loss"]
        self.optimizer = kwargs["optimizer"]
        kwargs["metrics"] = kwargs["metrics"] + [
            self.acc_metric,
            self.loss_tracker,
            self.eval_acc_metric,
            self.eval_loss_tracker,
            self.eval_recall,
            self.eval_precision,
        ]
        super(ModelGenerator, self).compile(*args, **kwargs)

    def save(self, *args, **kwargs):
        """
        saves the model
        """
        super(ModelGenerator, self).save(*args, **kwargs)

    @tf.function
    def train_step(self, data):
        """
        Custom training step, capable of handling weights
        """
        if len(data) == 3:
            x, y, w = data
        else:
            x, y = data
            w = None
        y = tf.convert_to_tensor(y)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss

            if w is None:
                loss = self.compiled_loss(y, y_pred)
            else:
                loss = self.compiled_loss(y, y_pred, w)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(y, y_pred)
        return self.loss_tracker.result(), self.acc_metric.result()

    @tf.function
    def eval_step(self, data):
        """
        Custom evaluation step
        """
        x, y = data
        y = tf.convert_to_tensor(y)

        y_pred = self(x, training=False)
        tf.convert_to_tensor(y_pred)
        loss = self.compiled_loss(y, y_pred)
        loss = tf.reduce_mean(loss)

        self.eval_loss_tracker.update_state(loss)
        self.eval_acc_metric.update_state(y, y_pred)
        self.eval_recall.update_state(y, y_pred)
        self.eval_precision.update_state(y, y_pred)

        return self.eval_loss_tracker.result(), self.eval_acc_metric.result()

    def train(
            self,
            dataset,
            epochs=1,
            batch_size=32,
            learning_rate=1e-2,
            steps_per_epoch=512,
            validation_dataset=None,
            validation_steps=50,
            callbacks=[],
    ):
        """
        Custom training loop
        """
        self.optimizer.learning_rate = learning_rate
        self.val_data = validation_dataset
        logs = {}
        metrics = [
            self.loss_tracker,
            self.acc_metric,
            self.eval_loss_tracker,
            self.eval_acc_metric,
            self.eval_recall,
            self.eval_precision,
        ]
        for callback in callbacks:
            callback.set_model(self)
            callback.set_params({"epochs": epochs, "verbose": 1})

        for callback in callbacks:  # on train begin callbacks
            callback.on_train_begin()

        for epoch in range(epochs):
            for callback in callbacks:  # on epoch begin callbacks
                callback.on_epoch_begin(epoch)

            tf.print(f"\nEpoch: {epoch+1}/{epochs}")

            pbar = tf.keras.utils.Progbar(
                target=steps_per_epoch, stateful_metrics=["time_to_complete"]
            )

            for dataset_idx, data in enumerate(
                    itertools.islice(dataset, steps_per_epoch)
            ):
                # for batch_idx, mini_batch in enumerate(data):

                loss, accuracy = self.train_step(data)
                pbar.update(
                    dataset_idx + 1, values=[("loss", loss), ("accuracy", accuracy)]
                )
                for callback in callbacks:  # on batch end callbacks
                    callback.on_train_batch_end(dataset_idx)

            pbar = tf.keras.utils.Progbar(target=validation_steps)
            if not validation_dataset is None:
                validation_dataset.set_mini_batch_size(batch_size)
                tf.print("Performing validation")
                validation_cycle = itertools.cycle(validation_dataset)
                for dataset_idx, data in enumerate(
                        itertools.islice(validation_cycle, validation_steps)
                ):
                    loss, accuracy = self.eval_step(data)
                    pbar.update(
                        dataset_idx + 1,
                        values=[
                            ("val_loss", loss),
                            ("val_accuracy", accuracy),
                            ("val_recall", self.eval_recall.result()),
                            ("val_precision", self.eval_precision.result()),
                        ],
                        )

            for metric in metrics:
                if hasattr(metric, "result"):
                    logs[metric.name] = metric.result().numpy()

                    metric.reset_states()
                else:
                    logs[metric.name] = metric.numpy()
            self.backup_logs = logs.copy()
            for callback in callbacks:  # on epoch end callbacks
                callback.on_epoch_end(epoch, logs=logs)
            # dataset.on_epoch_end()
        for callback in callbacks:  # on train end callbacks
            callback.on_train_end()