from abc import ABC
import torch


class Callback(ABC):
    def on_batch_start(self):
        pass

    def on_epoch_start(self):
        pass

    def on_train_start(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_end(self):
        pass

    def on_train_end(self):
        pass

    def add_model(self):
        pass



class ModelCheckpoint(Callback):
    def __init__(self, path, monitor='val_loss', mode='min'):
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf')

    def add_model(self, model):
        self.model = model

    def on_train_start(self):
        torch.save(self.model.state_dict(), self.path)
        print(f"Model saved at {self.path}")

    def on_epoch_end(self, epoch, logs):
        if self.mode == 'min':
            if logs[self.monitor] < self.best:
                self.best = logs[self.monitor]
                torch.save(self.model.state_dict(), self.path)
        elif self.mode == 'max':
            if logs[self.monitor] > self.best:
                self.best = logs[self.monitor]
                torch.save(self.model.state_dict(), self.path)

        print(f"Model saved at {self.path}")

