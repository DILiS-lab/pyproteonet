from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class TrainingEarlyStopping(EarlyStopping):

    def on_validation_end(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        self._run_early_stopping_check(trainer)