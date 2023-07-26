import collections

from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only

class DictLogger(Logger):
    '''
    Slightly modified version of:
    https://stackoverflow.com/questions/69276961/how-to-extract-loss-and-accuracy-from-logger-by-each-epoch-in-pytorch-lightning
    '''
    def __init__(self):
        super().__init__()
        self.hyperparameters = None
        self.logs = collections.defaultdict(list) # copy not necessary here  
        # The defaultdict in contrast will simply create any items that you try to access

    @property
    def name(self):
        return "DictLogger"

    @property
    def version(self):
        return "1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        self.hyperparameters = params

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        for metric_name, metric_value in metrics.items():
            self.logs[metric_name].append(metric_value)
        return