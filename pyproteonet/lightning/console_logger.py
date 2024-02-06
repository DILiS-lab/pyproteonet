import collections
from abc import ABCMeta

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import Logger


class ConsoleLogger(Logger):
    def __init__(self, name: str = 'Logger', version: str = "1.0", keep_dict: bool = True) -> None:
        super().__init__()
        self._name = name
        self._version = version
        self.keep_dict = keep_dict
        self.logs = collections.defaultdict(list)

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def experiment(self):
        return None

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        print_line = f"step{step}: "
        for name, value in metrics.items():
            print_line += f"{name}:{value:.3f} || "
        print(print_line)
        if self.keep_dict:
            for metric_name, metric_value in metrics.items():
                self.logs[metric_name].append(metric_value)

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        pass
