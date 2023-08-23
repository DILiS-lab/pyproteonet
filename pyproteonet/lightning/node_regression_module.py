from .abstract_node_regressor import AbstractNodeRegressor

import torch.nn.functional as F

class NodeRegressionModule(AbstractNodeRegressor):
    def __init__(
        self,
        model,
        nan_substitute_value: float = 0.0,
        mask_substitute_value: float = 0.0,
        hide_substitute_value: float = 0.0,
        lr: float = 0.001,
    ):
        super().__init__(
            nan_substitute_value=nan_substitute_value,
            mask_substitute_value=mask_substitute_value,
            hide_substitute_value=hide_substitute_value,
            lr=lr,
        )
        self._model = model

    @property
    def model(self):
        return self._model
    
    def calculate_loss(self, pred, target):
        return F.mse_loss(pred, target)
