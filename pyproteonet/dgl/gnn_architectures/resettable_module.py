from abc import abstractmethod

from torch import nn

class ResettableModule(nn.Module):

    @abstractmethod
    def reset_parameters():
        raise NotImplementedError()

