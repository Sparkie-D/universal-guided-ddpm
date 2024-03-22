from abc import abstractmethod
import torch


class BaseAlgorithm(object):
    def __init__(self, device=torch.device('cuda')) -> None:
        self.device = device


    @abstractmethod
    def update(self, data, mask):
        pass

    @abstractmethod
    def generate(self, batch_num):
        pass

    


