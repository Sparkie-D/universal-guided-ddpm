import numpy as np
import torch


class GaussianNormalizer:
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, data):
        self.means = self.data.mean(axis=0)
        self.stds = self.data.std(axis=0) + 1e-5

    def normalize(self, x):                 
        if torch.is_tensor(x):
            device = x.device
            x = (x.cpu().numpy() - self.means) / self.stds
            return torch.as_tensor(x, device=device, dtype=torch.float)
        else:
            return ((x - self.means) / self.stds).astype(np.float32)

    def unnormalize(self, x):
        return x * self.stds + self.means


class MinMaxNormalizer:
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, data):
        self.maxs = data.max(axis=0).values
        self.mins = data.min(axis=0).values

    def normalize(self, x):
        if torch.is_tensor(x):
            device = x.device
            x = (x.cpu().numpy() - self.mins) / (self.maxs - self.mins + 1e-5)
            x = x * 2 - 1
            return torch.as_tensor(x, device=device, dtype=torch.float32)
        else:
            x =  (x - self.mins) / (self.maxs - self.mins + 1e-5)
            x = x * 2 - 1
            return x.astype(np.float32)

    def unnormalize(self, x):
        x = x.clip(-1, 1)
        x = (x + 1) / 2
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return x *  (self.maxs - self.mins + 1e-5) + self.mins


class SimLogNormalizer:
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, data):
        pass

    def normalize(self, x):                 
        if torch.is_tensor(x):
            device = x.device
            x = torch.sign(x) * torch.log(torch.abs(x) + 1)
            return torch.as_tensor(x, device=device)
        else:
            return np.sign(x) * np.log(np.abs(x) + 1)

    def unnormalize(self, x):
        if torch.is_tensor(x):
            device = x.device
            x = torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
            return torch.as_tensor(x, device=device)
        else:
            return np.sign(x) * (np.exp(np.abs(x)) - 1)


class MinMaxSimLogNormalizer:
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, data):
        simlog_data = np.sign(data.values) * np.log(np.abs(data.values) + 1)
        self.maxs = simlog_data.max(axis=0)
        self.mins = simlog_data.min(axis=0)

    def normalize(self, x):                 
        if torch.is_tensor(x):
            device = x.device
            x = torch.sign(x) * torch.log(torch.abs(x) + 1)
            x = (x.cpu().numpy() - self.mins) / (self.maxs - self.mins)
            x = x * 2 - 1
            return torch.as_tensor(x, device=device)
        else:
            x = np.sign(x) * np.log(np.abs(x) + 1) 
            x =  (x - self.mins) / (self.maxs - self.mins + 1e-5)
            x = x * 2 - 1
            return x

    def unnormalize(self, x):
        if torch.is_tensor(x):
            device = x.device
            x = (x + 1) / 2
            x = x.cpu().numpy() * (self.maxs - self.mins + 1e-5) + self.mins
            x = torch.as_tensor(x, device=device)
            x = torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
            return x
        else:
            x = (x + 1) / 2
            x = x * (self.maxs - self.mins + 1e-5) + self.mins
            x = np.sign(x) * (np.exp(np.abs(x)) - 1)
            return x
        

class DatasetNormalizer:
    def __init__(self, normalizer_dict):

        self.normalizers = normalizer_dict
        self.keys = list(normalizer_dict.keys())

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        assert key in self.keys
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        assert key in self.keys
        return self.normalizers[key].unnormalize(x)


