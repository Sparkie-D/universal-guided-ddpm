import numpy as np
import pandas as pd
from utils.normalizer import MinMaxNormalizer 

def load_data(path):
    data = pd.read_csv(path, index_col=None)
    normalizer = MinMaxNormalizer(data)
    return DiffusionDataset(data, normalizer), data.columns

class DiffusionDataset:
    def __init__(self, data, normalizer) -> None:
        self.data = data
        self.normalizer = normalizer
        self.input_dim = data.shape[1]
        
    def __getitem__(self, i):
        if isinstance(self.data, pd.DataFrame):
            return self.normalizer.normalize(self.data.values[i])
        elif isinstance(self.data, np.ndarray):
            return self.normalizer.normalize(self.data[i])
    
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        
    def __len__(self):
        return self.data.shape[0]