import numpy as np
import pandas as pd
from utils.normalizer import MinMaxNormalizer 

def load_data(path, cat_cols):
    data = pd.read_csv(path, index_col=None)
    normalizer = MinMaxNormalizer(data, cat_cols)
    return DiffusionDataset(data, normalizer), data.columns

class DiffusionDataset:
    def __init__(self, 
                 data:pd.DataFrame, 
                 normalizer:MinMaxNormalizer) -> None:
        self.data = data
        self.set_normalizer(normalizer)
        
    def __getitem__(self, i):
        return self.data_normed[i]
    
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        self.data_normed = self.normalizer.normalize(self.data[self.normalizer.num_cols].values, 
                                                self.data[self.normalizer.cat_cols].values,
                                                concat=True)
        self.input_dim = self.normalizer.normed_len
        
    def __len__(self):
        return self.data_normed.shape[0]