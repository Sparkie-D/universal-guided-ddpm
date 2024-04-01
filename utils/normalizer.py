import numpy as np
import torch

class MinMaxNormalizer:
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, data, cat_cols):
        self.num_cols = [col for col in data.columns if col not in cat_cols]
        self.cat_cols = cat_cols
        self.cat_dict = {cat:data[cat].unique() for cat in self.cat_cols}    
        self.cat_len = sum([len(self.cat_dict[cat]) for cat in cat_cols])
        self.maxs = data[self.num_cols].max(axis=0).values
        self.mins = data[self.num_cols].min(axis=0).values
        self.normed_len = len(self.num_cols) + sum([len(self.cat_dict[col]) for col in self.cat_cols])
    
    def normalize(self, x_num, x_cat, concat=False):
        # x_num
        if torch.is_tensor(x_num):
            device = x_num.device
            x_num = (x_num.cpu().numpy() - self.mins) / (self.maxs - self.mins + 1e-5)
            x_num = torch.as_tensor(x_num, device=device, dtype=torch.float32)
            x_num = x_num * 2 - 1
            
            x_cat_ret = []
            device = x_cat.device
            for i, col in enumerate(self.cat_cols):
                cur = x_cat[:, i].cpu().numpy().reshape(-1, 1)
                cur = np.repeat(cur, axis=-1, repeats=len(self.cat_dict[col]))
                x_cat_ret.append((cur == self.cat_dict[col]))
                
            x_cat = np.concatenate(x_cat_ret, axis=-1)
            x_cat = torch.from_numpy(x_cat).to(device).astype(torch.float32)
            
            if concat:
                return torch.cat([x_num, x_cat], dim=-1).astype(torch.float32)
            else:
                return x_num, x_cat
            
        elif isinstance(x_num, np.ndarray):
            x_num =  (x_num - self.mins) / (self.maxs - self.mins + 1e-5)
            x_num = x_num * 2 - 1
            
            x_cat_ret = []
            for i, col in enumerate(self.cat_cols):
                cur = x_cat[:, i].reshape(-1, 1)
                cur = np.repeat(cur, axis=-1, repeats=len(self.cat_dict[col]))
                cur = (cur == self.cat_dict[col]).astype(np.float32)
                x_cat_ret.append(cur)
                
            x_cat = np.concatenate(x_cat_ret, axis=-1)
            
            if concat:
                return np.concatenate([x_num, x_cat], axis=-1).astype(np.float32)
            else:
                return x_num, x_cat
        else:
            raise NotImplementedError

    def unnormalize(self, 
                    x_num, 
                    x_cat, 
                    concat=False) -> np.ndarray:
        '''
        unnormalize maps data into original space, which contains str type
        '''
        if isinstance(x_num, torch.Tensor):
            x_num = x_num.detach().cpu().numpy()
            x_cat = x_cat.detach().cpu().numpy()
        x_num = x_num.clip(-1, 1)
        x_num = (x_num + 1) / 2
        x_num = x_num *  (self.maxs - self.mins + 1e-5) + self.mins

        start = 0
        cat_ret = []
        for col in self.cat_cols:
            cat_len = len(self.cat_dict[col])
            idx = np.argmax(x_cat[:, start:start+cat_len], axis=-1)
            cat_ret.append(self.cat_dict[col][idx].reshape(-1, 1))
            start += cat_len
        x_cat = np.concatenate(cat_ret, axis=-1)
        if concat:
            return np.concatenate([x_num, x_cat], axis=-1)
        else:
            return x_num, x_cat 
    
    
if __name__ == '__main__':
    import pandas as pd
    # data = pd.DataFrame({
    #     'a':[0,0,1,1],
    #     'aa':[1,2,2,0],
    #     'aaa':['Amy', 'Bob', 'Cat', 'Cat'],
    #     'aaaa':['ICLR', 'ICML', 'AAAI','ICLR'],
    #     'b':[2,3,4,2.5],
    #     'bb':[1,2,3,4]
    # })
    data = pd.DataFrame({
        'a':['ICLR'],
        'b':[2.5],
        'bb':[4]
    })
    # print(data)
    cat_cols = ['a']
    num_cols = [col for col in data if col not in cat_cols]
    print(data[num_cols+cat_cols])
    
    norm = MinMaxNormalizer(data, cat_cols=cat_cols)
    x_num, x_cat = norm.normalize(data[num_cols].values, data[cat_cols].values)
    # print(np.concatenate([x_num, x_cat], axis=-1))
    # print(x_num)
    
    x_num, x_cat = torch.from_numpy(x_num), torch.from_numpy(x_cat)
    print(norm.unnormalize(x_num, x_cat, concat=True))