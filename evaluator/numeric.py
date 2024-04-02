import argparse
import json
import os
import pandas as pd
import numpy as np
import pickle

from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance, ks_2samp, entropy, kurtosis

def sample_diversity(samples):
    return entropy(np.histogram(samples, bins='auto')[0])

def shannon_diversity(data):
    total_sum = np.sum(data)
    proportions = data / total_sum
    shannon_index = -np.sum(proportions * np.log(proportions))
    return shannon_index

def numeric_difference(data1, data2, save_path):
    """
        data1: target
        data2: result
    """
    columns = [col for col in data1 if col in data2]
    data1 = data1[columns]
    data2 = data2.sample(n=data1.shape[0], replace=True)[columns] # leads to randomness
    result = {
              # similarity 
              'wasserstein distance': {}, 
              'KS statistics': {},
              'Cos similarity': {},
              # diversity
            #   'Simpson diversity': {},
              'Shannon diveristy': {},
            }
    for col in data2.columns:
        # similarity
        result['wasserstein distance'][col] = wasserstein_distance(data1[col].values, data2[col].values)
        result['KS statistics'][col], _ = ks_2samp(data1[col].values, data2[col].values)
        result['Cos similarity'][col] = 1 - cosine(data1[col].values, data2[col].values)
        
        # diversity
        # result['Simpson diversity'][col] = np.sum(np.square(data2[col].values / np.sum(data2[col].values)))
        result['Shannon diveristy'][col] = sample_diversity(data2[col].values)


    for key in result.keys():
        result[key]['mean'] = sum(result[key].values()) / len(result[key])
        for name in result[key].keys():
            if isinstance(result[key][name], np.ndarray):
                result[key][name] = result[key][name].tolist()
    
    with open(save_path ,'w') as f:
        json.dump(result, f, sort_keys=False, indent=4)
        
    return json.dumps(result, sort_keys=False, indent=4) # print this directly

def normalize_num(data, normalizr):
    num_data, cat_data = data[normalizer.num_cols].values, data[normalizer.cat_cols].values
    num_normed, cat_normed = normalizer.normalize(num_data, cat_data)
    data_normed = np.concatenate([num_normed, cat_data], axis=-1) 
    return pd.DataFrame(data_normed, columns=normalizer.num_cols+normalizer.cat_cols)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='010')
    parser.add_argument('--cat_cols', nargs='*')
    parser.add_argument('--log_name', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--pretrain_path', type=str, default=None)
    
    args = parser.parse_args()
    args.log_path = os.path.join('logs/', args.log_name, 'results', 'data')
    
    with open(f'{args.pretrain_path}/models/normalizer.pickle', 'rb') as f:
        normalizer=pickle.load(f)   
        
    raw_full = pd.read_csv(f'{args.data_path}/fewshot_all.csv', index_col=None)
    pretrain = pd.read_csv(f'{args.data_path}/fewshot.csv', index_col=None)
    fewshot = pd.read_csv(f'{args.log_path}/synthetic.csv', index_col=None)
    
    raw_full = normalize_num(raw_full, normalizer)
    pretrain = normalize_num(pretrain, normalizer)
    fewshot = normalize_num(fewshot, normalizer)
    
    columns = [col for col in fewshot.columns if col not in args.cat_cols]
    numeric_difference(raw_full[columns], pretrain[columns], os.path.join(args.log_path, 'credits_pretrain.json'))
    numeric_difference(raw_full[columns], fewshot[columns], os.path.join(args.log_path, 'credits_fewshot.json'))