import argparse
import json
import os
import pandas as pd
import numpy as np

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
              'Simpson diversity': {},
              'Shannon diveristy': {},
            }
    for col in data2.columns:
        # similarity
        result['wasserstein distance'][col] = wasserstein_distance(data1[col].values, data2[col].values)
        result['KS statistics'][col], _ = ks_2samp(data1[col].values, data2[col].values)
        result['Cos similarity'][col] = 1 - cosine(data1[col].values, data2[col].values)
        
        # diversity
        result['Simpson diversity'][col] = np.sum(np.square(data2[col].values / np.sum(data2[col].values)))
        result['Shannon diveristy'][col] = sample_diversity(data2[col].values)


    for key in result.keys():
        result[key]['mean'] = sum(result[key].values()) / len(result[key])
        for name in result[key].keys():
            if isinstance(result[key][name], np.ndarray):
                result[key][name] = result[key][name].tolist()
    
    with open(save_path ,'w') as f:
        json.dump(result, f, sort_keys=False, indent=4)
        
    return json.dumps(result, sort_keys=False, indent=4) # print this directly


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='010')
    parser.add_argument('--n_cols', type=str, default='2')
    parser.add_argument('--log_name', type=str, default=None)
    args = parser.parse_args()
    args.log_path = os.path.join('logs/', args.log_name, 'results', 'data')
    
    raw_full = pd.read_csv(f'../Datasets/diffusion/pksim_{args.n_cols}/{args.id}/fewshot_all.csv', index_col=None)
    pretrain = pd.read_csv(f'../Datasets/diffusion/pksim_{args.n_cols}/{args.id}/fewshot.csv')
    fewshot = pd.read_csv(f'{args.log_path}/synthetic.csv', index_col=None)
    
    numeric_difference(raw_full, pretrain, os.path.join(args.log_path, 'credits_pretrain.json'))
    numeric_difference(raw_full, fewshot, os.path.join(args.log_path, 'credits_fewshot.json'))