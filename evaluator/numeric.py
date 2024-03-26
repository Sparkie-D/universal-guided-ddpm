import argparse
import json
import os
import pandas as pd

from scipy.stats import wasserstein_distance, entropy, ks_2samp

def numeric_difference(data1, data2):
    result = {'wasserstein distance':{}, 
            #   'entropy':{}, 
              'KS statistics':{}
            }
    for col in data1.columns:
        result['wasserstein distance'][col] = wasserstein_distance(data1[col].values, data2[col].values)
        # result['entropy'][col] = entropy(data1[col].values, data2[col].values) # data must have same n_samples
        result['KS statistics'][col], _ = ks_2samp(data1[col].values, data2[col].values)
    for key in result.keys():
        result[key]['mean'] = sum(result[key].values()) / len(result[key])
    return json.dumps(result, sort_keys=False, indent=4) # print this directly


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='010')
    parser.add_argument('--log_name', type=str, default=None)
    args = parser.parse_args()
    args.log_path = os.path.join('logs/ddpm', args.log_name, 'results')
    
    raw_full = pd.read_csv(f'../Datasets/diffusion_data/synther2col/synther{args.id}/fewshot_all.csv', index_col=None)
    
    pretrain = pd.read_csv(f'{args.log_path}/results/data/synthetic_wo_guidance.csv', index_col=None)
    finetune = pd.read_csv(f'{args.log_path}/results/data/synthetic.csv', index_col=None)
    
    print(numeric_difference(raw_full, pretrain))
    print(numeric_difference(raw_full, finetune))