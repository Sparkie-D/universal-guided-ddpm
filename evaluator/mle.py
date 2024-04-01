import os
import argparse
import pickle
import json
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


def get_data(data, normalizer):
    normalizer.cat_cols.remove('y')
    x_num, x_cat, labels = data[normalizer.num_cols], data[normalizer.cat_cols], data[['y']]
    data_normed = normalizer.normalize(x_num, x_cat, concat=True)
    return data_normed, labels

def get_eval(task_type='clf'):
    if task_type == 'clf':
        return KNeighborsClassifier(), lambda x,y : (x==y) / len(y)
    elif task_type == 'reg':
        return LinearRegression(), lambda x,y : (x-y)**2 / len(y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='010')
    parser.add_argument('--n_cols', type=str, default='2')
    parser.add_argument('--log_name', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--pretrain_path', type=str, default=None)
    
    args = parser.parse_args()
    args.log_path = os.path.join('logs/', args.log_name, 'results', 'data')
    
    with open(f'{args.pretrain_path}/models/normalizer.pickle', 'rb') as f:
        normalizer=pickle.load(f)   
        
    raw_full = pd.read_csv(f'{args.data_path}/fewshot_all.csv', index_col=None)
    fewshot_valid = pd.read_csv(f'{args.data_path}/fewshot_valid.csv', index_col=None)
    fewshot = pd.read_csv(f'{args.data_path}/fewshot.csv', index_col=None)
    synthetic = pd.read_csv(f'{args.log_path}/synthetic.csv', index_col=None)
    
    datasets = {
        'fewshot' : pd.read_csv(f'{args.data_path}/fewshot.csv', index_col=None),
        'synthetic' : pd.read_csv(f'{args.log_path}/synthetic.csv', index_col=None),
    }
    
    results = {}
    valid_x, valid_y = get_data()
    for key, data in datasets:
        x, y = get_data(data, normalizer)
        clf, loss_func = get_eval('clf')
        clf.fit(x, y)
        y_pred = clf.predict(valid_x)
        results[key] = loss_func(y_pred, valid_y)
    
    save_path = os.path.join(args.log_path, 'credits_fewshot.json')
    with open(save_path ,'w') as f:
        json.dump(results, f, sort_keys=False, indent=4)
    
    