import os
import argparse
import pickle
import json
import pandas as pd
import numpy as np


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

def get_data(data, normalizer):
    x_num, x_cat, labels = data[normalizer.num_cols].values, data[normalizer.cat_cols].values, data[['y']].values
    data_normed = normalizer.normalize(x_num, x_cat, concat=True)
    return data_normed, labels

def square(data):
    return np.square(data)

def mean(data):
    return np.mean(data)

def get_eval(task_type='clf'):
    if task_type == 'clf':
        return KNeighborsClassifier(), lambda x,y : mean((x==y).astype(float))
    elif task_type == 'reg':
        return LinearRegression(), lambda x,y : mean(square(x-y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--task_type', type=str, default='clf')
    
    args = parser.parse_args()
    args.log_path = os.path.join('logs/', args.log_name, 'results', 'data')
    
    with open(f'{args.pretrain_path}/models/normalizer.pickle', 'rb') as f:
        normalizer=pickle.load(f)   
        
    if 'y' in normalizer.cat_cols:
        normalizer.cat_cols.remove('y')
        
    raw_full = pd.read_csv(f'{args.data_path}/fewshot_all.csv', index_col=None)
    fewshot_valid = pd.read_csv(f'{args.data_path}/fewshot_valid.csv', index_col=None)
    fewshot = pd.read_csv(f'{args.data_path}/fewshot.csv', index_col=None)
    synthetic = pd.read_csv(f'{args.log_path}/synthetic.csv', index_col=None)
    
    datasets = {
        'fewshot' : pd.read_csv(f'{args.data_path}/fewshot.csv', index_col=None),
        'synthetic' : pd.read_csv(f'{args.log_path}/synthetic.csv', index_col=None),
    }
    
    results = {}
    valid_x, valid_y = get_data(fewshot_valid, normalizer)
    for key in datasets.keys():
        x, y = get_data(datasets[key], normalizer)
        clf, loss_func = get_eval(args.task_type)
        clf.fit(x, y.ravel())
        y_pred = clf.predict(valid_x)
        results[key] = loss_func(y_pred, valid_y)

    for key in results.keys():
        if isinstance(results[key], np.ndarray):
            results[key] = results[key].tolist()       
                 
    save_path = os.path.join(args.log_path, 'mle.json')
    with open(save_path ,'w') as f:
        json.dump(results, f, sort_keys=False, indent=4)
    
    