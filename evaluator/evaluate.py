import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns
import os
import numpy as np

def visual_correlation(data:dict, savefig=True, name='correlations'):
    # plot correlations in pair
    fig, axes = plt.subplots(nrows=2, ncols=len(data.keys()), figsize=(10*len(data.keys()), 10))
    fig.suptitle("Correlations")
    for i, key in enumerate(data.keys()):
        sns.heatmap(data=data[key][0].corr(min_periods=1), ax=axes[0, i], annot=False, vmin=0, vmax=1, xticklabels=False, yticklabels=False)
        axes[0, i].set_title(f'{key} Pretrain Data')
        sns.heatmap(data=data[key][1].corr(min_periods=1), ax=axes[1, i], annot=False, vmin=0, vmax=1, xticklabels=False, yticklabels=False)
        axes[1, i].set_title(f'{key} Finetune Data')
    if savefig:
        plt.savefig(f'{name}.png')
        plt.close()
    
def visual_correlation_row(data:dict, savefig=True, path='correlations', enable_labels=False):
    # plot given correlations in one row
    fig, axes = plt.subplots(nrows=1, ncols=len(data.keys()), figsize=(10*len(data.keys()), 10))
    fig.suptitle("Correlations")
    for i, key in enumerate(data.keys()):
        sns.heatmap(data=data[key].corr(min_periods=1), ax=axes[i], annot=False, vmin=0, vmax=1, xticklabels=enable_labels, yticklabels=enable_labels)
        axes[i].set_title(f'{key}')
    if savefig:
        plt.savefig(f'{path}')
        plt.close()
    return fig, axes

def get_range(data):
    mins, maxs = [[] for _ in range(len(data.keys()))], [[] for _ in range(len(data.keys()))]
    for i, key in enumerate(data.keys()):
        cur_data = data[key]
        for name in cur_data.columns:
            mins[i].append(cur_data[name].min())
            maxs[i].append(cur_data[name].max())
    mins, maxs = np.array(mins), np.array(maxs)
    return mins.min(axis=0), maxs.max(axis=0)

def visual_distribution(data:dict, savefig=True, path='distributions', enable_labels=False):
    cols = len(data.keys())
    rows = data[list(data.keys())[0]].shape[1]
    xmin, xmax = get_range(data)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10*cols, 10*rows))
    fig.suptitle('Distributions')
    for i, key in enumerate(data.keys()):
        cur_data = data[key]
        for j, name in enumerate(cur_data.columns):
            axes[j, i].hist(cur_data[name], bins=20, alpha=0.5, color='blue')
            axes[j, i].set_title(f'{key}-{name}')
            axes[j, i].set_xlim(xmin[j], xmax[j])
    plt.subplots_adjust(top=1)
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{path}')
        plt.close()
    return fig, axes


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='010')
    parser.add_argument('--log_name', type=str, default=None)
    args = parser.parse_args()
    args.log_path = os.path.join('logs', args.log_name)
    
    raw_full = pd.read_csv(f'../Datasets/diffusion_data/synther2col/synther{args.id}/fewshot_all.csv', index_col=None)
    raw_fine = pd.read_csv(f'../Datasets/diffusion_data/synther2col/synther{args.id}/fewshot.csv', index_col=None)
    
    pretrain = pd.read_csv(f'{args.log_path}/results/data/synthetic_wo_guidance.csv', index_col=None)
    finetune = pd.read_csv(f'{args.log_path}/results/data/synthetic.csv', index_col=None)
    
    visual_correlation_row({
        'Raw Data':raw_full,
        'Fewshot Data':raw_fine,
        'Pretrain Synthetic':pretrain,
        'Finetune Synthetic':finetune,
    }, path=f'{args.log_path}/results/figures/correlations.png', enable_labels=True)
    visual_distribution({
        'Raw Data':raw_full,
        'Fewshot Data':raw_fine,
        'Pretrain Synthetic':pretrain,
        'Finetune Synthetic':finetune,
    }, path=f'{args.log_path}/results/figures/distributions.png', enable_labels=True)