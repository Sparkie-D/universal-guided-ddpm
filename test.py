import pandas as pd
import pickle

if __name__ == '__main__':
    # with open(f'logs/pretrain/column#all/models/ddpm.pickle', 'rb') as f:
    #     diffuser=pickle.load(f)
    with open(f'logs/pretrain/column#all/models/normalizer.pickle', 'rb') as f:
        normalizer=pickle.load(f)    
    # with open(f'logs/fewshot/column#all_id#3050/models/discriminator.pickle', 'rb') as f:
    #     disc=pickle.load(f)
    
    raw_full = pd.read_csv(f'../Datasets/diffusion/pksim_all/3050/fewshot_all.csv', index_col=None)
    pretrain = pd.read_csv(f'../Datasets/diffusion/pksim_all/3050/fewshot.csv', index_col=None)
    
    full_num, full_cat = normalizer.normalize(raw_full[normalizer.num_cols].values, raw_full[normalizer.cat_cols].values)
    full = normalizer.unnormalize(full_num, full_cat, concat=True)
    full = pd.DataFrame(data=full, columns=normalizer.num_cols+normalizer.cat_cols)
    full.to_csv(f'../Datasets/diffusion/pksim_all/3050/fewshot_all_norm.csv', index=None)
    
    pre_num, pre_cat = normalizer.normalize(pretrain[normalizer.num_cols].values, pretrain[normalizer.cat_cols].values)
    pre = normalizer.unnormalize(pre_num, pre_cat, concat=True)
    pre = pd.DataFrame(data=pre, columns=normalizer.num_cols+normalizer.cat_cols)
    pre.to_csv(f'../Datasets/diffusion/pksim_all/3050/fewshot_norm.csv', index=None)