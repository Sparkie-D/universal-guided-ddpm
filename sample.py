import numpy as np
import pandas as pd
import torch

from config import *
import pickle

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device='cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'

    with open(f'{args.pretrain_path}/models/ddpm.pickle', 'rb') as f:
        diffuser=pickle.load(f)
    with open(f'{args.pretrain_path}/models/normalizer.pickle', 'rb') as f:
        normalizer=pickle.load(f)    
    with open(f'{args.fewshot_path}/models/discriminator.pickle', 'rb') as f:
        disc=pickle.load(f)
        
    gen_data = diffuser.universal_guided_sample(batch_size=args.batch_size, 
                                                disc_model=disc,
                                                forward_weight=args.forward_weight,
                                                backward_step=args.backward_step,
                                                self_recurrent_step=args.self_recurrent_step, 
                                                n_samples=args.n_samples)
    gen_num = gen_data[:, :len(normalizer.num_cols)]
    gen_cat = gen_data[:, len(normalizer.num_cols):]
    generated = pd.DataFrame(data=normalizer.unnormalize(gen_num, gen_cat, concat=True),
                             columns=normalizer.num_cols+normalizer.cat_cols)

    # generated.drop(columns=[col for col in generated.columns if col.startswith('organ_lumen')], inplace=True)
    generated.to_csv(os.path.join(args.save_data_path, 'synthetic.csv'), index=None)
    
    if not os.path.exists(os.path.join(args.save_data_path, 'synthetic_wo_guidance.csv')):
        gen_data = diffuser.generate_wo_guidance(batch_size=args.batch_size,
                                                 n_samples=args.n_samples)
        gen_num = gen_data[:, :len(normalizer.num_cols)]
        gen_cat = gen_data[:, len(normalizer.num_cols):]
        generated = pd.DataFrame(data=normalizer.unnormalize(gen_num, gen_cat, concat=True),
                                columns=normalizer.num_cols+normalizer.cat_cols)
        
        # generated.drop(columns=[col for col in generated.columns if col.startswith('organ_lumen')], inplace=True)
        generated.to_csv(os.path.join(args.save_data_path, 'synthetic_wo_guidance.csv'), index=None)