import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from config import *
from fewshot.discriminator import discriminator
from fewshot.trainer import Trainer
from utils.dataset import load_data
import pickle
from utils.dataset import DiffusionDataset

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    with open(f'{args.pretrain_path}/models/ddpm.pickle', 'rb') as f:
        diffuser=pickle.load(f)
    with open(f'{args.pretrain_path}/models/normalizer.pickle', 'rb') as f:
        normalizer=pickle.load(f)
    
    # with open('logs/fewshot010/models/discriminator.pickle', 'rb') as f:
    #     model=pickle.load(f)
    pos_data, columns=load_data(args.train_path, cat_cols=args.cat_cols)
    pos_data.set_normalizer(normalizer)
    valid_data, columns=load_data(args.valid_path, cat_cols=args.cat_cols)
    valid_data.set_normalizer(normalizer)
    logger = SummaryWriter(args.log_path)
    device='cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'

    gen_raw = diffuser.generate_wo_guidance(n_samples=10000)
    neg_data_np = normalizer.unnormalize(x_num=gen_raw[:, :len(normalizer.num_cols)], 
                                         x_cat=gen_raw[:, len(normalizer.num_cols):],
                                         concat=True)
    neg_data_df = pd.DataFrame(data=neg_data_np,columns=normalizer.num_cols+normalizer.cat_cols)
    neg_data = DiffusionDataset(neg_data_df, pos_data.normalizer)
    neg_data.set_normalizer(pos_data.normalizer)
    trainer = Trainer(
        model=discriminator(input_dim=pos_data.input_dim, 
                            hidden_dims=4,
                            device=device),
        # model=model,
        pos_data=pos_data,
        neg_data=neg_data,
        valid_data=valid_data,
        logger=logger,
        device=device,
        args=args
    )
    trainer.train(batch_size=args.batch_size,
                  num_epoch=args.num_epoch)
    