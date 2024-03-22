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
    
    pos_data, columns=load_data(args.train_path)
    logger = SummaryWriter(args.log_path)
    device='cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'

    with open('logs/ddpm/2col_pretrain/ddpm_9999.pickle') as f:
        diffuser=pickle.load(f)
    neg_data_np = diffuser.generate_wo_guidance(n_samples=len(pos_data))
    trainer = Trainer(
        model=discriminator(input_dim=args.input_dim, 
                            hidden_dims=2,
                            device=device),
        pos_data=pos_data,
        neg_data=DiffusionDataset(neg_data_np, pos_data.normalizer),
        logger=logger,
        device=device,
        args=args
    )

    # generated = pd.DataFrame(data=train_data.normalizer.unnormalize(trainer.model.generate(n_samples=args.n_samples)), columns=columns)
    # generated.to_csv(os.path.join(args.log_path, 'results', 'data', 'generated.csv'), index=None)
    