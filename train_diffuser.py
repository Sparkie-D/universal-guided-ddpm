import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.tensorboard import SummaryWriter

from config import *
from model.ddpm import DDPM
from utils.trainer import Trainer
from utils.dataset import load_data


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    train_data, columns=load_data(args.train_path)
    valid_data, _=load_data(args.valid_path)
    with open(os.path.join(args.model_path, 'normalizer.pickle'), 'wb') as f:
        pickle.dump(train_data.normalizer, f)
    logger = SummaryWriter(args.log_path)
    device='cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    trainer = Trainer(
        model=DDPM(num_layers=args.num_layers,
                   input_dim=train_data.input_dim,
                   hidden_dim=args.hidden_dim,
                   n_steps=args.n_steps,
                   diff_lr=args.diff_lr,
                   save_model_epoch=args.save_model_epoch,
                   device=device
                   ),
        train_data=train_data,
        valid_data=valid_data,
        logger=logger,
        device=device,
        args=args
    )
    trainer.train(num_epoch=args.num_epoch)
    