import argparse
import os
import torch
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--train_path', type=str, default=None)
parser.add_argument('--valid_path', type=str, default=None)
parser.add_argument('--seed',  type=int, default=43)
parser.add_argument('--use_gpu',  type=bool, default=True)
parser.add_argument('--load_data',  type=bool, default=False)
parser.add_argument('--latent_dim', type=int, default=4)
parser.add_argument('--num_epoch', type=int, default=100000)
parser.add_argument('--reset_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--diff_lr', type=float, default=1e-3)
parser.add_argument('--disc_lr', type=float, default=1e-4)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=256) 
parser.add_argument('--n_steps', type=int, default=1000)
parser.add_argument('--beta', type=int, default=0.95)
parser.add_argument('--guidance_w', type=int, default=0.5)
parser.add_argument('--lamda', type=int, default=10)
parser.add_argument('--coef', type=float, default=0.)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--save_model_epoch', type=int, default=1000)
parser.add_argument('--data_dir', type=str, default='data') 
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--log_name', type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
parser.add_argument('--model_path', type = str, default = os.path.join('model'))
parser.add_argument('--n_samples', type=int, default=5000)
parser.add_argument('--id', type=str, default='010')
parser.add_argument('--forward_weight', '-f', type=int, default=1.)
parser.add_argument('--backward_step', '-b',type=int, default=0)
parser.add_argument('--self_recurrent_step', '-r', type=int, default=1)
parser.add_argument('--cat_cols', nargs='*')
parser.add_argument('--pretrain_path', type=str, default=None)
parser.add_argument('--fewshot_path', type=str, default=None)

args = parser.parse_args()
args.log_interval = min(args.log_interval, args.num_epoch)
args.device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

args.log_path = os.path.join(args.log_dir, args.log_name)
args.model_path = os.path.join(args.log_path, 'models')
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)
args.save_data_path = os.path.join(args.log_path, 'results', 'data')
if not os.path.exists(args.save_data_path):
    os.makedirs(args.save_data_path)
    os.makedirs(os.path.join(args.log_path, 'results', 'figures'))


# 清理服务器内存：fuser -vk /dev/nvidia*