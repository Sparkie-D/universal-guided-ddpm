import argparse
import os
import torch
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--algo', type=str, default='ddpm')
parser.add_argument('--train_path', type=str, default=None)
parser.add_argument('--valid_path', type=str, default=None)
parser.add_argument('--seed',  type=int, default=43)
parser.add_argument('--use_gpu',  type=bool, default=True)
parser.add_argument('--load_data',  type=bool, default=False)
parser.add_argument('--latent_dim', type=int, default=4)
parser.add_argument('--num_epoch', type=int, default=100000)
parser.add_argument('--reset_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--g_update_num', type=int, default=15)
parser.add_argument('--g_lr', type=float, default=1e-4)
parser.add_argument('--d_lr', type=float, default=1e-4)
parser.add_argument('--diff_lr', type=float,help='learning rate of diffusion.', default=1e-3)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=256) 
parser.add_argument('--n_steps', type=int, default=1000)
parser.add_argument('--extra_steps', type=int, default=100) # extra steps when generating
parser.add_argument('--beta', type=int, default=0.95)
parser.add_argument('--guidance_w', type=int, default=0.5)
parser.add_argument('--lamda', type=int, default=10)
parser.add_argument('--coef', type=float, default=0.)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--draw_interval', type=int, default=1000) # set to 0 if no plots
parser.add_argument('--save_model_epoch', type=int, help='epoch when model is saved. Set negative for not saving', default=1000)
parser.add_argument('--data_dir', type=str,help='Path to the datasets.', default='data') 
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--log_name', type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
parser.add_argument('--pretrain', type=bool,help='set True if the model can be loaded. (model path should be provided)' , default=False) # 是否从保存的模型中恢复
parser.add_argument('--predict', type=bool, default=False) # need to provide a valid model path if set True
parser.add_argument('--model_path', type = str, default = os.path.join('model'))
parser.add_argument('--datasets', nargs='+', type=str, default=(['pksim_5000.csv', 'lab_726.csv'])) 
parser.add_argument('--n_samples', type=int, default=5000)
parser.add_argument('--policy', type=str, default='cad')
parser.add_argument('--random_rate', type=float, default=0.3)
parser.add_argument('--threshold', type=int, default=1)

args = parser.parse_args()
args.features = []
args.log_interval = min(args.log_interval, args.num_epoch)
args.draw_interval = 0 if args.draw_interval == 0 else max(args.draw_interval, args.log_interval)
args.log_path = os.path.join(args.log_dir, args.algo, args.log_name)
args.figure_path = os.path.join(args.log_path, 'figures')
args.device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
if not os.path.exists(args.figure_path):
    os.makedirs(args.figure_path)
if not os.path.exists(os.path.join(args.log_path, 'results', 'data')):
    os.makedirs(os.path.join(args.log_path, 'results', 'data'))
    os.makedirs(os.path.join(args.log_path, 'results', 'figures'))


# 清理服务器内存：fuser -vk /dev/nvidia*