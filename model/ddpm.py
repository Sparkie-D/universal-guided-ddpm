import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

from model.base import BaseAlgorithm
from model.diffusion import MLPDiffusion
# from config import GENDER_INDEX


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


class DDPM(BaseAlgorithm):
    def __init__(self, 
                 num_layers=5,
                 input_dim=1,
                 hidden_dim=1024,
                 n_steps=1000,
                 diff_lr=1e-3,
                 save_model_epoch=100,
                 device=torch.device('cuda:0')) -> None:
        super().__init__(device=device)
        self.diffuser = MLPDiffusion(input_dim=input_dim, n_steps=n_steps, num_layers=num_layers, hidden_dim=hidden_dim).to(self.device)
        self.diff_optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=diff_lr)
        self.n_steps = n_steps
        # self.betas = cosine_beta_schedule(self.n_steps)
        betas = torch.linspace(-6, 6, self.n_steps)
        betas = torch.sigmoid(betas).to(self.device)
        self.betas = betas * (0.5e-2 - 1e-5) + 1e-5  
        self.input_dim=input_dim
        
        self.alphas = 1 - self.betas			
        self.alphas_prod = torch.cumprod(self.alphas, 0).to(self.device)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float().cuda(), self.alphas_prod[:-1]], 0)  
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        self.save_model_epoch = save_model_epoch
        self.name = 'ddpm'
    
    def update(self, data: torch.Tensor) -> None:
        batch_size = data.shape[0]

        t = torch.randint(0, self.n_steps, size=(batch_size // 2,)).to(self.device)
        t = torch.cat([t, self.n_steps - 1 - t], dim=0)
        if (batch_size % 2) != 0:
            t = torch.randint(0, self.n_steps, size=(batch_size,)).to(self.device)     					 
        t = t.unsqueeze(-1)

        a = self.alphas_bar_sqrt[t]
        aml = self.one_minus_alphas_bar_sqrt[t]
        epsilon = torch.randn_like(data).to(self.device)

        xt = data * a + epsilon * aml
        output = self.diffuser(xt, t.squeeze(-1))
        loss = (epsilon - output).square().mean()

        self.diff_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffuser.parameters(), 1.)		
        self.diff_optimizer.step()
        
        return {'diffusion loss': loss.item()}
    
    def sample_one_step(self, x, t):
        coeff = self.betas[t] / self.one_minus_alphas_bar_sqrt[t]
        diffuser_t = torch.LongTensor([t]).to(self.device)
        eps_theta = self.diffuser(x, diffuser_t)
        mean = (1 / (1 - self.betas[t]).sqrt()) * (x - (coeff * eps_theta))
        z = torch.randn_like(x)
        sigma_t = self.betas[t].sqrt()
        sample = mean + sigma_t * z
        return sample

    def generate_wo_guidance(self, batch_num=400, extra_step=100, n_samples=0):            
        x = np.random.randn(batch_num, self.input_dim)

        for start in range(0, n_samples, batch_num):
            end = min(start+batch_num, batch_num)
            batch_x = torch.tensor(x[start:end], dtype=torch.float32).to(self.device)
            for i in range(self.n_steps-1, -extra_step, -1):
                t = max(i, 0)
                batch_x = self.sample_one_step(batch_x, t)
            x[start:end] = batch_x.cpu().detach().numpy()
        
        return x

    def generate(self, batch_num):
        return super().generate(batch_num)