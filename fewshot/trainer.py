import torch
import torch.nn as nn
import pickle
import os
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler

class Trainer(object):
    def __init__(self, 
                 model,
                 pos_data,
                 neg_data, 
                 logger,
                 args,
                 device,
                 ) -> None:
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.logger = logger
        self.device = device
        self.model = model
        self.log_path = args.log_path
        self.save_model_epoch = args.save_model_epoch
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=args.disc_lr)
        
    def train_epoch(self, neg_loader, pos_loader):
        total_loss = 0
        for i, (neg_data, pos_data) in enumerate(zip(neg_loader, pos_loader)):
            pos_data = pos_data.to(self.device)
            neg_data = neg_data.to(self.device)
            pos_pred = self.model(pos_data)
            neg_pred = self.model(neg_data)
            
            ones = torch.ones_like(pos_pred)
            zeros = torch.zeros_like(neg_pred)
            loss = nn.CrossEntropyLoss(reduction='mean')(pos_pred, ones) + \
                   nn.CrossEntropyLoss(reduction='mean')(neg_pred, zeros) + \
                   self._gradient_penalty(neg_data, pos_data)
                
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += loss.item()
            
        return total_loss
            
        

    def train(self, batch_size=32, num_epoch=1000):  
        sampler = WeightedRandomSampler(weights=torch.ones(len(self.pos_data)), num_samples=batch_size, replacement=True)
        pos_loader = torch.utils.data.DataLoader(self.pos_data, batch_size=batch_size, sampler=sampler)
        neg_loader = torch.utils.data.DataLoader(self.neg_data, batch_size=batch_size, shuffle=True)             
        for epoch in tqdm(range(num_epoch), desc='Training'):
            loss = self.train_epoch(neg_loader, pos_loader)
            
            self.logger.add_scalar('train/loss', loss, epoch)
            
            if self.save_model_epoch > 0 and (epoch + 1) % self.save_model_epoch == 0:
                with open(os.path.join(self.log_path, f'disc.pickle'), 'wb') as f:
                    pickle.dump(self.model, f)

   
    def _gradient_penalty(self, real_data, generated_data, LAMBDA=10):
        batch_size = real_data.size()[0]

        # Calculate interpolationsubsampling_rate=20
        alpha = torch.rand(batch_size, 1).requires_grad_()
        alpha = alpha.expand_as(real_data).to(self.device)
        # print(alpha.shape, real_data.shape, generated_data.shape)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data

        # Calculate probability of interpolated examples
        # print(self.device, self.energy_model.device)
        prob_interpolated = self.model(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Return gradient penalty
        return LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()