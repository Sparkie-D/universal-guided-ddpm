import torch
import torch.nn as nn
import pickle
import os
from tqdm import tqdm

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
        
    def train_epoch(self, neg_loader, pos_loader):
        loss
        for neg_data, pos_data in enumerate(zip(neg_loader, pos_loader)):
            pos_pred = self.model(pos_data)
            neg_pred = self.model(neg_data)
            
            ones = torch.ones_like(pos_pred)
            zeros = torch.zeros_like(neg_pred)
            loss = nn.CrossEntropyLoss(reduction='mean')(pos_pred, ones) + \
                   nn.CrossEntropyLoss(reduction='mean')(pos_pred, ones) + \
                   self._gradient_penalty(neg_data, pos_data)
            
        

    def train(self, batch_size=32, num_epoch=1000):  
        pos_loader = torch.utils.data.DataLoader(self.pos_data, batch_size=batch_size, shuffle=True)
        neg_loader = torch.utils.data.DataLoader(self.neg_data, batch_size=batch_size, shuffle=True)             
        for epoch in tqdm(range(num_epoch), desc='Training'):
            self.train_epoch(neg_loader, pos_loader)
                
            if self.model.save_model_epoch > 0 and (epoch + 1) % self.model.save_model_epoch == 0:
                with open(os.path.join(self.log_path, f'ddpm_{epoch}.pickle'), 'wb') as f:
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
        prob_interpolated = self.energy_model(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Return gradient penalty
        return LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()