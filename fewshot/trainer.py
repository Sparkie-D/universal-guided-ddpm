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
                 valid_data, 
                 logger,
                 args,
                 device,
                 ) -> None:
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.valid_data = valid_data
        self.logger = logger
        self.device = device
        self.model = model
        self.log_path = args.log_path
        self.model_path = args.model_path
        self.save_model_epoch = args.save_model_epoch
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=args.disc_lr)
        
    def train_epoch(self, neg_loader, pos_loader):
        total_loss = 0
        n_batch = 0
        for i, (neg_data, pos_data) in enumerate(zip(neg_loader, pos_loader)):
            pos_data = pos_data.to(self.device)
            neg_data = neg_data.to(self.device)
            pos_pred = self.model(pos_data)
            neg_pred = self.model(neg_data)
            
            learner_loss = -torch.mean(torch.log(1 - torch.sigmoid(neg_pred)))
            expert_loss = -torch.mean(torch.log(torch.sigmoid(pos_pred)))
            
            loss = learner_loss + expert_loss + self._gradient_penalty(neg_data, pos_data, LAMBDA=0)
            # loss = learner_loss + expert_loss
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += loss.item()
            n_batch += 1
            
        return total_loss / n_batch
            
        

    def train(self, batch_size=32, num_epoch=1000):  
        sampler = WeightedRandomSampler(weights=torch.ones(len(self.pos_data)), num_samples=batch_size, replacement=True)
        pos_loader = torch.utils.data.DataLoader(self.pos_data, batch_size=batch_size, sampler=sampler)
        neg_loader = torch.utils.data.DataLoader(self.neg_data, batch_size=batch_size, shuffle=True)             
        for epoch in tqdm(range(num_epoch), desc='Training'):
            loss = self.train_epoch(neg_loader, pos_loader)
            
            self.logger.add_scalar('train/loss', loss, epoch)
            
            self.eval_epoch(batch_size, epoch)
            
            if self.save_model_epoch > 0 and (epoch + 1) % self.save_model_epoch == 0:
                with open(os.path.join(self.model_path, f'discriminator.pickle'), 'wb') as f:
                    pickle.dump(self.model, f)
    
    def eval_epoch(self, batch_size, epoch):
        sampler = WeightedRandomSampler(weights=torch.ones(len(self.pos_data)), num_samples=batch_size, replacement=True)
        pos_loader = torch.utils.data.DataLoader(self.valid_data, batch_size=batch_size, sampler=sampler)
        neg_loader = torch.utils.data.DataLoader(self.neg_data, batch_size=batch_size, shuffle=True) 
        pos_pred, neg_pred = [], []
        
        for i, (neg_data, pos_data) in enumerate(zip(neg_loader, pos_loader)):
            pos_data = pos_data.to(self.device)
            neg_data = neg_data.to(self.device)
            pos_pred.append(self.model(pos_data).squeeze())
            neg_pred.append(self.model(neg_data).squeeze())
        
        pos_pred = torch.sigmoid(torch.cat(pos_pred))
        neg_pred = torch.sigmoid(torch.cat(neg_pred))
        # print(pos_pred, neg_pred)
        self.logger.add_histogram('eval/pos_prediction', pos_pred, epoch)
        self.logger.add_histogram('eval/neg_prediction', neg_pred, epoch)            

   
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