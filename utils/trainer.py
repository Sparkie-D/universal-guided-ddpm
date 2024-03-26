import torch
import pickle
import os
from tqdm import tqdm

class Trainer(object):
    def __init__(self, 
                 model,
                 train_data,
                 valid_data, 
                 logger,
                 args,
                 device,
                 ) -> None:
        self.train_data = train_data
        self.valid_data = valid_data
        self.logger = logger
        self.device = device
        self.model = model
        self.log_path = args.log_path
        self.model_path = args.model_path
        
    def train_epoch(self, train_loader, epoch):
        total_loss = 0
        n_batch = 0
        for data in train_loader:
            data = data.to(self.device)
            n_batch += 1
            total_loss += self.model.update(data)
        self.logger.add_scalar('train/diffusion loss', total_loss / n_batch, epoch)
        

    def train(self, batch_size=32, num_epoch=1000):  
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)             
        for epoch in tqdm(range(num_epoch), desc='Training'):
            self.train_epoch(train_loader, epoch)
                
            if self.model.save_model_epoch > 0 and (epoch + 1) % self.model.save_model_epoch == 0:
                with open(os.path.join(self.model_path, f'ddpm.pickle'), 'wb') as f:
                    pickle.dump(self.model, f)
