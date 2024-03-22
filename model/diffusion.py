import torch
import torch.nn as nn


class MLPDiffusion(nn.Module):					
    def __init__(self, input_dim=13, n_steps=100, num_layers=3, hidden_dim=128):
        super(MLPDiffusion, self).__init__()

        self.num_layers = num_layers

        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()])
        for _ in range(num_layers - 1):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.LeakyReLU())
        self.linears.append(nn.Linear(hidden_dim, input_dim))

        self.embeddings = nn.ModuleList([nn.Embedding(n_steps, hidden_dim) for _ in range(num_layers)])
    
        self._init_net()

    def _init_net(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                nn.init.orthogonal_(layer.weight)
                # nn.init.kaiming_uniform_(layer.weight) # no difference 

    def forward(self, x, t):
        for idx, embedding_layer in enumerate(self.embeddings):	
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)	
        				
        return x


class MLPConditionDiffusion(nn.Module):					
    def __init__(self, input_dim=10, latent_dim = 4, n_steps=100, num_layers=3, hidden_dim=128):
        super(MLPConditionDiffusion, self).__init__()

        self.num_layers = num_layers

        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()])
        for _ in range(num_layers - 1):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.LeakyReLU())
        self.linears.append(nn.Linear(hidden_dim, input_dim))
        # self.linears.append(nn.Tanh())
        
        ############################### condition embeddings ####################################
        self.conditions = nn.ModuleList([nn.Linear(latent_dim, hidden_dim) for _ in range(num_layers)])
        
        self.embeddings = nn.ModuleList([nn.Embedding(n_steps, hidden_dim) for _ in range(num_layers)])

        self._init_net()

    def _init_net(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, x, t, condition):
        for idx in range(self.num_layers):	
            condition_embedding = self.conditions[idx](condition)
            t_embedding = self.embeddings[idx](t)
            x = self.linears[2 * idx](x)
            x += t_embedding + condition_embedding
            x = self.linears[2 * idx + 1](x)

        # x = self.linears[-2](x)
        x = self.linears[-1](x)					
        return x