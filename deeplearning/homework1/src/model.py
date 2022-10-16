from turtle import forward
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,args,act) -> None:
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers
        self.activation = act

        self.proj_layer  = nn.Linear(1,self.hidden_dim)
        self.activation = self.activation
        self.hidd_layers = nn.Sequential()
        for _ in range(self.n_layers):
            self.hidd_layers.append(nn.Linear(self.hidden_dim,self.hidden_dim))
            self.hidd_layers.append(self.activation)
        self.out_layer = nn.Linear(self.hidden_dim,1)
        # self.apply(self._init_params)
    
    def forward(self,x):
        x = self.activation(self.proj_layer(x))
        h = self.hidd_layers(x)
        out = self.out_layer(h)
        
        return out

    # def _init_params(self,m):
    #     if isinstance(m,nn.Linear):








