import torch
import torch.nn as nn
import torch.optim as optim

Activation = dict(Relu = torch.nn.ReLU(),
                LeReLu = nn.LeakyReLU(),
                S = nn.Sigmoid(),
                Tanh = nn.Tanh(),
)

optimier = dict(adam = optim.Adam,
                SGD = optim.SGD,
                RMSprop = optim.RMSprop,
)





